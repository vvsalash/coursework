import contextlib
import time
import warnings
from pathlib import Path

import hydra
import torch
import torchaudio
from omegaconf import DictConfig
from speechbrain.inference.TTS import FastSpeech2 as _FastSpeech2
from speechbrain.inference.vocoders import HIFIGAN

from src.datasets.ljspeech_dataset import LJspeechTTSDataset
from src.metrics.tts_metrics import (
    CompositeMetric,
    PESQMetric,
    SSNRMetric,
    STOIMetric,
    UTMOSScore,
)
from src.utils.init_utils import set_random_seed


class CUDATimer(contextlib.AbstractContextManager):
    """Measures *synchronous* wall‑time regardless of CUDA/CPU."""

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.interval = time.perf_counter() - self.start
        return False  # do not suppress exceptions


class RTFTracker:
    """Aggregates inference time vs produced audio duration."""

    def __init__(self):
        self.reset()

    def add(self, infer_sec: float, n_samples: int, sr: int):
        self._infer += infer_sec
        self._audio += n_samples / sr

    @property
    def value(self):
        return float("inf") if self._audio == 0 else self._infer / self._audio

    def reset(self):
        self._infer = 0.0
        self._audio = 0.0


class FastSpeech2Wrapper(_FastSpeech2):
    def encode_text(self, text):
        if isinstance(text, str):
            text = [text]
        return super().encode_text(text)

    def encode_input(self, data):
        if (
            "txt" in data
            and isinstance(data["txt"], list)
            and data["txt"]
            and isinstance(data["txt"][0], str)
        ):
            data["txt"] = [{"txt": t} for t in data["txt"]]
        return super().encode_input(data)


def stream_vocoder_decode(
    mel_outputs: torch.Tensor,
    hifigan: HIFIGAN,
    chunk_size: int,
    overlap_frames: int,
    hop_length: int,
    device: torch.device,
    sample_rate: int,
):

    mel = mel_outputs[0]
    T = mel.shape[0]
    wave_chunks = []
    rtf_chunks = []

    overlap_samples = overlap_frames * hop_length

    starts = list(range(0, T, chunk_size))
    num_chunks = len(starts)

    for i, start in enumerate(starts):
        chunk_start = max(0, start - overlap_frames) if i else 0
        chunk_end = min(T, start + chunk_size + overlap_frames)
        mel_chunk = mel[chunk_start:chunk_end].unsqueeze(0).to(device)

        with CUDATimer() as timer:
            wav_chunk = hifigan.decode_batch(mel_chunk).cpu()
        infer_time = timer.interval

        if wav_chunk.dim() == 3:
            wav_chunk = wav_chunk[0]
        elif wav_chunk.dim() == 1:
            wav_chunk = wav_chunk.unsqueeze(0)

        if i == 0 and num_chunks > 1:
            wav_trim = wav_chunk[:, :-overlap_samples]
        elif i == num_chunks - 1 and i != 0:
            wav_trim = wav_chunk[:, overlap_samples:]
        elif i != 0:
            wav_trim = wav_chunk[:, overlap_samples:-overlap_samples]
        else:
            wav_trim = wav_chunk

        wave_chunks.append(wav_trim)

        audio_dur = wav_chunk.shape[-1] / sample_rate
        rtf_chunk = infer_time / audio_dur if audio_dur > 0 else float("inf")
        rtf_chunks.append(rtf_chunk)
        print(f"      · chunk {i+1}/{num_chunks}: time={infer_time:.3f}s  RTF={rtf_chunk:.3f}")

    waveform = torch.cat(wave_chunks, dim=-1)
    return waveform, rtf_chunks

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference_tts")
def main(cfg: DictConfig):
    set_random_seed(cfg.tts.seed)
    device = (
        torch.device(cfg.tts.device)
        if cfg.tts.device != "auto"
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("Loading FastSpeech2…")
    fastspeech2 = FastSpeech2Wrapper.from_hparams(
        source=cfg.tts.fastspeech2.source,
        savedir=cfg.tts.fastspeech2.savedir,
        run_opts={"device": str(device)},
    )

    print("Loading HiFi‑GAN…")
    hifigan = HIFIGAN.from_hparams(
        source=cfg.tts.hifigan.source,
        savedir=cfg.tts.hifigan.savedir,
        run_opts={"device": str(device)},
    )

    out_dir = Path(cfg.tts.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = LJspeechTTSDataset(part="test", limit=cfg.tts.limit)

    composite_metric = CompositeMetric()
    pesq_metric = PESQMetric(sample_rate=cfg.tts.sample_rate, mode="wb")
    stoi_metric = STOIMetric(sample_rate=cfg.tts.sample_rate)
    ssnr_metric = SSNRMetric(sample_rate=cfg.tts.sample_rate)
    mos_metric = UTMOSScore(device)

    rtf_global = RTFTracker()

    # chunk parameters
    chunk_size = cfg.tts.get("chunk_size", 200)
    overlap_frames = cfg.tts.get("overlap_frames", 10)
    hop_length = cfg.tts.get("hop_length", 240)

    print(f"Streaming mode: chunk={chunk_size}  overlap={overlap_frames}")

    for idx, sample in enumerate(dataset):
        text = (sample.get("text") or "").strip()
        if not text:
            continue
        print(f"\n[{idx}] '{text[:60]}…'")

        with CUDATimer() as t_enc:
            mel_outputs, *_ = fastspeech2.encode_text(text)
        t_fastspeech = t_enc.interval

        waveform, rtf_chunks = stream_vocoder_decode(
            mel_outputs,
            hifigan,
            chunk_size,
            overlap_frames,
            hop_length,
            device,
            cfg.tts.sample_rate,
        )
        waveform = waveform.squeeze(0) if waveform.dim() == 3 else waveform

        total_infer_time = t_fastspeech + sum(rtf * (chunk_size * hop_length) / cfg.tts.sample_rate for rtf in rtf_chunks)
        rtf_global.add(total_infer_time, waveform.shape[-1], cfg.tts.sample_rate)


        wav_path = out_dir / f"sample_{idx}.wav"
        torchaudio.save(str(wav_path), waveform, cfg.tts.sample_rate)
        print(f"   saved → {wav_path.name}\n   sentence‑RTF = {total_infer_time / (waveform.shape[-1] / cfg.tts.sample_rate):.3f}")


        mos_val = mos_metric.score(waveform).item()
        pesq_val = pesq_metric(output_audio=waveform, reference_audio=waveform)
        stoi_val = stoi_metric(output_audio=waveform, reference_audio=waveform)
        ssnr_val = ssnr_metric(output_audio=waveform, reference_audio=waveform)["overall"]
        cm = composite_metric(output_audio=waveform, reference_audio=waveform)["covl"]
        print(f"   MOS={mos_val:.2f}  PESQ={pesq_val:.2f}  STOI={stoi_val:.2f}  SSNR={ssnr_val:.2f}  COVL={cm:.2f}")


    print("\n===============================")
    print(f"Global streaming RTF = {rtf_global.value:.3f}")


if __name__ == "__main__":
    main()
