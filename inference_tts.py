import time
import warnings
from pathlib import Path

import hydra
import torch
import torchaudio
from omegaconf import DictConfig
from speechbrain.inference.TTS import FastSpeech2 as BaseFastSpeech2
from speechbrain.inference.vocoders import HIFIGAN

from src.datasets.ljspeech_dataset import LJspeechTTSDataset
from src.metrics.tts_metrics import (
    CompositeMetric,
    PESQMetric,
    RTFMetric,
    SSNRMetric,
    STOIMetric,
    UTMOSScore
)
from src.utils.init_utils import set_random_seed

warnings.filterwarnings("ignore", category=UserWarning)


class FastSpeech2Wrapper(BaseFastSpeech2):
    """
    Обёртка для FastSpeech2, которая гарантирует,
    что входной текст преобразуется в нужный формат.
    """

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


def chunked_vocoder_decode(mel_outputs, hifigan, chunk_size, overlap_frames, hop_length):
    """
    Разбивает мел-спектрограмму на чанки с перекрытием и декодирует каждый через вокодер.

    Аргументы:
        mel_outputs: torch.Tensor формы (1, T, n_mels)
        hifigan: вокодер с методом decode_batch
        chunk_size: число кадров мел-спектрограммы на один чанк (без учета перекрытия)
        overlap_frames: число перекрывающих кадров с каждой стороны
        hop_length: число аудиосемплов, соответствующее одному кадру (frame shift)

    Возвращает:
        waveform: результирующий аудиосигнал после последовательного декодирования
    """
    mel = mel_outputs[0]
    T = mel.shape[0]
    waveform_chunks = []
    overlap_samples = overlap_frames * hop_length

    starts = list(range(0, T, chunk_size))
    num_chunks = len(starts)

    for i, start in enumerate(starts):
        if i == 0:
            chunk_start = 0
            chunk_end = min(T, start + chunk_size + overlap_frames)
        else:
            chunk_start = max(0, start - overlap_frames)
            chunk_end = min(T, start + chunk_size + overlap_frames)
        mel_chunk = mel[chunk_start:chunk_end].unsqueeze(0)
        wav_chunk = hifigan.decode_batch(mel_chunk).cpu()
        if wav_chunk.dim() == 3:
            wav_chunk = wav_chunk[0]
        elif wav_chunk.dim() == 1:
            wav_chunk = wav_chunk.unsqueeze(0)
        if i == 0 and num_chunks > 1:
            wav_chunk = wav_chunk[:, : -overlap_samples]
        elif i == num_chunks - 1 and i != 0:
            wav_chunk = wav_chunk[:, overlap_samples:]
        elif i != 0:
            wav_chunk = wav_chunk[:, overlap_samples: -overlap_samples]

        waveform_chunks.append(wav_chunk)
    waveform = torch.cat(waveform_chunks, dim=-1)
    return waveform


@hydra.main(version_base=None, config_path="src/configs", config_name="inference_tts")
def main(config: DictConfig):
    set_random_seed(config.tts.seed)
    device = (
        config.tts.device
        if config.tts.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("Loading FastSpeech2...")
    fastspeech2 = FastSpeech2Wrapper.from_hparams(
        source=config.tts.fastspeech2.source,
        savedir=config.tts.fastspeech2.savedir,
        run_opts={"device": device},
    )

    print("Loading HIFIGAN...")
    hifigan = HIFIGAN.from_hparams(
        source=config.tts.hifigan.source,
        savedir=config.tts.hifigan.savedir,
        run_opts={"device": device},
    )

    output_dir = Path(config.tts.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = LJspeechTTSDataset(
        part="test",
        limit=config.tts.limit,
    )

    comp_metric = CompositeMetric()
    pesq_metric = PESQMetric(sample_rate=config.tts.sample_rate, mode="wb")
    ssnr_metric = SSNRMetric(sample_rate=config.tts.sample_rate)
    stoi_metric = STOIMetric(sample_rate=config.tts.sample_rate)
    rtf_metric = RTFMetric()

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    mos_metric = UTMOSScore(device)

    chunk_size = config.tts.get("chunk_size", 100)
    overlap_frames = config.tts.get("overlap_frames", 10)
    hop_length = config.tts.get("hop_length", 240)

    result_mos = 0
    length = 0
    for idx, sample in enumerate(dataset):
        text = sample.get("text", "").strip()
        if not text:
            print(f"Sample {idx} does not have text. Skipping.")
            continue

        print(f"Synthesizing audio for sample {idx}: {text}")
        start_time = time.time()
        try:
            result = fastspeech2.encode_text(text)
            if len(result) == 5:
                mel_outputs, mel_lens, durations, attention, stop_tokens = result
            elif len(result) == 4:
                mel_outputs, mel_lens, durations, attention = result
            else:
                raise ValueError(f"Unexpected return length: {len(result)}")
        except Exception as e:
            print(f"Error during text encoding: {e}")
            continue
        fastspeech_infer_time = time.time() - start_time

        start_time = time.time()
        wav_baseline = hifigan.decode_batch(mel_outputs).cpu()
        if wav_baseline.dim() == 3:
            waveform_baseline = wav_baseline[0]
        elif wav_baseline.dim() == 2:
            waveform_baseline = wav_baseline
        elif wav_baseline.dim() == 1:
            waveform_baseline = wav_baseline.unsqueeze(0)
        baseline_vocoder_time = time.time() - start_time

        baseline_total_time = fastspeech_infer_time + baseline_vocoder_time

        baseline_output_path = output_dir / f"sample_{idx}_baseline.wav"
        torchaudio.save(str(baseline_output_path), waveform_baseline, config.tts.sample_rate)
        print(f"Saved baseline synthesized audio to {baseline_output_path}")

        start_time = time.time()
        waveform_chunked = chunked_vocoder_decode(mel_outputs, hifigan, chunk_size, overlap_frames, hop_length)
        chunked_vocoder_time = time.time() - start_time

        chunked_total_time = fastspeech_infer_time + chunked_vocoder_time

        chunked_output_path = output_dir / f"sample_{idx}_chunked.wav"
        torchaudio.save(str(chunked_output_path), waveform_chunked, config.tts.sample_rate)
        print(f"Saved chunked synthesized audio to {chunked_output_path}")

        audio_duration_baseline = waveform_baseline.shape[-1] / config.tts.sample_rate
        audio_duration_chunked = waveform_chunked.shape[-1] / config.tts.sample_rate

        metrics_baseline = {
            "composite": comp_metric(output_audio=waveform_baseline, reference_audio=waveform_baseline),
            "pesq": pesq_metric(output_audio=waveform_baseline, reference_audio=waveform_baseline),
            "ssnr": ssnr_metric(output_audio=waveform_baseline, reference_audio=waveform_baseline),
            "stoi": stoi_metric(output_audio=waveform_baseline, reference_audio=waveform_baseline),
            "rtf": rtf_metric(infer_time=baseline_total_time, audio_duration=audio_duration_baseline),
            "mos": mos_metric.score(waveform_baseline)
        }
        result_mos += metrics_baseline['mos']
        length += 1
        print(f"Baseline Metrics for sample {idx}: {metrics_baseline}")

        metrics_chunked = {
            "composite": comp_metric(output_audio=waveform_chunked, reference_audio=waveform_chunked),
            "pesq": pesq_metric(output_audio=waveform_chunked, reference_audio=waveform_chunked),
            "ssnr": ssnr_metric(output_audio=waveform_chunked, reference_audio=waveform_chunked),
            "stoi": stoi_metric(output_audio=waveform_chunked, reference_audio=waveform_chunked),
            "rtf": rtf_metric(infer_time=chunked_total_time, audio_duration=audio_duration_chunked),
            "mos": mos_metric.score(waveform_chunked)
        }
        print(f"Chunked Metrics for sample {idx}: {metrics_chunked}")

    print(f"Result avg MOS = {result_mos / length}")
    print("Done TTS inference experiments.")


if __name__ == "__main__":
    main()
