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
)
from src.utils.init_utils import set_random_seed

warnings.filterwarnings("ignore", category=UserWarning)


class FastSpeech2Wrapper(BaseFastSpeech2):
    """
    Обёртка для FastSpeech2, которая гарантирует,
    что входной текст преобразуется в нужный формат.

    Если вызывается encode_text с одиночной строкой или списком строк,
    то метод encode_input преобразует их в список словарей с ключом "txt".
    """

    def encode_text(self, text):
        # Если вход – одиночная строка, оборачиваем её в список
        if isinstance(text, str):
            text = [text]
        return super().encode_text(text)

    def encode_input(self, data):
        # Если ключ "txt" содержит список строк, преобразуем их в список словарей
        if (
            "txt" in data
            and isinstance(data["txt"], list)
            and data["txt"]
            and isinstance(data["txt"][0], str)
        ):
            data["txt"] = [{"txt": t} for t in data["txt"]]
        return super().encode_input(data)


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
        infer_time = time.time() - start_time

        wav = hifigan.decode_batch(mel_outputs).cpu()
        if wav.dim() == 3:
            waveform = wav[0]
        elif wav.dim() == 2:
            waveform = wav
        elif wav.dim() == 1:
            waveform = wav.unsqueeze(0)
        else:
            print(f"Unexpected waveform dimension: {wav.shape}")
            continue

        output_path = output_dir / f"sample_{idx}.wav"
        torchaudio.save(str(output_path), waveform, config.tts.sample_rate)
        print(f"Saved synthesized audio to {output_path}")

        audio_duration = waveform.shape[-1] / config.tts.sample_rate

        print("Metrics:")

        metrics_result = {
            "composite": comp_metric(output_audio=waveform, reference_audio=waveform),
            "pesq": pesq_metric(output_audio=waveform, reference_audio=waveform),
            "ssnr": ssnr_metric(output_audio=waveform, reference_audio=waveform),
            "stoi": stoi_metric(output_audio=waveform, reference_audio=waveform),
            "rtf": rtf_metric(infer_time=infer_time, audio_duration=audio_duration),
        }
        print(f"Metrics for sample {idx}: {metrics_result}")
    print("Done TTS inference.")


if __name__ == "__main__":
    main()
