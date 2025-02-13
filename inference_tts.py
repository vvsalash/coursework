import warnings
from pathlib import Path

import hydra
import torch
import torchaudio
from omegaconf import DictConfig
from speechbrain.inference.TTS import FastSpeech2
from speechbrain.inference.vocoders import HIFIGAN

from src.datasets.ljspeech_dataset import LJspeechTTSDataset
from src.utils.init_utils import set_random_seed

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference_tts")
def main(config: DictConfig):
    set_random_seed(config.tts.seed)
    device = (
        config.tts.device
        if config.tts.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("Loading FastSpeech2...")
    fastspeech2 = FastSpeech2.from_hparams(
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

    for idx, sample in enumerate(dataset):
        text = sample.get("text", "")
        if not text:
            print(f"Sample {idx} does not have text. Skipping.")
            continue

        print(f"Synthesizing audio for sample {idx}: {text}")
        (
            mel_outputs,
            mel_lens,
            durations,
            attention,
            stop_tokens,
        ) = fastspeech2.encode_text(text)
        wav = hifigan.decode_batch(mel_outputs)
        wav = wav.squeeze(0).cpu()
        output_path = output_dir / f"sample_{idx}.wav"
        torchaudio.save(str(output_path), wav.unsqueeze(0), config.tts.sample_rate)
        print(f"Saved synthesized audio to {output_path}")


if __name__ == "__main__":
    main()
