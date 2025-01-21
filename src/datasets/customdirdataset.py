import torchaudio

from src.utils.io_utils import ROOT_PATH


class CustomTextDirDataset:
    def __init__(self, path):
        self.path = ROOT_PATH / path

    def get_texts(self):
        texts = []
        for path in (self.path / "transcriptions").iterdir():
            assert path.suffix == ".txt"

            sample = {}
            with path.open() as file:
                sample["text"] = file.read()
            sample["filename"] = path.stem

            texts.append(sample)
        return texts


class CustomAudioDirDataset:
    def __init__(self, path):
        self.path = ROOT_PATH / path

    def get_audios(self):
        audios = []
        for path in (self.path / "utterences").iterdir():
            assert path.suffix in {".wav", ".mp3", ".m4a", ".flac"}

            sample = {}
            target_sr = 22050
            audio, sr = torchaudio.load(path)
            audio = audio[0:1, :]
            if sr != target_sr:
                audio = torchaudio.functional.resample(audio, sr, target_sr)
            sample["audio"] = audio
            sample["filename"] = path.stem
            audios.append(sample)

        return audios
