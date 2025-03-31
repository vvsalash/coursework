import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import requests
from tqdm import tqdm
from transformers import Wav2Vec2Model

# Импорт для сохранения аудио
import torchaudio

# Импорт для метрик
from torchinfo import summary
from pesq import pesq
from pystoi import stoi
from scipy.linalg import toeplitz

# ------------------------ Константы и утилиты ------------------------

UTMOS_CKPT_URL = "https://huggingface.co/spaces/sarulab-speech/UTMOS-demo/resolve/main/epoch%3D3-step%3D7459.ckpt"
WAV2VEC_URL = "https://huggingface.co/spaces/sarulab-speech/UTMOS-demo/resolve/main/wav2vec_small.pt"

def download_file(url, filename):
    print(f"Downloading file {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            progress_bar.update(len(chunk))
            f.write(chunk)
    progress_bar.close()

# ------------------------ Модель SSL (Wav2Vec2) ------------------------

def load_ssl_model(ckpt_path="wav2vec_small.pt"):
    """
    Загрузка модели wav2vec2 из transformers.
    Если локального файла нет – загружается по URL (хотя для transformers это не обязательно).
    """
    filepath = os.path.join(os.path.dirname(__file__), ckpt_path)
    if not os.path.exists(filepath):
        download_file(WAV2VEC_URL, filepath)
    # Загружаем предобученную модель из transformers
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    SSL_OUT_DIM = model.config.hidden_size
    return SSL_model(model, SSL_OUT_DIM)

class SSL_model(nn.Module):
    """
    Обёртка для SSL модели.
    Метод forward возвращает признаки (last_hidden_state) из модели wav2vec2.
    """
    def __init__(self, ssl_model, ssl_out_dim) -> None:
        super(SSL_model, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_out_dim = ssl_out_dim

    def forward(self, batch):
        wav = batch["wav"]
        wav = wav.squeeze(1)  # [batch, audio_len]
        outputs = self.ssl_model(wav)
        x = outputs.last_hidden_state  # используем последнее скрытое состояние как признаки
        return {"ssl-feature": x}

    def get_output_dim(self):
        return self.ssl_out_dim

# ------------------------ Дополнительные слои для UTMOS ------------------------

class DomainEmbedding(nn.Module):
    def __init__(self, n_domains, domain_dim) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_domains, domain_dim)
        self.output_dim = domain_dim

    def forward(self, batch):
        return {"domain-feature": self.embedding(batch["domains"])}

    def get_output_dim(self):
        return self.output_dim

class LDConditioner(nn.Module):
    """
    Условие (conditioning) для SSL признаков с помощью встраивания judge.
    """
    def __init__(self, input_dim, judge_dim, num_judges=None):
        super().__init__()
        self.input_dim = input_dim
        self.judge_dim = judge_dim
        self.num_judges = num_judges
        assert num_judges is not None
        self.judge_embedding = nn.Embedding(num_judges, self.judge_dim)
        self.decoder_rnn = nn.LSTM(
            input_size=self.input_dim + self.judge_dim,
            hidden_size=512,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.out_dim = self.decoder_rnn.hidden_size * 2

    def get_output_dim(self):
        return self.out_dim

    def forward(self, x, batch):
        judge_ids = batch["judge_id"]
        if "phoneme-feature" in x:
            concatenated_feature = torch.cat(
                (x["ssl-feature"],
                 x["phoneme-feature"].unsqueeze(1).expand(-1, x["ssl-feature"].size(1), -1)),
                dim=2
            )
        else:
            concatenated_feature = x["ssl-feature"]
        if "domain-feature" in x:
            concatenated_feature = torch.cat(
                (concatenated_feature,
                 x["domain-feature"].unsqueeze(1).expand(-1, concatenated_feature.size(1), -1)),
                dim=2,
            )
        if judge_ids is not None:
            concatenated_feature = torch.cat(
                (concatenated_feature,
                 self.judge_embedding(judge_ids).unsqueeze(1).expand(-1, concatenated_feature.size(1), -1)),
                dim=2,
            )
            decoder_output, _ = self.decoder_rnn(concatenated_feature)
        return decoder_output

class Projection(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, range_clipping=False):
        super(Projection, self).__init__()
        self.range_clipping = range_clipping
        output_dim = 1
        if range_clipping:
            self.proj = nn.Tanh()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), activation, nn.Dropout(0.3), nn.Linear(hidden_dim, output_dim)
        )
        self.output_dim = output_dim

    def forward(self, x, batch):
        output = self.net(x)
        if self.range_clipping:
            return self.proj(output) * 2.0 + 3
        else:
            return output

    def get_output_dim(self):
        return self.output_dim

# ------------------------ Модель для UTMOS (LightningModule) ------------------------

class BaselineLightningModule(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        # Извлекаем дополнительные аргументы (например, 'cfg') если они есть
        self.cfg = kwargs.pop("cfg", None)
        super().__init__(*args, **kwargs)
        self.construct_model()
        self.save_hyperparameters()

    def construct_model(self):
        self.feature_extractors = nn.ModuleList(
            [load_ssl_model(ckpt_path="wav2vec_small.pt"), DomainEmbedding(3, 128)]
        )
        output_dim = sum([fe.get_output_dim() for fe in self.feature_extractors])
        output_layers = [LDConditioner(judge_dim=128, num_judges=3000, input_dim=output_dim)]
        output_dim = output_layers[-1].get_output_dim()
        output_layers.append(
            Projection(input_dim=output_dim, hidden_dim=2048, activation=torch.nn.ReLU(), range_clipping=False)
        )
        self.output_layers = nn.ModuleList(output_layers)

    def forward(self, inputs):
        outputs = {}
        for fe in self.feature_extractors:
            outputs.update(fe(inputs))
        x = outputs
        for layer in self.output_layers:
            x = layer(x, inputs)
        return x

# ------------------------ UTMOSScore ------------------------

class UTMOSScore:
    """
    Класс для предсказания MOS (Mean Opinion Score) по аудиоклипу.
    """
    def __init__(self, device, ckpt_path="epoch=3-step=7459.ckpt"):
        self.device = device
        filepath = os.path.join(os.path.dirname(__file__), ckpt_path)
        if not os.path.exists(filepath):
            download_file(UTMOS_CKPT_URL, filepath)
        # Используем strict=False, чтобы игнорировать несовпадения ключей в state_dict
        self.model = BaselineLightningModule.load_from_checkpoint(filepath, strict=False).eval().to(device)

    def score(self, wavs: torch.Tensor) -> torch.Tensor:
        if len(wavs.shape) == 1:
            out_wavs = wavs.unsqueeze(0).unsqueeze(0)
        elif len(wavs.shape) == 2:
            out_wavs = wavs.unsqueeze(0)
        elif len(wavs.shape) == 3:
            out_wavs = wavs
        else:
            raise ValueError("Dimension of input tensor needs to be <= 3.")
        bs = out_wavs.shape[0]
        batch = {
            "wav": out_wavs,
            "domains": torch.zeros(bs, dtype=torch.int).to(self.device),
            "judge_id": torch.ones(bs, dtype=torch.int).to(self.device) * 288,
        }
        with torch.no_grad():
            output = self.model(batch)
        return output.mean(dim=1).squeeze(1).cpu().detach() * 2 + 3
# ------------------------ Другие метрики ------------------------

class BaseMetric:
    def __init__(self, name="BaseMetric"):
        self.name = name

    def __call__(self, **kwargs):
        raise NotImplementedError

class MACsMetric(BaseMetric):
    """Вычисление количества MACs (Multiply-Accumulate Operations)"""
    def __init__(self, model, example_input, name="MACs") -> None:
        super().__init__(name=name)
        self.model = model
        self.example_input = example_input

    def __call__(self, **kwargs):
        try:
            model_stats = summary(self.model, input_data=self.example_input, verbose=0)
            return model_stats.total_mult_adds
        except Exception as e:
            print(f"[MACsMetric] Error in computing MACs: {e}")
            return None

class FlopsMetric(BaseMetric):
    """Вычисление FLOPs"""
    def __init__(self, macs_metric: MACsMetric, name="FLOPs"):
        super().__init__(name=name)
        self.macs_metric = macs_metric

    def __call__(self, **kwargs):
        macs = self.macs_metric(**kwargs)
        return 2 * macs if macs is not None else None

def PESQ_func(ref_wav, deg_wav, fs=16000, mode="wb"):
    try:
        return pesq(fs, ref_wav, deg_wav, mode)
    except Exception:
        return 1.0

def SSNR(ref_wav, deg_wav, srate=16000, eps=1e-10):
    clean = ref_wav
    proc = deg_wav
    dif = ref_wav - deg_wav
    overall = 10 * np.log10(np.sum(ref_wav**2) / (np.sum(dif**2) + 1e-20))
    winlength = int(np.round(30 * srate / 1000))
    skip = winlength // 4
    MIN_SNR, MAX_SNR = -10, 35
    num_frames = int(ref_wav.shape[0] / skip - (winlength / skip))
    start = 0
    window = 0.5 * (1 - np.cos(2 * np.pi * np.linspace(1, winlength, winlength) / (winlength + 1)))
    seg_snr = []
    for _ in range(int(num_frames)):
        clean_frame = clean[start : start + winlength] * window
        proc_frame = proc[start : start + winlength] * window
        sig_energy = np.sum(clean_frame**2)
        noise_energy = np.sum((clean_frame - proc_frame) ** 2)
        s = 10 * np.log10(sig_energy / (noise_energy + eps) + eps)
        s = np.clip(s, MIN_SNR, MAX_SNR)
        seg_snr.append(s)
        start += skip
    return overall, seg_snr

def wss(ref_wav, deg_wav, srate):
    winlength = round(30 * srate / 1000.0)
    skip = int(np.floor(winlength / 4))
    max_freq = srate / 2
    num_crit = 25
    n_fft = int(2 ** np.ceil(np.log(2 * winlength) / np.log(2)))
    n_fftby2 = int(n_fft / 2)
    cent_freq = [
        50.0, 120, 190, 260, 330, 400, 470, 540, 617.372, 703.378, 798.717, 904.128,
        1020.38, 1148.30, 1288.72, 1442.54, 1610.70, 1794.16, 1993.93, 2211.08, 2446.71,
        2701.97, 2978.04, 3276.17, 3597.63,
    ]
    bandwidth = [
        70.0, 70, 70, 70, 70, 70, 70, 77.3724, 86.0056, 95.3398, 105.411, 116.256,
        127.914, 140.423, 153.823, 168.154, 183.457, 199.776, 217.153, 235.631, 255.255,
        276.072, 298.126, 321.465, 346.136,
    ]
    bw_min = bandwidth[0]
    min_factor = np.exp(-30.0 / (2 * 2.303))
    crit_filter = np.zeros((num_crit, n_fftby2))
    for i in range(num_crit):
        f0 = (cent_freq[i] / max_freq) * n_fftby2
        bw = (bandwidth[i] / max_freq) * n_fftby2
        norm_factor = np.log(bw_min) - np.log(bandwidth[i])
        j = np.arange(n_fftby2)
        crit_filter[i, :] = np.exp(-11 * (((j - np.floor(f0)) / bw) ** 2) + norm_factor)
        crit_filter[i, :] = crit_filter[i, :] * (crit_filter[i, :] > min_factor)
    num_frames = int(ref_wav.shape[0] / skip - (winlength / skip))
    start = 0
    window = 0.5 * (1 - np.cos(2 * np.pi * np.linspace(1, winlength, winlength) / (winlength + 1)))
    distortion = []
    for _ in range(num_frames):
        clean_frame = ref_wav[start : start + winlength] * window
        proc_frame = deg_wav[start : start + winlength] * window
        clean_spec = np.abs(np.fft.fft(clean_frame, n_fft)) ** 2
        proc_spec = np.abs(np.fft.fft(proc_frame, n_fft)) ** 2
        clean_energy = np.array([np.sum(clean_spec[:n_fftby2] * crit_filter[i, :]) for i in range(num_crit)])
        proc_energy = np.array([np.sum(proc_spec[:n_fftby2] * crit_filter[i, :]) for i in range(num_crit)])
        eps_arr = np.ones(num_crit) * 1e-10
        clean_energy_db = 10 * np.log10(np.maximum(clean_energy, eps_arr))
        proc_energy_db = 10 * np.log10(np.maximum(proc_energy, eps_arr))
        clean_slope = clean_energy_db[1:] - clean_energy_db[:-1]
        proc_slope = proc_energy_db[1:] - proc_energy_db[:-1]
        distortion.append(np.sqrt(np.mean((clean_slope - proc_slope) ** 2)))
        start += skip
    return distortion

def lpcoeff(speech_frame, model_order):
    winlength = speech_frame.shape[0]
    R = []
    for k in range(model_order + 1):
        R.append(np.sum(speech_frame[: winlength - k] * speech_frame[k:winlength]))
    a = np.ones(model_order)
    E = np.zeros(model_order + 1)
    rcoeff = np.zeros(model_order)
    E[0] = R[0]
    for i in range(model_order):
        if i == 0:
            sum_term = 0
        else:
            a_past = a[:i]
            sum_term = np.sum(a_past * np.array(R[i:0:-1]))
        rcoeff[i] = (R[i + 1] - sum_term) / E[i]
        a[i] = rcoeff[i]
        if i > 0:
            a[:i] = a_past - rcoeff[i] * a_past[::-1]
        E[i + 1] = (1 - rcoeff[i] ** 2) * E[i]
    acorr = np.array(R, dtype=np.float32)
    refcoeff = np.array(rcoeff, dtype=np.float32)
    a = a * -1
    lpparams = np.array([1] + list(a), dtype=np.float32)
    return acorr, refcoeff, lpparams

def llr(ref_wav, deg_wav, srate):
    winlength = round(30 * srate / 1000.0)
    skip = int(np.floor(winlength / 4))
    P = 10 if srate < 10000 else 16
    num_frames = int(ref_wav.shape[0] / skip - (winlength / skip))
    start = 0
    window = 0.5 * (1 - np.cos(2 * np.pi * np.linspace(1, winlength, winlength) / (winlength + 1)))
    llr_values = []
    for _ in range(num_frames):
        clean_frame = ref_wav[start : start + winlength] * window
        proc_frame = deg_wav[start : start + winlength] * window
        R_clean, _, A_clean = lpcoeff(clean_frame, P)
        _, _, A_proc = lpcoeff(proc_frame, P)
        A_clean = A_clean[None, :]
        A_proc = A_proc[None, :]
        num = A_proc.dot(toeplitz(R_clean)).dot(A_proc.T)
        den = A_clean.dot(toeplitz(R_clean)).dot(A_clean.T)
        llr_values.append(np.squeeze(np.log(num / den)))
        start += skip
    return np.array(llr_values)

def composite_eval(ref_wav, deg_wav):
    alpha = 0.95
    len_ = min(ref_wav.shape[0], deg_wav.shape[0])
    ref_wav = ref_wav[:len_]
    deg_wav = deg_wav[:len_]
    wss_dist_vec = wss(ref_wav, deg_wav, 16000)
    wss_dist = np.mean(sorted(wss_dist_vec)[: int(round(len(wss_dist_vec) * alpha))])
    LLR_dist = llr(ref_wav, deg_wav, 16000)
    llr_mean = np.mean(sorted(LLR_dist)[: int(round(len(LLR_dist) * alpha))])
    overall_snr, seg_snr = SSNR(ref_wav, deg_wav, 16000)
    segSNR = np.mean(seg_snr)
    try:
        pesq_raw = pesq(16000, ref_wav, deg_wav, "wb")
    except Exception:
        pesq_raw = 1.0
    def trim_mos(val):
        return min(max(val, 1), 5)
    Csig = 3.093 - 1.029 * llr_mean + 0.603 * pesq_raw - 0.009 * wss_dist
    Cbak = 1.634 + 0.478 * pesq_raw - 0.007 * wss_dist + 0.063 * segSNR
    Covl = 1.594 + 0.805 * pesq_raw - 0.512 * llr_mean - 0.007 * wss_dist
    return {
        "csig": trim_mos(Csig),
        "cbak": trim_mos(Cbak),
        "covl": trim_mos(Covl),
        "pesq": pesq_raw,
        "ssnr": segSNR,
    }

class CompositeMetric(BaseMetric):
    def __init__(self, name="Composite"):
        super().__init__(name=name)

    def __call__(self, **kwargs):
        ref = kwargs.get("reference_audio")
        deg = kwargs.get("output_audio")
        if ref is None or deg is None:
            return None
        if isinstance(ref, torch.Tensor):
            ref = ref.detach().cpu().numpy().squeeze()
        if isinstance(deg, torch.Tensor):
            deg = deg.detach().cpu().numpy().squeeze()
        return composite_eval(ref, deg)

class PESQMetric(BaseMetric):
    def __init__(self, sample_rate=16000, mode="wb", name="PESQ"):
        super().__init__(name=name)
        self.fs = sample_rate
        self.mode = mode

    def __call__(self, **kwargs):
        ref = kwargs.get("reference_audio")
        deg = kwargs.get("output_audio")
        if ref is None or deg is None:
            return None
        if isinstance(ref, torch.Tensor):
            ref = ref.detach().cpu().numpy().squeeze()
        if isinstance(deg, torch.Tensor):
            deg = deg.detach().cpu().numpy().squeeze()
        return PESQ_func(ref, deg, fs=self.fs, mode=self.mode)

class SSNRMetric(BaseMetric):
    def __init__(self, sample_rate=16000, name="SSNR"):
        super().__init__(name=name)
        self.fs = sample_rate

    def __call__(self, **kwargs):
        ref = kwargs.get("reference_audio")
        deg = kwargs.get("output_audio")
        if ref is None or deg is None:
            return None
        if isinstance(ref, torch.Tensor):
            ref = ref.detach().cpu().numpy().squeeze()
        if isinstance(deg, torch.Tensor):
            deg = deg.detach().cpu().numpy().squeeze()
        overall, seg = SSNR(ref, deg, self.fs)
        return {"overall": overall, "seg_mean": np.mean(seg)}

class STOIMetric(BaseMetric):
    def __init__(self, sample_rate=16000, extended=False, name="STOI"):
        super().__init__(name=name)
        self.fs = sample_rate
        self.extended = extended
        self.stoi_func = stoi

    def __call__(self, **kwargs):
        if self.stoi_func is None:
            return None
        ref = kwargs.get("reference_audio")
        deg = kwargs.get("output_audio")
        if ref is None or deg is None:
            return None
        if isinstance(ref, torch.Tensor):
            ref = ref.detach().cpu().numpy().squeeze()
        if isinstance(deg, torch.Tensor):
            deg = deg.detach().cpu().numpy().squeeze()
        return self.stoi_func(ref, deg, self.fs, extended=self.extended)

class RTFMetric(BaseMetric):
    def __init__(self, name="RTF"):
        super().__init__(name=name)

    def __call__(self, **kwargs):
        infer_time = kwargs.get("infer_time")
        audio_duration = kwargs.get("audio_duration")
        if infer_time is None or audio_duration is None or audio_duration == 0:
            return None
        return infer_time / audio_duration