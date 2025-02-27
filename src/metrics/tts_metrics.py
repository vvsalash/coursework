import time

import numpy as np
import torch
from pesq import pesq
from pystoi import stoi
from scipy.linalg import toeplitz
from torchinfo import summary

from src.metrics.base_metric import BaseMetric


class MACsMetric(BaseMetric):
    """Computing MACs (Multiply-Accumulate Operations)"""

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
    """Computing FLOPs"""

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
    window = 0.5 * (
        1 - np.cos(2 * np.pi * np.linspace(1, winlength, winlength) / (winlength + 1))
    )
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
        50.0,
        120,
        190,
        260,
        330,
        400,
        470,
        540,
        617.372,
        703.378,
        798.717,
        904.128,
        1020.38,
        1148.30,
        1288.72,
        1442.54,
        1610.70,
        1794.16,
        1993.93,
        2211.08,
        2446.71,
        2701.97,
        2978.04,
        3276.17,
        3597.63,
    ]
    bandwidth = [
        70.0,
        70,
        70,
        70,
        70,
        70,
        70,
        77.3724,
        86.0056,
        95.3398,
        105.411,
        116.256,
        127.914,
        140.423,
        153.823,
        168.154,
        183.457,
        199.776,
        217.153,
        235.631,
        255.255,
        276.072,
        298.126,
        321.465,
        346.136,
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
    window = 0.5 * (
        1 - np.cos(2 * np.pi * np.linspace(1, winlength, winlength) / (winlength + 1))
    )
    distortion = []
    for _ in range(num_frames):
        clean_frame = ref_wav[start : start + winlength] * window
        proc_frame = deg_wav[start : start + winlength] * window
        clean_spec = np.abs(np.fft.fft(clean_frame, n_fft)) ** 2
        proc_spec = np.abs(np.fft.fft(proc_frame, n_fft)) ** 2
        clean_energy = np.array(
            [np.sum(clean_spec[:n_fftby2] * crit_filter[i, :]) for i in range(num_crit)]
        )
        proc_energy = np.array(
            [np.sum(proc_spec[:n_fftby2] * crit_filter[i, :]) for i in range(num_crit)]
        )
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
    window = 0.5 * (
        1 - np.cos(2 * np.pi * np.linspace(1, winlength, winlength) / (winlength + 1))
    )
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
