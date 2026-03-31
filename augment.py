import random
from typing import Any, Optional

import numpy as np
import torch
from scipy.signal import fftconvolve, resample


EPS = 1e-8


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def load_hf_audio_pool(dataset_cfg: Any) -> list[np.ndarray]:
    dataset_id = _cfg_get(dataset_cfg, "dataset_id", "")
    if not dataset_id:
        return []

    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise ImportError("datasets is required to load HF noise/RIR sources.") from exc

    split = _cfg_get(dataset_cfg, "split", "train")
    audio_column = _cfg_get(dataset_cfg, "audio_column", "audio")
    cast_sampling_rate = _cfg_get(dataset_cfg, "cast_column_to_sampling_rate", 16000)
    max_items_cfg = _cfg_get(dataset_cfg, "max_items", None)

    ds = load_dataset(dataset_id, split=split)
    if cast_sampling_rate is not None:
        ds = ds.cast_column(
            audio_column,
            Audio(sampling_rate=cast_sampling_rate),
        )
    if max_items_cfg is not None:
        max_items = min(int(max_items_cfg), len(ds))
        ds = ds.select(range(max_items))

    pool: list[np.ndarray] = []
    for row in ds:
        audio_value = row.get(audio_column)
        if audio_value is None:
            continue
        if isinstance(audio_value, dict) and "array" in audio_value:
            arr = np.asarray(audio_value["array"], dtype=np.float32)
        else:
            arr = np.asarray(audio_value, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        if arr.size == 0:
            continue
        pool.append(arr)

    return pool


class VTLP:
    """
    Vocal Tract Length Perturbation (VTLP) applied directly to
    log-mel spectrograms (e.g. processor output features).
    """

    @staticmethod
    def get_scale_factors(
        n_mels: int,
        sampling_rate: int,
        fhi: float = 4800.0,
        alpha: float = 0.9,
    ) -> np.ndarray:
        freqs = np.linspace(0, 1, n_mels) * sampling_rate / 2.0
        half_sr = sampling_rate / 2.0
        scale = fhi * min(alpha, 1.0)
        f_boundary = scale / alpha

        factors = np.where(
            freqs <= f_boundary,
            freqs * alpha,
            half_sr
            - (half_sr - scale) / (half_sr - scale / alpha) * (half_sr - freqs),
        )
        factors *= (n_mels - 1) / factors.max()
        return factors

    @staticmethod
    def warp_mel(mel_spec: np.ndarray, factors: np.ndarray) -> np.ndarray:
        n_mels, _ = mel_spec.shape
        warped = np.zeros_like(mel_spec)

        for i in range(n_mels):
            if i == 0 or i + 1 >= n_mels:
                warped[i, :] += mel_spec[i, :]
            else:
                pos = int(np.floor(factors[i]))
                frac = factors[i] - pos
                warped[pos, :] += (1.0 - frac) * mel_spec[i, :]
                if pos + 1 < n_mels:
                    warped[pos + 1, :] += frac * mel_spec[i, :]

        return warped

    def __call__(
        self,
        input_features: "np.ndarray | torch.Tensor",
        sampling_rate: int = 16000,
        alpha: float = 0.9,
        fhi: float = 4800.0,
    ) -> "np.ndarray | torch.Tensor":
        is_tensor = isinstance(input_features, torch.Tensor)
        if is_tensor:
            device = input_features.device
            arr = input_features.detach().cpu().numpy()
        else:
            arr = np.asarray(input_features, dtype=np.float32)

        single = arr.ndim == 2
        if single:
            arr = arr[np.newaxis, ...]

        batch_size, n_mels, _ = arr.shape
        factors = self.get_scale_factors(n_mels, sampling_rate, fhi=fhi, alpha=alpha)
        warped_batch = np.empty_like(arr)
        for b in range(batch_size):
            warped_batch[b] = self.warp_mel(arr[b], factors)

        if single:
            warped_batch = warped_batch.squeeze(0)

        if is_tensor:
            return torch.from_numpy(warped_batch).to(device)
        return warped_batch


class SpecAugment:
    POLICIES = {
        "LB": {"F": 27, "m_F": 1, "T": 100, "p": 1.0, "m_T": 1},
        "LD": {"F": 27, "m_F": 2, "T": 100, "p": 1.0, "m_T": 2},
        "SM": {"F": 15, "m_F": 2, "T": 70, "p": 0.2, "m_T": 2},
        "SS": {"F": 27, "m_F": 2, "T": 70, "p": 0.2, "m_T": 2},
    }

    def __init__(self, policy: str = "LB"):
        if policy not in self.POLICIES:
            raise ValueError(f"Unknown policy '{policy}'. Choose from {list(self.POLICIES)}")
        params = self.POLICIES[policy]
        self.F: int = params["F"]
        self.m_F: int = params["m_F"]
        self.T: int = params["T"]
        self.p: float = params["p"]
        self.m_T: int = params["m_T"]

    def freq_mask(self, mel: np.ndarray) -> np.ndarray:
        n_mels = mel.shape[0]
        for _ in range(self.m_F):
            f = int(np.random.uniform(0, self.F))
            f0 = random.randint(0, max(0, n_mels - f))
            mel[f0 : f0 + f, :] = 0
        return mel

    def time_mask(self, mel: np.ndarray) -> np.ndarray:
        tau = mel.shape[1]
        for _ in range(self.m_T):
            t = int(np.random.uniform(0, min(self.T, int(self.p * tau))))
            if t == 0:
                continue
            t0 = random.randint(0, max(0, tau - t))
            mel[:, t0 : t0 + t] = 0
        return mel

    def __call__(self, mel: np.ndarray) -> np.ndarray:
        mel = mel.copy()
        mel = self.freq_mask(mel)
        mel = self.time_mask(mel)
        return mel


class WaveformAugment:
    def __init__(
        self,
        config: Any,
        noise_pool: Optional[list[np.ndarray]] = None,
        rir_pool: Optional[list[np.ndarray]] = None,
    ):
        self.config = config
        self.noise_pool = noise_pool or []
        self.rir_pool = rir_pool or []

    def _safe_waveform(self, waveform: np.ndarray) -> np.ndarray:
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if waveform.size == 0:
            return waveform
        peak = np.max(np.abs(waveform))
        if peak > 1.0:
            waveform = waveform / (peak + EPS)
        gain_min = _cfg_get(self.config, "gain_min", None)
        gain_max = _cfg_get(self.config, "gain_max", None)
        if gain_min is not None and gain_max is not None:
            waveform = waveform * float(np.random.uniform(gain_min, gain_max))
            peak = np.max(np.abs(waveform))
            if peak > 1.0:
                waveform = waveform / (peak + EPS)
        return waveform.astype(np.float32)

    def add_rir(self, waveform: np.ndarray, rir: np.ndarray) -> np.ndarray:
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
        rir = np.asarray(rir, dtype=np.float32).reshape(-1)
        if waveform.size == 0 or rir.size == 0:
            return waveform

        rir = rir - np.mean(rir)
        rir = rir / (np.sqrt(np.sum(rir**2)) + EPS)

        conv = fftconvolve(waveform, rir, mode="full")
        peak_idx = int(np.argmax(np.abs(rir)))
        start = peak_idx
        end = start + waveform.shape[0]
        if end > conv.shape[0]:
            padded = np.zeros(end, dtype=np.float32)
            padded[: conv.shape[0]] = conv.astype(np.float32)
            conv = padded
        out = conv[start:end]
        return self._safe_waveform(out)

    def add_noise(self, waveform: np.ndarray, noise: np.ndarray) -> np.ndarray:
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
        noise = np.asarray(noise, dtype=np.float32).reshape(-1)
        if waveform.size == 0 or noise.size == 0:
            return waveform

        target_len = waveform.shape[0]
        if noise.shape[0] < target_len:
            repeats = int(np.ceil(target_len / max(1, noise.shape[0])))
            noise = np.tile(noise, repeats)
        if noise.shape[0] > target_len:
            start = random.randint(0, noise.shape[0] - target_len)
            noise = noise[start : start + target_len]

        min_snr_db = float(_cfg_get(self.config, "min_snr_db", 5.0))
        max_snr_db = float(_cfg_get(self.config, "max_snr_db", 30.0))
        snr_db = float(np.random.uniform(min_snr_db, max_snr_db))

        waveform_power = float(np.mean(waveform**2) + EPS)
        noise_power = float(np.mean(noise**2) + EPS)
        target_noise_power = waveform_power / (10 ** (snr_db / 10.0))
        scale = np.sqrt(target_noise_power / noise_power)
        mixed = waveform + (noise * scale).astype(np.float32)
        return self._safe_waveform(mixed)

    def time_stretch(self, waveform: np.ndarray, factor: float) -> np.ndarray:
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if waveform.size == 0:
            return waveform
        if factor <= 0:
            return waveform

        stretched_len = int(round(waveform.shape[0] / factor))
        max_len = int(_cfg_get(self.config, "max_stretch_samples", waveform.shape[0]))
        if stretched_len <= 1 or stretched_len > max_len:
            return waveform

        stretched = resample(waveform, stretched_len).astype(np.float32)
        return self._safe_waveform(stretched)

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        out = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if out.size == 0:
            return out

        if random.random() < float(_cfg_get(self.config, "time_stretch_prob", 0.0)):
            min_speed = float(_cfg_get(self.config, "min_speed", 0.9))
            max_speed = float(_cfg_get(self.config, "max_speed", 1.1))
            speed = float(np.random.uniform(min_speed, max_speed))
            out = self.time_stretch(out, speed)

        if self.rir_pool and random.random() < float(_cfg_get(self.config, "rir_prob", 0.0)):
            rir = random.choice(self.rir_pool)
            out = self.add_rir(out, rir)

        if self.noise_pool and random.random() < float(_cfg_get(self.config, "noise_prob", 0.0)):
            noise = random.choice(self.noise_pool)
            out = self.add_noise(out, noise)

        return out.astype(np.float32)


class Augmentor:
    def __init__(
        self,
        config: Any,
        noise_pool: Optional[list[np.ndarray]] = None,
        rir_pool: Optional[list[np.ndarray]] = None,
    ):
        aug_cfg = _cfg_get(config, "augmentation", config)
        waveform_cfg = _cfg_get(aug_cfg, "waveform", {})
        spec_cfg = _cfg_get(aug_cfg, "spec", {})
        vtlp_cfg = _cfg_get(aug_cfg, "vtlp", {})
        noise_dataset_cfg = _cfg_get(aug_cfg, "noise_dataset", None)
        rir_dataset_cfg = _cfg_get(aug_cfg, "rir_dataset", None)

        if noise_pool is None and noise_dataset_cfg is not None:
            noise_pool = load_hf_audio_pool(noise_dataset_cfg)

        if rir_pool is None and rir_dataset_cfg is not None:
            rir_pool = load_hf_audio_pool(rir_dataset_cfg)

        self.waveform_aug = WaveformAugment(
            config=waveform_cfg,
            noise_pool=noise_pool,
            rir_pool=rir_pool,
        )
        self.vtlp = VTLP()
        self.spec_augment = SpecAugment(policy=_cfg_get(spec_cfg, "policy", "LB"))
        self.vtlp_cfg = vtlp_cfg
        self.spec_cfg = spec_cfg

    def augment_waveform(self, waveform: np.ndarray) -> np.ndarray:
        return self.waveform_aug(waveform)

    def augment_features(
        self,
        input_features: "np.ndarray | torch.Tensor",
        sampling_rate: int = 16000,
    ) -> "np.ndarray | torch.Tensor":
        out = input_features
        if random.random() < float(_cfg_get(self.vtlp_cfg, "prob", 0.0)):
            alpha = float(
                np.random.uniform(
                    _cfg_get(self.vtlp_cfg, "alpha_min", 0.9),
                    _cfg_get(self.vtlp_cfg, "alpha_max", 1.1),
                )
            )
            out = self.vtlp(
                out,
                sampling_rate=sampling_rate,
                alpha=alpha,
                fhi=float(_cfg_get(self.vtlp_cfg, "fhi", 4800.0)),
            )

        if random.random() < float(_cfg_get(self.spec_cfg, "prob", 0.0)):
            out = self._apply_specaugment(out)
        return out

    def _apply_specaugment(
        self,
        input_features: "np.ndarray | torch.Tensor",
    ) -> "np.ndarray | torch.Tensor":
        is_tensor = isinstance(input_features, torch.Tensor)
        if is_tensor:
            device = input_features.device
            arr = input_features.detach().cpu().numpy()
        else:
            arr = np.asarray(input_features, dtype=np.float32)

        single = arr.ndim == 2
        if single:
            arr = arr[np.newaxis, ...]

        augmented = np.empty_like(arr)
        for i in range(arr.shape[0]):
            augmented[i] = self.spec_augment(arr[i])

        if single:
            augmented = augmented.squeeze(0)

        if is_tensor:
            return torch.from_numpy(augmented).to(device)
        return augmented

    def __call__(
        self,
        waveform: np.ndarray,
        input_features: Optional["np.ndarray | torch.Tensor"] = None,
        sampling_rate: int = 16000,
    ):
        waveform_aug = self.augment_waveform(waveform)
        if input_features is None:
            return waveform_aug
        features_aug = self.augment_features(input_features, sampling_rate=sampling_rate)
        return waveform_aug, features_aug
