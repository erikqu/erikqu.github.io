"""Shared LibriSpeech audio + log-mel helpers for mel blog figures."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.io import wavfile
from scipy.signal import istft as sp_istft, resample_poly, stft as sp_stft

SR = 22050
N_FFT = 1024
HOP = 256
N_MELS = 80
FMIN = 0.0
FMAX = SR / 2
ROOT = Path(__file__).resolve().parent
SOURCE_WAV = ROOT / "outputs" / "audio" / "00_librispeech_source.wav"


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_basis(n_mels: int = N_MELS, n_fft: int = N_FFT, sr: int = SR) -> np.ndarray:
    """Slaney-style mel filterbank (librosa-compatible layout)."""
    n_freqs = n_fft // 2 + 1
    mel_pts = np.linspace(_hz_to_mel(FMIN), _hz_to_mel(FMAX), n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_freqs))
    for m in range(1, n_mels + 1):
        left, center, right = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        if center == left or right == center:
            continue
        for k in range(left, center):
            if 0 <= k < n_freqs:
                fb[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            if 0 <= k < n_freqs:
                fb[m - 1, k] = (right - k) / (right - center)
    # Slaney normalization
    enorm = 2.0 / (hz_pts[2 : n_mels + 2] - hz_pts[:n_mels])
    fb *= enorm[:, np.newaxis]
    return fb


def _write_wav(path: Path, y: np.ndarray, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.clip(y, -1.0, 1.0)
    wavfile.write(path, sr, (y * 32767).astype(np.int16))


def _fetch_librispeech_sample() -> np.ndarray:
    """Fetch one small LibriSpeech test-clean sample without downloading the corpus."""
    from datasets import Audio, load_dataset

    ds = load_dataset("Cidoyi/librispeech-samples", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    ex = next(iter(ds))
    y, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = resample_poly(y, SR, sr)
    y = y.astype(float)
    y -= float(np.mean(y))
    y /= np.max(np.abs(y)) or 1.0
    return 0.85 * y


def load_speech_demo(duration: float = 1.8) -> np.ndarray:
    """Return a cached public LibriSpeech-derived speech clip."""
    if not SOURCE_WAV.exists():
        _write_wav(SOURCE_WAV, _fetch_librispeech_sample())
    sr, y = wavfile.read(SOURCE_WAV)
    orig_dtype = y.dtype
    y = y.astype(float)
    if orig_dtype.kind in {"i", "u"}:
        y /= np.iinfo(orig_dtype).max
    if sr != SR:
        y = resample_poly(y, SR, sr)
    y -= float(np.mean(y))
    y /= np.max(np.abs(y)) or 1.0
    # Skip initial breath/silence and use a compact voiced region for legible figures.
    start = int(0.20 * SR)
    n = int(duration * SR)
    y = y[start:start + n]
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return 0.85 * y


def stft_mag(y: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    _, _, Z = sp_stft(y, fs=SR, nperseg=n_fft, noverlap=n_fft - hop, boundary="zeros", padded=True)
    return np.abs(Z)


def stft_complex(y: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    _, _, Z = sp_stft(y, fs=SR, nperseg=n_fft, noverlap=n_fft - hop, boundary="zeros", padded=True)
    return Z


def log_mel_from_mag(
    mag: np.ndarray,
    n_mels: int = N_MELS,
    ref: float = 1.0,
    amin: float = 1e-10,
) -> np.ndarray:
    mel = mel_basis(n_mels) @ (mag**2)
    mel = np.maximum(mel, amin)
    return 10.0 * np.log10(mel / ref)


def log_mel(y: np.ndarray) -> np.ndarray:
    return log_mel_from_mag(stft_mag(y))


def phase_scramble(y: np.ndarray, seed: int = 0) -> np.ndarray:
    Z = stft_complex(y)
    mag = np.abs(Z)
    rng = np.random.default_rng(seed)
    phase = rng.uniform(-np.pi, np.pi, size=Z.shape)
    Zs = mag * np.exp(1j * phase)
    _, y_out = sp_istft(Zs, fs=SR, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=True)
    n = len(y)
    if len(y_out) >= n:
        return y_out[:n]
    out = np.zeros(n)
    out[: len(y_out)] = y_out
    return out


def mel_pinv_roundtrip_mag(mag: np.ndarray, n_mels: int = N_MELS) -> tuple[np.ndarray, np.ndarray]:
    E = mel_basis(n_mels)
    mel_pow = E @ (mag**2)
    approx = np.maximum(E.T @ mel_pow, 0.0)
    approx = np.sqrt(approx)
    residual = np.maximum(mag - approx, 0.0)
    return approx, residual


def db_to_power(db: np.ndarray, ref: float = 1.0) -> np.ndarray:
    return ref * (10.0 ** (db / 10.0))


def mel_to_linear_mag(log_mel_db: np.ndarray, n_mels: int = N_MELS) -> np.ndarray:
    """Pseudo-inverse mel round-trip on magnitude spectrogram."""
    E = mel_basis(n_mels)
    mel_pow = db_to_power(log_mel_db)
    linear_pow = np.maximum(E.T @ mel_pow, 0.0)
    return np.sqrt(linear_pow)


def griffin_lim(
    mag: np.ndarray,
    n_iter: int = 32,
    n_fft: int = N_FFT,
    hop: int = HOP,
    length: int | None = None,
) -> np.ndarray:
    """Estimate waveform from magnitude spectrogram via Griffin-Lim."""
    rng = np.random.default_rng(0)
    phase = rng.uniform(-np.pi, np.pi, size=mag.shape)
    Z = mag * np.exp(1j * phase)
    for _ in range(n_iter):
        _, y = sp_istft(Z, fs=SR, nperseg=n_fft, noverlap=n_fft - hop, boundary=True)
        _, _, Z_next = sp_stft(y, fs=SR, nperseg=n_fft, noverlap=n_fft - hop, boundary="zeros", padded=True)
        phase_next = np.angle(Z_next)
        if phase_next.shape != mag.shape:
            aligned = np.zeros_like(mag)
            rows = min(aligned.shape[0], phase_next.shape[0])
            cols = min(aligned.shape[1], phase_next.shape[1])
            aligned[:rows, :cols] = phase_next[:rows, :cols]
            phase_next = aligned
        Z = mag * np.exp(1j * phase_next)
    _, y = sp_istft(Z, fs=SR, nperseg=n_fft, noverlap=n_fft - hop, boundary=True)
    if length is not None:
        if len(y) >= length:
            return y[:length]
        out = np.zeros(length)
        out[: len(y)] = y
        return out
    return y


def classical_mel_roundtrip(y: np.ndarray) -> np.ndarray:
    """Full classical path: audio -> log-mel -> E+ -> Griffin-Lim -> audio."""
    lm = log_mel(y)
    mag_hat = mel_to_linear_mag(lm)
    return griffin_lim(mag_hat, length=len(y))
