"""PNG output (numpy + zlib only), WAV input, and a linear spectrogram."""

from __future__ import annotations

import struct
import zlib

import numpy as np

# Sequential single-hue ramp (light -> dark).
RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _rgb(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float)


def colorize(v: np.ndarray) -> np.ndarray:
    """v in [0, 1] -> uint8 RGB through RAMP."""
    stops = np.stack([_rgb(h) for h in RAMP])
    x = np.clip(v, 0, 1) * (len(RAMP) - 1)
    i = np.minimum(x.astype(int), len(RAMP) - 2)
    f = (x - i)[..., None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)


def write_png(path, rgb: np.ndarray) -> None:
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def side_by_side(a: np.ndarray, b: np.ndarray, s: int = 3) -> np.ndarray:
    """Two [0, 1] images, each upscaled s times, with a gap between them, as RGB."""
    up = lambda m: np.kron(m, np.ones((s, s)))  # noqa: E731
    return colorize(np.concatenate([up(a), np.zeros((a.shape[0] * s, 2 * s)), up(b)], axis=1))


def spectrogram(audio: np.ndarray, sr: int, n_fft: int = 4096, hop: int = 256, fmin: float = 300.0,
                fmax: float = 6800.0, rows: int = 520, db_range: float = 40.0) -> np.ndarray:
    """Linear-frequency spectrogram in [0, 1], high frequencies on top."""
    x = audio.mean(axis=0) if audio.ndim == 2 else audio
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop] * np.hanning(n_fft)
    mag = np.abs(np.fft.rfft(frames, axis=1))
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    edges = np.linspace(fmin, fmax, rows + 1)
    out = np.zeros((rows, mag.shape[0]))
    for r in range(rows):
        sel = (freqs >= edges[r]) & (freqs < edges[r + 1])
        out[r] = mag[:, sel].max(axis=1) if sel.any() else mag[:, np.argmin(np.abs(freqs - edges[r]))]
    db = 20 * np.log10(out + 1e-9)
    db -= db.max()
    return np.clip(1 + db / db_range, 0, 1)[::-1]


def read_wav(path) -> tuple[np.ndarray, int]:
    """PCM 16/24/32-bit or float 32/64-bit WAV -> ([channels, samples] float64, sample rate)."""
    data = open(path, "rb").read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF/WAVE file")
    pos, fmt = 12, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack_from("<I", data, pos + 4)[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            tag, ch, sr = struct.unpack_from("<HHI", body)
            bits = struct.unpack_from("<H", body, 14)[0]
            if tag == 0xFFFE:                            # WAVE_FORMAT_EXTENSIBLE
                tag = struct.unpack_from("<H", body, 24)[0]
            fmt = (tag, ch, sr, bits)
        elif cid == b"data" and fmt:
            tag, ch, sr, bits = fmt
            if tag == 3:
                x = np.frombuffer(body, "<f4" if bits == 32 else "<f8").astype(np.float64)
            elif tag == 1 and bits == 24:
                b = np.frombuffer(body[:len(body) // 3 * 3], np.uint8).reshape(-1, 3).astype(np.int32)
                x = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
                x = np.where(x >= 1 << 23, x - (1 << 24), x) / float(1 << 23)
            elif tag == 1 and bits in (16, 32):
                x = np.frombuffer(body, f"<i{bits // 8}") / float(1 << (bits - 1))
            else:
                raise ValueError(f"{path}: unsupported WAV format {tag}, {bits} bits")
            n = len(x) // ch * ch
            return x[:n].reshape(-1, ch).T, sr
        pos += 8 + size + (size & 1)
    raise ValueError(f"{path}: no fmt/data chunk")
