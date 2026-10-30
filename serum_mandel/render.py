"""The rendering part (free to fudge): short pixel notes on a tilted grid, one linear response.

Layout: R rows at f_lo + r*df (linear Hz), columns every P seconds. Row r's pixel in
column c starts at c*P + o_r, with o_r = ((r*k) mod R)/R * P: a tilted, staggered
lattice. Notes last T_n = reset + compute + burst, and P = R*T_n/32, so at most 32
voices sound at once. Each pixel's c comes from where its footprint lands: cy from its
row, cx from the time its footprint is centred on.

A pixel shows only a short burst (about 1/df long, so its own bandwidth covers its row
band) at a level set by its escape time. One convolution IR (Serum's Conv FX with an
embedded IR) spreads every burst in time across its lattice cell: a comb with taps
every 1/df, so its teeth sit on the rows, with jitter for diffusion and a decaying
envelope. A small per-pixel chirp tilts each stroke.

The look (shading curve, fade, IR, wet, chirp) was tuned by CMA-ES against a reference
escape-time image, scoring each candidate with a full Serum render.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from .core import CMap, ET, L, build, escape_reference, escape_timer
from .patch import SRC_ENV
from .prims import curve, slot_flex

SR = 48000          # sample rate the embedded IR is written at
RESET_S = 0.012     # READY window (Env 2 attack 10 ms) plus margin
BASE_HZ = 261.6256  # Osc C with key tracking off, octave +4, semitone +12
SINE = np.sin(2 * np.pi * np.arange(2048) / 2048).astype(np.float32)[None]
# 1 while Env 3 is at its peak (the display burst), else 0
SHOW = {"curveVals": [0.5, 0.5, 0.5, 0.5], "numPoints": 3,
        "xVals": [0.0, 0.98, 0.98, 1.0], "yVals": [1.0, 1.0, 0.0, 0.0]}


class Note(NamedTuple):
    pitch: int
    velocity: int
    start: float      # seconds
    duration: float   # seconds


@dataclass
class Grid:
    rows: int = 64
    cols: int = 96
    f_lo: float = 400.0
    df: float = 100.0
    clock_hz: float = 500.0
    iters: int = 20
    burst_s: float = 0.010    # ~1/df: the burst's own bandwidth covers its row band
    gap_s: float = 0.006
    tilt_k: int = 27          # o_r = ((r * tilt_k) mod rows) / rows * P; coprime with rows,
                              # far from 1 so neighbouring rows don't burst together
    voices: float = 32.0      # concurrency budget (P = rows * note_s / voices)
    cx_lo: float = -2.0
    cx_hi: float = 0.5
    cy_lo: float = -1.2
    cy_hi: float = 1.2

    @property
    def compute_s(self) -> float:
        return self.iters / self.clock_hz

    @property
    def note_s(self) -> float:
        return RESET_S + self.compute_s + self.burst_s + self.gap_s

    @property
    def period(self) -> float:            # column period P
        return self.rows * self.note_s / self.voices

    @property
    def notes(self) -> list[int]:
        step = 127 // (self.rows - 1)
        assert step >= 1 and self.rows <= 128
        return [k * step for k in range(self.rows)]

    def cmap(self) -> CMap:
        return CMap(self.notes[0], self.notes[-1], self.cy_lo, self.cy_hi)

    def offset(self, r: int) -> float:
        return ((r * self.tilt_k) % self.rows) / self.rows * self.period

    def row_hz(self, r) -> float:
        return self.f_lo + r * self.df

    def burst_t(self, r: int, c: int) -> float:
        """Start of pixel (r, c)'s display burst."""
        return c * self.period + self.offset(r) + RESET_S + self.compute_s

    def t_to_cx(self, t: np.ndarray) -> np.ndarray:
        """Image x at time t measured from the first burst (the image's left edge)."""
        return self.cx_lo + (np.asarray(t) / (self.cols * self.period)) * (self.cx_hi - self.cx_lo)


@dataclass
class Look:
    shade: list = field(default_factory=lambda: [0.0, 0.06, 0.15, 0.32, 0.58])
    fade: float = 25.0           # burst edge smoothing (slot smoothing units)
    ir_len: float = 1.0          # IR length in column periods
    tap_s: float = 0.010         # comb tap spacing (s); 1/df puts the comb's teeth on the rows
    jitter: float = 0.2          # comb tap jitter, fraction of tap spacing (diffusion)
    decay: float = 0.0           # envelope decay over the IR (0 = flat)
    wet: float = 100.0
    seed: int = 0
    noise_depth: float = 0.0     # kFM_NOISE depth from brown noise; below ~0.38 no effect. 0 = off
    chirp_hz: float = 0.0        # pitch sweep across the display burst, in Hz (signed), same on all rows


def make_ir(g: Grid, lk: Look) -> np.ndarray:
    rng = np.random.default_rng(lk.seed)
    n = max(64, int(lk.ir_len * g.period * SR))
    env = np.exp(-lk.decay * np.arange(n) / n)
    h = np.zeros(n)
    spacing = max(1, int(lk.tap_s * SR))
    for k, pos in enumerate(range(0, n, spacing)):
        j = int(round(pos + rng.normal(0, lk.jitter * spacing))) if k else 0
        if 0 <= j < n:
            h[j] += env[pos]
    return (h / np.sqrt(np.sum(h ** 2) + 1e-12)).astype(np.float32)


def _row_curve(g: Grid, values) -> dict:
    """Slot curve on the Note source: values[r] at row r's note, flat outside the rows."""
    xs, outs = [n / 127 for n in g.notes], list(values)
    if g.notes[0] > 0:
        xs, outs = [0.0] + xs, [outs[0]] + outs
    if g.notes[-1] < 127:
        xs, outs = xs + [1.0], outs + [outs[-1]]
    return curve(xs, outs)


def make_patch(g: Grid, lk: Look):
    p = build(g.cmap(), g.clock_hz)
    p.set("Global0", kParamPolyCount=32.0)
    p.set("Env0", kParamAttack=0.001, kParamRelease=0.004)
    p.set("Env2", kParamAttack=RESET_S + g.compute_s, kParamHold=g.burst_s, kParamDecay=0.0,
          kParamSustain=0.0, kParamRelease=0.0)                    # Env 3: the display burst
    escape_timer(p, RESET_S + g.compute_s, SRC_ENV[3], SHOW)
    # Osc C: a sine at the row frequency (Note -> coarse pitch, one curve point per row)
    p.wavetable(2, SINE, "sine")
    p.osc(2, kParamEnable=1.0, kParamVolume=0.0, kParamPan=0.0, kParamPitchTrack=0.0,
          kParamOctave=4.0, kParamPitch=12.0)
    semis = [12 * np.log2(g.row_hz(r) / BASE_HZ) for r in range(g.rows)]
    assert 0.0 <= min(semis) and max(semis) <= 64.0, (min(semis), max(semis))
    s = p.route(17, "Oscillator", 2, "kParamCoarsePit", 50.0, slot=40)   # 50% = 64 semitones
    slot_flex(p, s, main=_row_curve(g, [o / 64.0 for o in semis]))
    # level = shade(escape time) during the display burst
    knots = np.linspace(0, 0.95, len(lk.shade))
    s = p.route(L(ET), "Oscillator", 2, "kParamVolume", 100.0, aux=SRC_ENV[3], slot=44)
    slot_flex(p, s, main=curve(list(knots) + [1.0], list(lk.shade) + [lk.shade[-1]]),
              aux=SHOW, kParamAuxCurveData=1.0, kParamSmoothRise=lk.fade, kParamSmoothFall=lk.fade)
    if lk.noise_depth > 0:
        p.osc(3, kParamEnable=1.0, kParamVolume=0.0, kParamPitchTrack=0.0)
        p.set("Oscillator3", "NoiseOsc3", kParamNoiseType="Brown", kParamPitch=0.0)
        p.wt(2, kParamWarpMenu="kFM_NOISE", kParamWarp=lk.noise_depth)
    if lk.chirp_hz:
        # Env 4 ramps over the whole note; the slot curve keeps only the display part, so the
        # burst sweeps fine pitch. Fine pitch is in cents, so an aux curve on Note scales the
        # sweep by f_lo / f_row: the same Hz on every row.
        note_len = g.note_s - g.gap_s
        p.set("Env3", kParamAttack=note_len, kParamDecay=0.0, kParamSustain=1.0, kParamRelease=0.0)
        a0 = (RESET_S + g.compute_s) / note_len
        cents_lo = 1200 * np.log2(1 + abs(lk.chirp_hz) / g.f_lo)      # sweep in cents at the lowest row
        amt = float(np.clip(cents_lo / 200.0 * 100.0, 0, 100)) * np.sign(lk.chirp_hz)  # fine: +-100 c
        s2 = p.route(SRC_ENV[4], "Oscillator", 2, "kParamFine", amt, aux=17, slot=46)
        slot_flex(p, s2, main=curve([0.0, a0, 1.0], [0.0, 0.0, 1.0]),
                  aux=_row_curve(g, [g.f_lo / g.row_hz(r) for r in range(g.rows)]),
                  kParamAuxCurveData=1.0)
    ir = make_ir(g, lk)
    fx = p.add_fx("FXConv", kParamWet=lk.wet, kParamSize=100.0, kParamDecay=100.0, kParamTone=0.0,
                  kParamPredelay=0.0, kParamIpTrim=0.0)
    fx["FXConv"].update(embeddedIR=[ir.tolist(), ir.tolist()], numChannels=2, numFrames=len(ir),
                        sampleRate=SR)
    return p


def schedule(g: Grid, ir_len: float = 1.0) -> list[Note]:
    """Pixel (r, c): its footprint (burst + IR spread over ir_len periods) is centred at
    burst_t + ir_len*P/2; cx is taken from that time."""
    notes = []
    for c in range(g.cols):
        for r, n in enumerate(g.notes):
            t0 = c * g.period + g.offset(r)
            cx = g.t_to_cx(g.burst_t(r, c) - g.burst_t(0, 0) + ir_len * g.period / 2)
            vel = int(np.clip(round((cx + 2.0) * 127 / 2.5), 1, 127))
            notes.append(Note(n, vel, t0, g.note_s - g.gap_s))
    return sorted(notes, key=lambda n: n.start)


# -- scoring -------------------------------------------------------------------------

def display(audio: np.ndarray, g: Grid, n_fft: int = 2048, hop: int = 240,
            db_range: float = 50.0, sr: int = SR) -> np.ndarray:
    """Linear spectrogram sampled on the image grid: one cell per row band, two per column,
    top row = highest frequency. audio is [channels, samples] starting at the first pixel
    note-on. Returns values in [0, 1]."""
    x = audio.mean(axis=0)
    x = np.concatenate([np.zeros(n_fft // 2), x, np.zeros(n_fft)])
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop] * np.hanning(n_fft)
    mag = np.abs(np.fft.rfft(frames, axis=1))
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    t = np.arange(mag.shape[0]) * hop / sr
    t_cells = 2 * g.cols
    out = np.zeros((g.rows, t_cells))
    edges_t = np.linspace(0, g.cols * g.period, t_cells + 1) + g.burst_t(0, 0)
    idx = np.searchsorted(t, edges_t)
    for i in range(g.rows):
        fc = g.f_lo + i * g.df
        sel = np.abs(freqs - fc) <= g.df / 2
        band = mag[:, sel].max(axis=1) if sel.any() else mag[:, np.argmin(np.abs(freqs - fc))]
        out[g.rows - 1 - i] = [band[a:b].mean() if b > a else 0.0 for a, b in zip(idx[:-1], idx[1:])]
    db = 20 * np.log10(out + 1e-9)
    db -= np.percentile(db, 99.5)
    return np.clip(1 + db / db_range, 0, 1)


def reference(g: Grid) -> np.ndarray:
    """The true escape-time image on the display grid, brightness = escape iteration / N."""
    t_cells = 2 * g.cols
    cys = [g.cmap().c(n, 64).imag for n in g.notes][::-1]
    cxs = g.t_to_cx((np.arange(t_cells) + 0.5) / t_cells * g.cols * g.period)
    return np.array([[escape_reference(complex(cx, cy), g.iters) / g.iters for cx in cxs]
                     for cy in cys])


def score(a: np.ndarray, b: np.ndarray) -> float:
    """1 - Pearson correlation: 0 = identical up to brightness and contrast."""
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(1 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
