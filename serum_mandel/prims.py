"""Building blocks: the starting patch, identity nodes, slot curves and track&hold."""

from __future__ import annotations

import numpy as np

from .patch import Patch

EPS = 1e-4
# 256-frame DC ramp: the table the primitives were measured with (table position -> output)
RAMP = np.repeat(np.linspace(-1, 1, 256, dtype=np.float32)[:, None], 2048, axis=1)


def base_patch() -> Patch:
    """Init state with sub, noise, voice filter, amp-envelope shaping and parameter smoothing
    off. Osc A and B hold the DC ramp at volume 0. They are leftovers of the measurement
    scope, kept so the preset matches the one that was verified."""
    p = Patch()
    p.set("Oscillator4", kParamVolume=0.0)           # sub off
    p.set("Oscillator3", kParamVolume=0.0)           # noise off
    p.set("VoiceFilter0", kParamEnable=0.0)
    p.set("Env0", kParamAttack=0.0, kParamDecay=0.0, kParamSustain=1.0, kParamRelease=0.0,
          kParamHold=0.0)
    p.set("Global0", kParamSmoothingDisable=1.0)
    for i, pan in ((0, -50.0), (1, 50.0)):
        p.wavetable(i, RAMP, name="ramp")
        p.osc(i, kParamEnable=1.0, kParamVolume=0.0, kParamPan=pan)
        p.wt(i, kParamXfadeMode=1.0, kParamTablePos=1.0)
    return p


def node(p: Patch, i: int, phase_deg: float = 0.0) -> None:
    """LFO i as an identity node: rate 0 and a saw shape, so its output is its phase."""
    p.lfo_curve(i, [0, 1 - EPS, 1], [1, 0, 1])
    p.set(f"LFO{i}", kParamBeatSync=0.0, kParamRate=0.0, kParamPhase=phase_deg)


def curve(xs, outs) -> dict:
    """Piecewise-linear slot curve through (x, out) points; Serum stores y = 1 - out."""
    xs, outs = list(map(float, xs)), list(map(float, outs))
    return {"curveVals": [0.5] * len(xs), "numPoints": len(xs) - 1, "xVals": xs,
            "yVals": [1.0 - o for o in outs]}


def slot_flex(p: Patch, s: int, main: dict | None = None, aux: dict | None = None, **plain) -> None:
    """Give slot s a main (and aux) curve. Curves, and slot smoothing, only take effect with
    kParamMainCurveData = 1."""
    m = p.body[f"ModSlot{s}"]
    m["flex"] = [main or {}, aux or {}]
    m["plainParams"]["kParamMainCurveData"] = 1.0
    m["plainParams"].update({k: float(v) for k, v in plain.items()})


def track_hold(p: Patch, s: int, hold_source: int, hold_curve: dict) -> None:
    """Make slot s a track&hold: its smoothing is 0 (instant track) and is modulated to 100
    (tau ~2.6 s, effectively frozen) while hold_source through hold_curve reads 1. Holding
    from a base of 100 and tracking by modulating down instead leaves a ~1.9 ms floor."""
    slot_flex(p, s, kParamSmoothRise=0.0, kParamSmoothFall=0.0)
    for prm in ("kParamSmoothRise", "kParamSmoothFall"):
        c = p.route(hold_source, "ModSlot", s, prm, 100.0)
        slot_flex(p, c, main=hold_curve)
