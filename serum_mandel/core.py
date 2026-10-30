"""The honest part: every voice iterates z <- z^2 + c and times its own escape, inside
Serum's modulation system.

Values v in [-2, 2] are encoded as u = (v + 2) / 4. Every node is an LFO with rate 0
and an identity (saw) shape, so its output is its phase, and its phase is the clamped
sum of everything routed to kParamPhase. A phase that clamps at 0 or 1 reads back as 0
(v = -2), so an orbit that leaves the box lands on -2.

    P  = (Rx + Ry) / 2                     u_P = (x + y + 4) / 8
    Fx = V(Rx)^2 - V(Ry)^2 + 0.625 vel     V(u) = |2u - 1|, so V(Rx)^2 = x^2 / 4
    Fy = 4 V(P)^2 - V(Rx)^2 - V(Ry)^2 + (cy + 2) / 4        = xy/2 + (cy+2)/4
    Sx, Sy <- Fx, Fy   (track&hold, clock phase A)
    Rx, Ry <- Sx, Sy   (track&hold, clock phase B)

Squares are aux products of a V-shaped curve with itself, which Serum computes
exactly. c comes from MIDI: cx = -2 + 2.5 vel/127, cy from the note number.

Clock: one free-running mono LFO (CA, a saw) shared by all voices; slot curves cut the
two non-overlapping hold windows from it. It has to be global: modulating a mod slot's
own parameter (here its smoothing) is not per voice in Serum 2. One voice's value
applies to every voice, so per-voice clocks on staggered notes corrupt each other.

Reset: slot smoothing state lives with the voice slot and survives note-on, so a reused
voice would start from the previous pixel's z. READY (Env 2 through a step curve) is 0
for the first ~10 ms of every note, and while it is 0 a NOT-READY slot pulls the R and
S phases down to the clamp at 0. R and S use the ZERO_AT_CLAMP shape, which reads 0.5
(v = 0) at the clamp point, so every note starts from z0 = 0. Because of that shape an
overflow shows up as 0 only on the F nodes (identity shape), where escape is read.
"""

from __future__ import annotations

from .patch import SRC_ENV, SRC_LFO, Patch
from .prims import base_patch, node, slot_flex, track_hold

SRC_VELOCITY, SRC_NOTE = 16, 17

V = {"curveVals": [0.5, 0.5, 0.5], "numPoints": 2, "xVals": [0.0, 0.5, 1.0], "yVals": [0.0, 1.0, 0.0]}
# 1 once Env 2 has finished its 10 ms attack, else 0.
READY = {"curveVals": [0.5, 0.5, 0.5, 0.5], "numPoints": 3,
         "xVals": [0.0, 0.98, 0.98, 1.0], "yVals": [1.0, 1.0, 0.0, 0.0]}
NOT_READY = {"curveVals": [0.5, 0.5, 0.5, 0.5], "numPoints": 3,
             "xVals": [0.0, 0.98, 0.98, 1.0], "yVals": [0.0, 0.0, 1.0, 1.0]}
# 1 when an F node reads u < 0.004 (it overflowed and clamped to 0), else 0.
ESCAPED_F = {"curveVals": [0.5, 0.5, 0.5, 0.5], "numPoints": 3,
             "xVals": [0.0, 0.004, 0.004, 1.0], "yVals": [0.0, 0.0, 1.0, 1.0]}
SRC_READY_ENV = SRC_ENV[2]
_E = 1e-4
# LFO indices: registers Rx, Ry, Sx, Sy, then P, Fx, Fy, the clock, the escape latch,
# and the escape timer
RX, RY, SX, SY, P, FX, FY, CA, CB, ET = range(10)
L = lambda i: SRC_LFO[i + 1]  # noqa: E731  (mod source id of LFO index i)


class CMap:
    """Note/velocity -> c. cx = -2 + 2.5 * vel/127. cy = cy0 + cy_slope * note/127."""

    def __init__(self, note_lo: int, note_hi: int, cy_lo: float = -1.2, cy_hi: float = 1.2):
        self.note_lo, self.note_hi = note_lo, note_hi
        self.cy_slope = (cy_hi - cy_lo) * 127.0 / (note_hi - note_lo)
        self.cy0 = cy_lo - self.cy_slope * note_lo / 127.0

    def c(self, note: int, vel: int) -> complex:
        return complex(-2.0 + 2.5 * vel / 127.0, self.cy0 + self.cy_slope * note / 127.0)


def square_slot(p: Patch, src: int, dest: int, amount: float) -> int:
    s = p.route(src, "LFO", dest, "kParamPhase", amount, aux=src)
    slot_flex(p, s, main=V, aux=V, kParamAuxCurveData=1.0)
    return s


def _window(lo: float, hi: float, inside: float) -> dict:
    """Slot curve on a saw input: `inside` on [lo, hi), 1 - inside elsewhere."""
    e = 1e-4
    out = [1 - inside, 1 - inside, inside, inside, 1 - inside, 1 - inside]
    return {"curveVals": [0.5] * 6, "numPoints": 5, "xVals": [0.0, lo - e, lo, hi - e, hi, 1.0],
            "yVals": [1.0 - o for o in out]}


HOLD_A = _window(0.05, 0.45, 0.0)    # hold (1) except while S tracks F
HOLD_B = _window(0.55, 0.95, 0.0)    # hold (1) except while R tracks S


def build(cmap: CMap, clock_hz: float = 500.0) -> Patch:
    """The iteration core: 9 LFOs and ~30 mod slots per voice."""
    p = base_patch()
    p.set("Env1", kParamAttack=0.01, kParamDecay=0.0, kParamSustain=1.0, kParamRelease=0.0)  # Env 2
    for i in (RX, RY, SX, SY):
        node(p, i, 180.0)
        # identity on (0, 1), but 0.5 (v = 0) at the clamp point phase 0 == 1
        p.lfo_curve(i, [0, _E, 1 - _E, 1], [0.5, 1 - _E, _E, 0.5])
    node(p, P, 0.0)
    node(p, FX, 0.0)
    fy_base = (cmap.cy0 + 2.0) / 4.0
    assert 0.0 <= fy_base < 1.0, fy_base
    node(p, FY, 360.0 * fy_base)
    tenx = clock_hz > 100.0
    node(p, CA, 0.0)                                                 # saw
    p.set(f"LFO{CA}", kParamRate=clock_hz / (10 if tenx else 1), kParamMono=1.0, kParamMode="Free",
          **({"kParamRate10x": 1.0} if tenx else {}))

    # P = (Rx + Ry) / 2
    p.route(L(RX), "LFO", P, "kParamPhase", 50.0)
    p.route(L(RY), "LFO", P, "kParamPhase", 50.0)
    # Fx = x^2/4 - y^2/4 + (cx + 2)/4
    square_slot(p, L(RX), FX, 100.0)
    square_slot(p, L(RY), FX, -100.0)
    p.route(SRC_VELOCITY, "LFO", FX, "kParamPhase", 62.5)
    # Fy = (x+y)^2/4 - x^2/4 - y^2/4 + (cy + 2)/4
    for _ in range(4):
        square_slot(p, L(P), FY, 100.0)
    square_slot(p, L(RX), FY, -100.0)
    square_slot(p, L(RY), FY, -100.0)
    assert cmap.cy_slope / 4.0 <= 1.0
    p.route(SRC_NOTE, "LFO", FY, "kParamPhase", 100.0 * cmap.cy_slope / 4.0)
    # registers: bipolar links around base phase 180 deg -> node = held source
    for dst, src, hold in ((SX, FX, HOLD_A), (SY, FY, HOLD_A), (RX, SX, HOLD_B), (RY, SY, HOLD_B)):
        # 100%, because a bipolar slot with modulated smoothing contributes half
        s = p.route(L(src), "LFO", dst, "kParamPhase", 100.0, kParamBipolar=1.0)
        track_hold(p, s, L(CA), hold)
        r = p.route(SRC_READY_ENV, "LFO", dst, "kParamPhase", -100.0)
        slot_flex(p, r, main=NOT_READY)
    return p


def escape_timer(p: Patch, compute_s: float, stop_src: int, stop_curve: dict,
                 latch_slots: tuple[int, int] = (52, 62)) -> None:
    """LFO 10 counts time from note-on (phase = rate * t, identity shape). It stops for good
    once either F node overflows, and while `stop_src` through `stop_curve` reads 1 (the
    display window). At the end of the compute window it reads
    (escape time) / compute_s * 0.95, or 0.95 if the orbit never escaped. Rate 0 holds an LFO
    phase exactly, so the reading does not drift while it is displayed.

    The latch is LFO CB, a per-voice integrator: its rate rises on overflow and it stops
    itself once its phase passes 0.1. It resets at note-on. The two latch slots sit apart:
    two latched slots on one destination misbehave when they are adjacent."""
    node(p, ET, 0.0)
    p.set(f"LFO{ET}", kParamRate=0.95 / compute_s)
    node(p, CB, 0.0)
    p.set(f"LFO{CB}", kParamRate=0.0, kParamRate10x=1.0)
    for f, slot in zip((FX, FY), latch_slots):
        s = p.route(L(f), "LFO", CB, "kParamRate", 100.0, aux=SRC_READY_ENV, slot=slot)
        slot_flex(p, s, main=ESCAPED_F, aux=READY, kParamAuxCurveData=1.0)
        s = p.route(L(CB), "LFO", CB, "kParamRate", -100.0, slot=slot + 1)
        slot_flex(p, s, main=_window(0.1, 1.0, 1.0))
    s = p.route(L(CB), "LFO", ET, "kParamRate", -100.0)
    slot_flex(p, s, main=_window(0.05, 1.0, 1.0))
    s = p.route(stop_src, "LFO", ET, "kParamRate", -100.0)
    slot_flex(p, s, main=stop_curve)


def escape_reference(c: complex, iters: int) -> int:
    """Iterations until the patch's escape rule fires (an F value leaves [-2, 2]); iters if never."""
    z = 0j
    for n in range(iters):
        z = z * z + c
        if abs(z.real) >= 2 or abs(z.imag) >= 2:
            return n
    return iters
