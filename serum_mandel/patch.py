"""Build Serum 2 patches in code, starting from the init state.

A Patch holds one preset-shaped body (module key -> dict) and turns it into
VST3 state blobs with preset.split_preset. Plain values are the ones stored in
presets (e.g. LFO kParamRate in Hz, ModSlot kParamAmount in percent).

Mod sources are [main, aux] pairs of source ids, found from the factory library
and probes: 2-5 Env 1-4, 6-15 LFO 1-10, 16 Velocity, 17 Note (both value/127),
25-32 Macro 1-8.
"""

from __future__ import annotations

import copy
from typing import Sequence

import numpy as np

from . import xferjson
from .preset import Templates, split_preset
from .xferjson import XferDoc

SRC_ENV = {i: i + 1 for i in range(1, 5)}      # Env 1-4
SRC_LFO = {i: 5 + i for i in range(1, 11)}     # LFO 1-10
SRC_MACRO = {i: 24 + i for i in range(1, 9)}   # Macro 1-8

WT_FRAME = 2048

# Mod destination param ids of the params this patch modulates, as Serum stores them in
# presets (checked against the factory library and by experiment). A wrong id can hit
# another param: an Oscillator route with a bad id moved Pan.
DEST_PARAM_IDS = {
    "LFO": {"kParamRate": 0, "kParamPhase": 4},
    "ModSlot": {"kParamSmoothRise": 11, "kParamSmoothFall": 12},
    "Oscillator": {"kParamVolume": 1, "kParamFine": 5, "kParamCoarsePit": 6},
}


def dest_param_id(dest: str, param: str) -> int:
    try:
        return DEST_PARAM_IDS[dest][param]
    except KeyError:
        raise KeyError(f"no known param id for {dest}.{param}; add it to DEST_PARAM_IDS") from None


def _floatify(params: dict) -> dict:
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) else v)
            for k, v in params.items()}


class Patch:
    def __init__(self, templates: Templates | None = None):
        self.templates = templates or Templates.load()
        self.body = copy.deepcopy(self.templates.component.body)
        self._next_slot = 0

    # -- generic ----------------------------------------------------------

    def module(self, *path: str) -> dict:
        d = self.body
        for p in path:
            d = d[p]
        return d

    def set(self, *path: str, **plain) -> "Patch":
        """set('LFO0', kParamRate=2.0) or set('Oscillator0', 'WTOsc0', kParamTablePos=3)."""
        mod = self.module(*path)
        pp = mod.get("plainParams")
        if not isinstance(pp, dict):
            pp = mod["plainParams"] = {}
        pp.update(_floatify(plain))
        return self

    # -- oscillators --------------------------------------------------------

    def osc(self, i: int, **plain) -> "Patch":
        return self.set(f"Oscillator{i}", **plain)

    def wt(self, i: int, **plain) -> "Patch":
        return self.set(f"Oscillator{i}", f"WTOsc{i}", **plain)

    def wavetable(self, i: int, frames: np.ndarray, name: str = "computer") -> "Patch":
        """Embed a wavetable (n_frames x 2048 samples) in WT oscillator i (0-2)."""
        frames = np.asarray(frames, dtype=np.float32)
        if frames.ndim == 1:
            frames = frames[None]
        assert frames.shape[1] == WT_FRAME, frames.shape
        w = self.module(f"Oscillator{i}", f"WTOsc{i}")
        w.pop("relativePathToWT", None)
        w.pop("pathToWT", None)
        w["embeddedWTData"] = [float(x) for x in frames.reshape(-1)]
        w["numFrames"] = int(frames.size)
        w["numChannels"] = 1
        w["sampleRate"] = 44100
        w["tableDisplayName"] = name
        return self

    # -- modulation ---------------------------------------------------------

    def route(self, source: int, dest: str, dest_id: int, param: str, amount: float = 100.0,
              aux: int = 0, slot: int | None = None, **plain) -> int:
        """Add a mod-matrix slot: source (x aux) -> dest_type[dest_id].param.

        dest is the module type string ('LFO', 'Oscillator', 'WTOsc', 'Env',
        'Macro', 'Global', ...). Returns the slot index.
        """
        if slot is None:
            slot = self._next_slot
            self._next_slot += 1
        m = self.body[f"ModSlot{slot}"]
        m["source"] = [int(source), int(aux)]
        m["destModuleTypeString"] = dest
        m["destModuleID"] = int(dest_id)
        m["destModuleParamID"] = dest_param_id(dest, param)
        m["destModuleParamName"] = param
        pp = {"kParamAmount": amount}
        pp.update(plain)
        m["plainParams"] = _floatify(pp)
        return slot

    def lfo_curve(self, i: int, xs: Sequence[float], ys: Sequence[float],
                  curves: Sequence[float] | None = None) -> "Patch":
        """Breakpoint shape for LFO i (0-9). xs in [0,1] ascending, ys in [0,1]
        (1 = top). curves holds one bend per segment plus a trailing value,
        0.5 = straight, as stored by Serum."""
        xs, ys = list(map(float, xs)), list(map(float, ys))
        assert len(xs) == len(ys) and xs[0] == 0.0 and xs[-1] == 1.0
        cv = [0.5] * len(xs) if curves is None else list(map(float, curves))
        self.body[f"LFO{i}"]["curveData"] = {
            "curveVals": cv, "numPoints": len(xs) - 1, "xVals": xs, "yVals": ys}
        return self

    # -- effects --------------------------------------------------------------

    FX_TYPES = {"FXDistortion": 0, "FXFlanger": 1, "FXPhaser": 2, "FXChorus": 3, "FXDelay": 4,
                "FXComp": 5, "FXReverb": 6, "FXEQ": 7, "FXFilter": 8, "FXHyperD": 9, "FXBode": 10,
                "FXConv": 11, "FXUtils": 12}

    def add_fx(self, kind: str, rack: int = 0, **plain) -> dict:
        """Append an effect (e.g. 'FXBode') to FX rack `rack` (0 = main). Returns its dict."""
        fx = {kind: {"plainParams": _floatify(plain)}, "kUIParamMixOrGain": 0.0,
              "type": self.FX_TYPES[kind]}
        self.body[f"FXRack{rack}"].setdefault("FX", []).append(fx)
        return fx

    # -- output -------------------------------------------------------------

    def state(self) -> tuple[bytes, bytes]:
        doc = XferDoc(dict(self.templates.component.header), self.body)
        comp, ctrl = split_preset(doc, self.templates)
        return xferjson.dump_bytes(comp), xferjson.dump_bytes(ctrl)

    def save_preset(self, path, name: str = "computer") -> None:
        """Write a .SerumPreset that opens in the Serum 2 GUI."""
        header = {"fileType": "SerumPreset", "presetAuthor": "serum-computer",
                  "presetDescription": "", "presetName": name, "product": "Serum2",
                  "productVersion": "2.1.5", "tags": [], "url": "https://xferrecords.com/",
                  "vendor": "Xfer Records", "version": 5.0}
        # A preset body is the controller and processor modules merged.
        body = copy.deepcopy(self.templates.controller.body)
        body.update(copy.deepcopy(self.body))
        for k in ("component", "hash"):
            body.pop(k, None)
        body.update(header)
        xferjson.dump(XferDoc(header, body), path)
