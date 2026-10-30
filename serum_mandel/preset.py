"""Convert .SerumPreset files into Serum 2 VST3 state blobs.

A preset body is the processor state and the controller state merged into one
CBOR map: shared module keys (FXRack*, LFO*, ...) carry the union of both
sides' fields. Conversion starts from template states captured from a fresh
plugin (templates/init.*.bin) and overlays every preset module onto the
side(s) that know that key. Each template keeps its own metadata keys
(component, version, product, ...).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

from . import xferjson
from .xferjson import XferDoc

DEFAULT_TEMPLATE_PREFIX = Path(__file__).resolve().parent / "templates" / "init"

# Keys that describe the file or UI rather than the sound.
PRESET_ONLY_KEYS = {"fileType"}
META_KEYS = {"component", "product", "productVersion", "url", "vendor", "version"}


@dataclass
class Templates:
    component: XferDoc
    controller: XferDoc

    @classmethod
    def load(cls, prefix: Path | str = DEFAULT_TEMPLATE_PREFIX) -> "Templates":
        prefix = str(prefix)
        return cls(xferjson.load(prefix + ".component.bin"),
                   xferjson.load(prefix + ".controller.bin"))


def split_preset(preset: XferDoc, templates: Templates) -> tuple[XferDoc, XferDoc]:
    comp_body = copy.deepcopy(templates.component.body)
    ctrl_body = copy.deepcopy(templates.controller.body)
    ctrl_keys = set(ctrl_body)
    for key, value in preset.body.items():
        if key in PRESET_ONLY_KEYS or key in META_KEYS:
            continue
        in_comp = key in comp_body
        in_ctrl = key in ctrl_keys
        if in_comp or not in_ctrl:
            comp_body[key] = copy.deepcopy(value)
        if in_ctrl:
            ctrl_body[key] = copy.deepcopy(value)
    comp = XferDoc(dict(templates.component.header), comp_body, templates.component.payload_version)
    ctrl = XferDoc(dict(templates.controller.header), ctrl_body, templates.controller.payload_version)
    for k in ("presetName", "presetAuthor", "presetDescription"):
        if k in preset.header:
            ctrl.header[k] = preset.header[k]
    return comp, ctrl


def preset_to_state(path, templates: Templates | None = None) -> tuple[bytes, bytes]:
    templates = templates or Templates.load()
    comp, ctrl = split_preset(xferjson.load(path), templates)
    return xferjson.dump_bytes(comp), xferjson.dump_bytes(ctrl)
