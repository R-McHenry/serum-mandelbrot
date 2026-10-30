"""Build the preset and the MIDI file from code, plus the reference image they should draw.

    python build.py                      # -> build/mandelbrot_v3.SerumPreset, .mid, reference.png
    python build.py --look look.json     # another rendering look

With the default look the preset and MIDI are byte-identical to the ones in the repo root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from serum_mandel.image import colorize, write_png
from serum_mandel.midi import BPM, notes, write_midi
from serum_mandel.render import Grid, Look, make_patch, reference

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--look", default=str(HERE / "look.json"), help="JSON with render.Look fields")
    ap.add_argument("--out", default=str(HERE / "build"))
    args = ap.parse_args()
    g, lk = Grid(), Look(**json.load(open(args.look)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    make_patch(g, lk).save_preset(out / "mandelbrot_v3.SerumPreset", name="Mandelbrot Computer v3")
    ns = notes(g, lk)
    write_midi(out / "mandelbrot_v3.mid", ns)
    write_png(out / "reference.png", colorize(reference(g).repeat(3, 0).repeat(3, 1)))
    end = max(n.start + n.duration for n in ns)
    print(f"{g.rows} x {g.cols} pixels, {len(ns)} notes, {end:.1f} s at {BPM:g} BPM -> {out}")


if __name__ == "__main__":
    main()
