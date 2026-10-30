"""Score a bounce of the preset + MIDI against the true escape-time image.

    python score.py bounce.wav               # bounce starts at the MIDI file's start
    python score.py bounce.wav --lead 0.8    # seconds from the bounce start to the first pixel note

Prints 1 - corr(spectrogram on the image grid, reference); 0 is perfect. The bounce of
the shipped files scored 0.021 at 48 kHz and 44.1 kHz. Writes <bounce>_vs_ref.png (the
bounce's image next to the reference) and <bounce>_spectrogram.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from serum_mandel.image import colorize, read_wav, side_by_side, spectrogram, write_png
from serum_mandel.midi import LEAD_S
from serum_mandel.render import Grid, display, reference, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--lead", type=float, default=LEAD_S,
                    help="seconds before the first pixel note (the warm-up note's lead)")
    args = ap.parse_args()
    audio, sr = read_wav(args.wav)
    g = Grid()
    img, ref = display(audio[:, int(round(args.lead * sr)):], g, sr=sr), reference(g)
    print(f"score (1 - corr): {score(img, ref):.4f}   ({sr} Hz, {audio.shape[1] / sr:.1f} s)")
    stem = str(Path(args.wav).with_suffix(""))
    write_png(stem + "_vs_ref.png", side_by_side(img, ref))
    write_png(stem + "_spectrogram.png", colorize(spectrogram(audio, sr)))


if __name__ == "__main__":
    main()
