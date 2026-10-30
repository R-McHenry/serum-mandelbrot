"""The pixel schedule as a standard MIDI file.

The patch's timing is in seconds (LFO rates in Hz, envelope times), so the file has to be
played at its own tempo, 150 BPM. A warm-up note at the start absorbs Serum's
first-note-after-load difference; the image starts LEAD_S later.
"""

from __future__ import annotations

import struct

from .render import Grid, Look, Note, schedule

BPM = 150.0
PPQ = 480
LEAD_S = 0.8    # warm-up note, then the image starts (2 beats at 150 BPM)


def _vlq(n: int) -> bytes:
    out = [n & 0x7F]
    while n > 0x7F:
        n >>= 7
        out.append(0x80 | (n & 0x7F))
    return bytes(reversed(out))


def notes(g: Grid, lk: Look) -> list[Note]:
    warm = [Note(g.notes[0], 2, 0.0, g.note_s - g.gap_s)]
    return warm + [Note(n.pitch, n.velocity, n.start + LEAD_S, n.duration)
                   for n in schedule(g, lk.ir_len)]


def write_midi(path, notes: list[Note], bpm: float = BPM) -> None:
    tick = lambda s: int(round(s * bpm / 60.0 * PPQ))  # noqa: E731
    ev = []
    for n in notes:
        on, off = tick(n.start), tick(n.start + n.duration)
        ev.append((off, 0, bytes([0x80, n.pitch, 0])))
        ev.append((on, 1, bytes([0x90, n.pitch, int(n.velocity)])))
    ev.sort(key=lambda e: (e[0], e[1]))             # note-offs before note-ons at a tick
    track = b"\x00\xff\x51\x03" + struct.pack(">I", int(round(60e6 / bpm)))[1:]
    track += b"\x00\xff\x58\x04\x04\x02\x18\x08"    # 4/4
    t = 0
    for at, _, data in ev:
        track += _vlq(at - t) + data
        t = at
    track += b"\x00\xff\x2f\x00"
    with open(path, "wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 0, 1, PPQ))
        f.write(b"MTrk" + struct.pack(">I", len(track)) + track)
