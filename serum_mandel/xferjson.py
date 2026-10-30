"""Xfer container format used by .SerumPreset files and Serum 2 VST3 state.

Layout:
    b"XferJson\\0"
    u64 LE   length of JSON header
    JSON header (metadata; "hash" = md5 hex of the zstd payload)
    u32 LE   uncompressed payload size
    u32 LE   payload format version (2 observed)
    zstd frame containing a CBOR document
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field

import zstandard

from . import cbor

MAGIC = b"XferJson\x00"


@dataclass
class XferDoc:
    header: dict
    body: object
    payload_version: int = 2
    raw_cbor: bytes | None = field(default=None, repr=False)


def load_bytes(data: bytes) -> XferDoc:
    if not data.startswith(MAGIC):
        raise ValueError("not an XferJson container")
    pos = len(MAGIC)
    (hlen,) = struct.unpack_from("<Q", data, pos)
    pos += 8
    header = json.loads(data[pos:pos + hlen])
    pos += hlen
    usize, version = struct.unpack_from("<II", data, pos)
    pos += 8
    raw = zstandard.ZstdDecompressor().decompress(data[pos:], max_output_size=max(usize, 1))
    if len(raw) != usize:
        raise ValueError(f"payload size mismatch: {len(raw)} != {usize}")
    return XferDoc(header, cbor.loads(raw), version, raw)


def load(path) -> XferDoc:
    with open(path, "rb") as f:
        return load_bytes(f.read())


def dump_bytes(doc: XferDoc, level: int = 3) -> bytes:
    raw = cbor.dumps(doc.body)
    compressed = zstandard.ZstdCompressor(level=level).compress(raw)
    header = dict(doc.header)
    header["hash"] = hashlib.md5(compressed).hexdigest()
    # Compact separators match Serum's own header JSON.
    hjson = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return b"".join([
        MAGIC,
        struct.pack("<Q", len(hjson)),
        hjson,
        struct.pack("<II", len(raw), doc.payload_version),
        compressed,
    ])


def dump(doc: XferDoc, path, level: int = 3) -> None:
    with open(path, "wb") as f:
        f.write(dump_bytes(doc, level))
