"""Minimal CBOR codec that round-trips Serum 2 payloads byte-for-byte.

Serum mixes float32 and float64 encodings; generic encoders widen everything
to float64. Here float32 values decode to F32 (a float subclass) and encode
back as float32, so decode->encode is exact.
"""

import struct


class F32(float):
    """A float that was (or should be) stored as CBOR float32."""

    __slots__ = ()


class F16(float):
    __slots__ = ()


class Tagged:
    __slots__ = ("tag", "value")

    def __init__(self, tag, value):
        self.tag = tag
        self.value = value

    def __eq__(self, other):
        return isinstance(other, Tagged) and (self.tag, self.value) == (other.tag, other.value)

    def __repr__(self):
        return f"Tagged({self.tag}, {self.value!r})"


def loads(data: bytes):
    value, pos = _decode(memoryview(data), 0)
    if pos != len(data):
        raise ValueError(f"trailing bytes: {len(data) - pos}")
    return value


def _read_arg(buf, pos, info):
    if info < 24:
        return info, pos
    if info == 24:
        return buf[pos], pos + 1
    if info == 25:
        return struct.unpack_from(">H", buf, pos)[0], pos + 2
    if info == 26:
        return struct.unpack_from(">I", buf, pos)[0], pos + 4
    if info == 27:
        return struct.unpack_from(">Q", buf, pos)[0], pos + 8
    raise ValueError(f"unsupported additional info {info} (indefinite length?)")


def _decode(buf, pos):
    ib = buf[pos]
    pos += 1
    major, info = ib >> 5, ib & 0x1F
    if major == 7:
        if info == 20:
            return False, pos
        if info == 21:
            return True, pos
        if info == 22:
            return None, pos
        if info == 25:
            return F16(struct.unpack_from(">e", buf, pos)[0]), pos + 2
        if info == 26:
            return F32(struct.unpack_from(">f", buf, pos)[0]), pos + 4
        if info == 27:
            return struct.unpack_from(">d", buf, pos)[0], pos + 8
        raise ValueError(f"unsupported simple value {info}")
    arg, pos = _read_arg(buf, pos, info)
    if major == 0:
        return arg, pos
    if major == 1:
        return -1 - arg, pos
    if major == 2:
        return bytes(buf[pos:pos + arg]), pos + arg
    if major == 3:
        return bytes(buf[pos:pos + arg]).decode("utf-8"), pos + arg
    if major == 4:
        out = []
        for _ in range(arg):
            v, pos = _decode(buf, pos)
            out.append(v)
        return out, pos
    if major == 5:
        out = {}
        for _ in range(arg):
            k, pos = _decode(buf, pos)
            v, pos = _decode(buf, pos)
            out[k] = v
        return out, pos
    if major == 6:
        v, pos = _decode(buf, pos)
        return Tagged(arg, v), pos
    raise ValueError(f"bad major type {major}")


def dumps(value) -> bytes:
    out = bytearray()
    _encode(value, out)
    return bytes(out)


def _head(major, n, out):
    if n < 24:
        out.append((major << 5) | n)
    elif n < 0x100:
        out += bytes(((major << 5) | 24, n))
    elif n < 0x10000:
        out.append((major << 5) | 25)
        out += struct.pack(">H", n)
    elif n < 0x100000000:
        out.append((major << 5) | 26)
        out += struct.pack(">I", n)
    else:
        out.append((major << 5) | 27)
        out += struct.pack(">Q", n)


def _encode(v, out):
    if v is None:
        out.append(0xF6)
    elif v is True:
        out.append(0xF5)
    elif v is False:
        out.append(0xF4)
    elif isinstance(v, int):
        if v >= 0:
            _head(0, v, out)
        else:
            _head(1, -1 - v, out)
    elif isinstance(v, F32):
        out.append(0xFA)
        out += struct.pack(">f", v)
    elif isinstance(v, F16):
        out.append(0xF9)
        out += struct.pack(">e", v)
    elif isinstance(v, float):
        out.append(0xFB)
        out += struct.pack(">d", v)
    elif isinstance(v, str):
        b = v.encode("utf-8")
        _head(3, len(b), out)
        out += b
    elif isinstance(v, (bytes, bytearray)):
        _head(2, len(v), out)
        out += v
    elif isinstance(v, (list, tuple)):
        _head(4, len(v), out)
        for x in v:
            _encode(x, out)
    elif isinstance(v, dict):
        _head(5, len(v), out)
        for k, x in v.items():
            _encode(k, out)
            _encode(x, out)
    elif isinstance(v, Tagged):
        _head(6, v.tag, out)
        _encode(v.value, out)
    else:
        raise TypeError(f"cannot encode {type(v).__name__}")
