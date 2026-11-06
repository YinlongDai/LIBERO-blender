"""Messages between the LIBERO env process and the Blender render server.

A message is an 8-byte big-endian header length, a JSON header, then the raw bytes of the arrays
listed in header["arrays"] (name, dtype, shape), in that order. Plain TCP and the standard library
plus numpy, so it runs unchanged on the env's Python 3.8 and on Blender's bundled Python.
"""
import json
import socket
import struct
import time

import numpy as np

HEADER_LENGTH = struct.Struct(">Q")


def connect(port, host="127.0.0.1", timeout_s=600.0):
    """Connect to a server, retrying while it starts up (Blender or a policy checkpoint can take minutes)."""
    deadline = time.time() + timeout_s
    while True:
        try:
            sock = socket.create_connection((host, port))
            break
        except ConnectionRefusedError:
            if time.time() > deadline:
                raise TimeoutError(f"nothing listening on {host}:{port} after {timeout_s} s")
            time.sleep(0.5)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


def send_message(sock, header, arrays=None):
    arrays = arrays or {}
    header = dict(header, arrays=[{"name": name, "dtype": array.dtype.str, "shape": list(array.shape)}
                                  for name, array in arrays.items()])
    encoded = json.dumps(header).encode()
    sock.sendall(HEADER_LENGTH.pack(len(encoded)) + encoded)
    for array in arrays.values():
        sock.sendall(np.ascontiguousarray(array).tobytes())


def recv_exactly(sock, num_bytes):
    buffer = bytearray(num_bytes)
    view = memoryview(buffer)
    received = 0
    while received < num_bytes:
        count = sock.recv_into(view[received:], num_bytes - received)
        if count == 0:
            raise ConnectionError("the other side closed the connection")
        received += count
    return buffer


def recv_message(sock):
    (length,) = HEADER_LENGTH.unpack(recv_exactly(sock, HEADER_LENGTH.size))
    header = json.loads(recv_exactly(sock, length).decode())
    arrays = {}
    for spec in header.pop("arrays"):
        dtype = np.dtype(spec["dtype"])
        num_bytes = int(np.prod(spec["shape"])) * dtype.itemsize
        arrays[spec["name"]] = np.frombuffer(recv_exactly(sock, num_bytes), dtype=dtype).reshape(spec["shape"])
    return header, arrays
