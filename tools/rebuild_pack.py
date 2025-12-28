#!/usr/bin/env python3
"""
Rebuild arg-dumps.pack.zst from the original arg-dumps.7z.

Usage:
    python tools/rebuild_pack.py --input arg-dumps.7z --output arg-dumps.pack.zst

This script is intentionally standalone so we can regenerate the pack if the
published artifact is missing or corrupt.
"""

import argparse
import os
import struct
import sys
import tarfile

import zstandard as zstd
try:
    import libarchive
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(
        "error: missing dependency: {}\n"
        "Install with: pip install libarchive-c zstandard\n".format(exc)
    )
    sys.exit(1)


PACK_MAGIC = 0x4F44504B  # 'ODPK'
PACK_VERSION = 2


def normalize_name(name: str) -> str:
    while name.startswith("./"):
        name = name[2:]
    return name


def parse_index(name: str) -> int:
    dash = name.rfind("-")
    dot = name.rfind(".")
    if dash == -1 or dot == -1 or dot <= dash + 1:
        raise ValueError(f"malformed entry name: {name}")
    return int(name[dash + 1 : dot])


def classify(name: str) -> int:
    if name.endswith(".args"):
        return 0
    if name.endswith(".outs"):
        return 1
    return -1


def iter_entries_with_tar(input_tar: str):
    with tarfile.open(input_tar, mode="r:*") as tf:
        members = [m for m in tf.getmembers() if m.isreg()]
        members.sort(key=lambda m: normalize_name(m.name))
        for m in members:
            name = normalize_name(m.name)
            kind = classify(name)
            if kind < 0:
                continue
            data_bytes = tf.extractfile(m).read()
            yield name, kind, data_bytes


def iter_entries_with_libarchive(archive):
    for entry in archive:
        name = normalize_name(entry.pathname)
        kind = classify(name)
        if kind < 0:
            continue
        data_bytes = b"".join(block for block in entry.get_blocks())
        yield name, kind, data_bytes


def build_pack(input_archive: str, output_pack: str, level: int = 3) -> None:
    print(f"[pack] input archive: {input_archive}")
    print(f"[pack] output pack  : {output_pack}")

    is_tar = input_archive.endswith((".tar", ".tar.xz", ".txz", ".tar.gz", ".tgz"))
    is_7z = input_archive.endswith(".7z")
    if not (is_tar or is_7z):
        raise RuntimeError(
            "Unsupported input. Provide .tar/.tar.xz/.tar.gz or .7z (handled via libarchive)."
        )

    compressor = zstd.ZstdCompressor(level=level)

    entries = []
    data_segments = []
    current_offset = 0  # relative to start of data region

    def consume(iterator):
        nonlocal current_offset
        for name, kind, data_bytes in iterator:
            idx = parse_index(name)
            while len(entries) <= idx:
                entries.append(
                    {
                        "offsets": [0, 0],
                        "comp_sizes": [0, 0],
                        "raw_sizes": [0, 0],
                        "seen": [False, False],
                    }
                )
            comp_bytes = compressor.compress(data_bytes)
            entries[idx]["offsets"][kind] = current_offset
            entries[idx]["comp_sizes"][kind] = len(comp_bytes)
            entries[idx]["raw_sizes"][kind] = len(data_bytes)
            entries[idx]["seen"][kind] = True
            data_segments.append(comp_bytes)
            current_offset += len(comp_bytes)
        for i, rec in enumerate(entries):
            if not all(rec["seen"]):
                raise RuntimeError(f"missing data for case {i} (args/outs)")

    if is_tar:
        consume(iter_entries_with_tar(input_archive))
    else:
        with libarchive.file_reader(input_archive) as archive:
            consume(iter_entries_with_libarchive(archive))

    # Offsets are relative to start of data region after header+entries
    header_size = struct.calcsize("<IIII")
    entry_size = struct.calcsize("<QQIIII")
    data_base = header_size + entry_size * len(entries)

    # Rewrite offsets with base
    for rec in entries:
        rec["offsets"][0] += data_base
        rec["offsets"][1] += data_base

    # write pack
    os.makedirs(os.path.dirname(os.path.abspath(output_pack)) or ".", exist_ok=True)
    with open(output_pack, "wb") as f:
        f.write(struct.pack("<IIII", PACK_MAGIC, PACK_VERSION, len(entries), 0))
        for rec in entries:
            f.write(
                struct.pack(
                    "<QQIIII",
                    rec["offsets"][0],
                    rec["offsets"][1],
                    rec["comp_sizes"][0],
                    rec["comp_sizes"][1],
                    rec["raw_sizes"][0],
                    rec["raw_sizes"][1],
                )
            )
        for segment in data_segments:
            f.write(segment)

    print(f"[pack] wrote {len(entries)} cases to {output_pack}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Rebuild arg-dumps.pack.zst from arg-dumps.7z")
    ap.add_argument("--input", "-i", required=True, help="Path to arg-dumps.7z")
    ap.add_argument("--output", "-o", required=True, help="Path to write arg-dumps.pack.zst")
    ap.add_argument("--level", "-l", type=int, default=3, help="zstd compression level (default: 3)")
    args = ap.parse_args()

    try:
        build_pack(args.input, args.output, level=args.level)
    except Exception as exc:  # pragma: no cover
        sys.stderr.write(f"error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

