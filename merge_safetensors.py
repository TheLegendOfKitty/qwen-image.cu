#!/usr/bin/env python3
"""Merge sharded safetensors files into a single file (streaming, low memory)."""
import json
import os
import struct
import sys


def merge_sharded_safetensors(index_path, output_path):
    base_dir = os.path.dirname(index_path)
    with open(index_path, 'r') as f:
        index = json.load(f)

    weight_map = index['weight_map']

    # Group tensors by shard
    shard_tensors = {}
    for name, shard in weight_map.items():
        shard_tensors.setdefault(shard, []).append(name)

    print(f"Merging {len(weight_map)} tensors from {len(shard_tensors)} shards -> {output_path}")

    # Pass 1: Read all shard headers, compute per-shard data offsets
    shard_headers = {}
    shard_data_starts = {}
    for shard_file in sorted(shard_tensors.keys()):
        path = os.path.join(base_dir, shard_file)
        with open(path, 'rb') as f:
            header_len = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(header_len))
            shard_headers[shard_file] = header
            shard_data_starts[shard_file] = 8 + header_len

    # Build output header with recomputed offsets
    out_header = {}
    current_offset = 0

    # Process shards in sorted order, tensors within each shard in offset order
    copy_plan = []  # [(shard_path, src_offset, length)]
    for shard_file in sorted(shard_tensors.keys()):
        header = shard_headers[shard_file]
        data_start = shard_data_starts[shard_file]
        shard_path = os.path.join(base_dir, shard_file)

        # Sort tensors by their data offset within this shard
        tensors = []
        for name in shard_tensors[shard_file]:
            if name in header:
                info = header[name]
                tensors.append((info['data_offsets'][0], name, info))

        tensors.sort()

        for _, name, info in tensors:
            src_start, src_end = info['data_offsets']
            size = src_end - src_start
            out_header[name] = {
                'dtype': info['dtype'],
                'shape': info['shape'],
                'data_offsets': [current_offset, current_offset + size]
            }
            copy_plan.append((shard_path, data_start + src_start, size))
            current_offset += size

    # Write output
    header_json = json.dumps(out_header, separators=(',', ':')).encode('utf-8')

    with open(output_path, 'wb') as out:
        out.write(struct.pack('<Q', len(header_json)))
        out.write(header_json)

        total = len(copy_plan)
        for i, (shard_path, offset, size) in enumerate(copy_plan):
            with open(shard_path, 'rb') as src:
                src.seek(offset)
                remaining = size
                while remaining > 0:
                    chunk = min(remaining, 64 * 1024 * 1024)  # 64MB chunks
                    out.write(src.read(chunk))
                    remaining -= chunk
            if (i + 1) % 50 == 0 or i == total - 1:
                print(f"  {i+1}/{total} tensors written")

    final_size = 8 + len(header_json) + current_offset
    print(f"Done: {final_size / 1e9:.2f} GB")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: merge_safetensors.py <index.json> <output.safetensors>")
        sys.exit(1)
    merge_sharded_safetensors(sys.argv[1], sys.argv[2])
