#!/usr/bin/env python3
# Convert a GGUF split shard to a standalone GGUF for single-shard quantization.

import sys
from pathlib import Path
import gguf

def make_standalone(src_path: Path, dst_path: Path):
    reader = gguf.GGUFReader(str(src_path))
    arch = bytes(reader.fields["general.architecture"].parts[0]).decode("utf-8")
    writer = gguf.GGUFWriter(str(dst_path), arch)

    for field in reader.fields.values():
        if field.name in ["split.no", "split.count", "general.architecture"]:
            continue
        vtype = field.types[0]
        if vtype == gguf.GGUFValueType.STRING:
            val = bytes(field.parts[field.data[0]]).decode("utf-8", errors="ignore")
            writer.add_string(field.name, val)
        elif vtype in [gguf.GGUFValueType.UINT32, gguf.GGUFValueType.INT32, gguf.GGUFValueType.UINT64, gguf.GGUFValueType.INT64]:
            writer.add_uint32(field.name, int(field.parts[field.data[0]][0]))
        elif vtype in [gguf.GGUFValueType.FLOAT32, gguf.GGUFValueType.FLOAT64]:
            writer.add_float32(field.name, float(field.parts[field.data[0]][0]))
        elif vtype == gguf.GGUFValueType.BOOL:
            writer.add_bool(field.name, bool(field.parts[field.data[0]][0]))

    # Copy tensors from reader to writer
    for tensor in reader.tensors:
        writer.add_tensor(
            tensor.name,
            tensor.data,
            raw_dtype=tensor.tensor_type
        )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Successfully created standalone shard at: {dst_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 make_standalone_shard.py <input.gguf> <output.gguf>")
        sys.exit(1)
    make_standalone(Path(sys.argv[1]), Path(sys.argv[2]))
