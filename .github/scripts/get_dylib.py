"""从 onnxruntime wheel 中提取 libonnxruntime*.dylib 到 dist/。

用法：python3 .github/scripts/get_dylib.py <wheel 目录> <输出目录>
"""

import glob
import os
import shutil
import sys
import zipfile


def main() -> int:
    wheel_dir = sys.argv[1] if len(sys.argv) > 1 else "whl"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "dist"

    wheels = sorted(glob.glob(os.path.join(wheel_dir, "onnxruntime-*.whl")))
    if not wheels:
        print(f"未找到 wheel: {wheel_dir}", file=sys.stderr)
        return 1
    wheel = wheels[0]
    print(f"wheel: {wheel}")

    os.makedirs(out_dir, exist_ok=True)
    found = 0
    with zipfile.ZipFile(wheel) as zf:
        for name in zf.namelist():
            base = os.path.basename(name)
            if not base.startswith("libonnxruntime") or not base.endswith(".dylib"):
                continue
            target = os.path.join(out_dir, base)
            with zf.open(name) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            size = os.path.getsize(target)
            print(f"提取: {target} ({size / 1048576:.1f} MB)")
            found += 1

    if not found:
        print("wheel 内未找到 libonnxruntime*.dylib", file=sys.stderr)
        return 1
    print(f"完成，共 {found} 个 dylib")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
