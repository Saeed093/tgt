"""
Run target identification on one or more image paths (BGR files).

Example:
  python scripts/run_target_pipeline.py path/to/figure1.png path/to/figure2.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as script from backend/: python scripts/run_target_pipeline.py
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2  # noqa: E402

from app.target_pipeline import TargetIdentificationPipeline, annotate_debug  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Identify figure_1 / figure_2 targets in images.")
    p.add_argument("images", nargs="+", help="Paths to PNG/JPG images")
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="If set, write annotated JPGs and JSON sidecars here",
    )
    args = p.parse_args()

    pipeline = TargetIdentificationPipeline()
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for path_str in args.images:
        path = Path(path_str)
        if not path.is_file():
            print(f"[skip] not a file: {path}")
            continue
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"[skip] cannot read: {path}")
            continue
        r = pipeline.identify(frame)
        row = {
            "path": str(path.resolve()),
            "target_type": r.target_type,
            "confidence": round(r.confidence, 4),
            "debug": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.debug.items()},
        }
        print(json.dumps(row, indent=2))
        if out_dir:
            stem = path.stem
            ann = annotate_debug(frame, r)
            cv2.imwrite(str(out_dir / f"{stem}_annotated.jpg"), ann)
            (out_dir / f"{stem}_result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
