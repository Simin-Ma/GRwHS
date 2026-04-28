from __future__ import annotations

import json
from pathlib import Path

import pyreadstat


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "real" / "nhanes_2003_2004" / "raw"
OUT_DIR = ROOT / "tmp" / "nhanes_xpt_columns"

FILES = [
    "DEMO_C.xpt",
    "BMX_C.xpt",
    "L40_C.xpt",
    "L06BMT_C.xpt",
    "L24PH_C.xpt",
    "L28OCP_C.xpt",
    "L28PBE_C.xpt",
    "L31PAH_C.xpt",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {}
    for name in FILES:
        path = RAW_DIR / name
        df, meta = pyreadstat.read_xport(str(path))
        cols = [str(col) for col in df.columns.tolist()]
        payload = {
            "file": name,
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "columns": cols,
        }
        out_path = OUT_DIR / f"{path.stem}_columns.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        manifest[name] = {
            "shape": payload["shape"],
            "path": str(out_path),
        }
        print(name, payload["shape"], out_path)
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
