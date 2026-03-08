from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict
from urllib.request import urlopen


NHANES_2003_2004_FILES: Dict[str, str] = {
    "DEMO_C.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/DEMO_C.xpt",
    "L40_C.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/L40_C.xpt",
    "L16_C.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/L16_C.xpt",
    "BMX_C.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/BMX_C.xpt",
    "L06BMT_C.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/L06BMT_C.xpt",
    "L24PH_C.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/L24PH_C.xpt",
    "L28OCP_C.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/L28OCP_C.xpt",
    "L28PBE_C.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/L28PBE_C.xpt",
    "L31PAH_C.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/L31PAH_C.xpt",
}


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response:
        payload = response.read()
    dest.write_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NHANES 2003-2004 XPT files used by the GRRHS real-data pipeline.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/real/nhanes_2003_2004/raw"),
        help="Destination directory for downloaded XPT files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite files even if they already exist.",
    )
    args = parser.parse_args()

    manifest = {"files": []}
    for name, url in NHANES_2003_2004_FILES.items():
        dest = args.dest / name
        if dest.exists() and not args.force:
            status = "skipped"
        else:
            download_file(url, dest)
            status = "downloaded"
        manifest["files"].append(
            {
                "name": name,
                "url": url,
                "path": str(dest.resolve()),
                "status": status,
                "bytes": dest.stat().st_size if dest.exists() else None,
            }
        )
        print(f"[{status}] {name} -> {dest}")

    manifest_path = args.dest / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[ok] manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
