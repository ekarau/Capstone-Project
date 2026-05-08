"""Package the project + unified dataset into two ZIPs for Colab upload.

Produces:
  Desktop/colab_upload/
    code.zip       (configs, src, scripts, tools, notebooks — small ~50KB)
    dataset.zip    (data/unified — large, ~1-3 GB depending on aug)

Workflow:
  1. Run this script.
  2. Upload BOTH zips to Google Drive at MyDrive/Capstone/.
  3. Open notebooks/02_train_colab.ipynb in Colab.
  4. The first cell of that notebook unzips both into the right place.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT.parent / "colab_upload"
OUT_DIR.mkdir(exist_ok=True)


def zip_dir(src: Path, dst_zip_no_ext: Path, *, include: list[str] | None = None) -> Path:
    """Zip src directory contents into dst_zip_no_ext + '.zip'.

    If include is given, only top-level entries with these names are kept.
    """
    if include is None:
        return Path(shutil.make_archive(str(dst_zip_no_ext), "zip", str(src)))

    # Build a temporary staging dir with only the requested entries
    staging = OUT_DIR / "_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    for name in include:
        s = src / name
        if not s.exists():
            print(f"  skip missing: {name}")
            continue
        d = staging / name
        if s.is_dir():
            shutil.copytree(s, d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(s, d)
    z = shutil.make_archive(str(dst_zip_no_ext), "zip", str(staging))
    shutil.rmtree(staging)
    return Path(z)


def main() -> None:
    print(f"Packaging from: {ROOT}")
    print(f"Output dir   : {OUT_DIR}\n")

    print("[1/2] Code ZIP (configs, src, scripts, tools, notebooks, requirements)")
    code_zip = zip_dir(
        ROOT,
        OUT_DIR / "code",
        include=[
            "configs",
            "src",
            "scripts",
            "tools",
            "notebooks",
            "requirements.txt",
            "README.md",
        ],
    )
    print(f"  -> {code_zip}  ({code_zip.stat().st_size / 1024:.1f} KB)\n")

    data_unified = ROOT / "data" / "unified"
    if data_unified.exists():
        print("[2/2] Dataset ZIP (data/unified — train/val/test + data.yaml)")
        dataset_zip = zip_dir(data_unified, OUT_DIR / "dataset")
        size_mb = dataset_zip.stat().st_size / (1024 * 1024)
        print(f"  -> {dataset_zip}  ({size_mb:.1f} MB)\n")
    else:
        print("[2/2] data/unified yok — once 'python -m scripts.prepare_dataset' calistirin.\n")
        return

    print("=" * 60)
    print("DONE. Sirada:")
    print(f"  1. {OUT_DIR}/code.zip ve dataset.zip dosyalarini Google Drive'a yukleyin")
    print("     (MyDrive/Capstone/ klasoru altinda olmalilar)")
    print("  2. Colab'da notebooks/02_train_colab.ipynb'i acin (Drive uzerinden)")
    print("  3. Ilk hucre otomatik olarak unzip eder ve egitime hazir hale getirir")
    print("=" * 60)


if __name__ == "__main__":
    main()
