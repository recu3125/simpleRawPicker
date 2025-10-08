import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import AppSettings, AppWindow, Photo


def _make_photo(path: str, selected: bool) -> Photo:
    return Photo(
        path=path,
        timestamp=datetime.now(),
        filesize=os.path.getsize(path),
        selected=selected,
    )


def test_unselected_base_removes_raw_and_xmp(tmp_path):
    root = tmp_path
    selected_raw = root / "keep.cr3"
    selected_raw.write_bytes(b"raw-selected")
    selected_xmp = root / "keep.xmp"
    selected_xmp.write_text("xmp-selected", encoding="utf-8")

    unselected_raw = root / "drop.cr3"
    unselected_raw.write_bytes(b"raw-unselected")
    unselected_xmp = root / "drop.xmp"
    unselected_xmp.write_text("xmp-unselected", encoding="utf-8")

    app = AppWindow.__new__(AppWindow)
    app.settings = AppSettings()
    app.settings.raw_output_folder_name = "_selected_raw"
    app.settings.jpeg_output_folder_name = "_selected_jpeg"

    raw_out_dir = root / app.settings.raw_output_folder_name
    raw_out_dir.mkdir()

    shutil.copy2(selected_raw, raw_out_dir / selected_raw.name)
    shutil.copy2(selected_xmp, raw_out_dir / selected_xmp.name)
    shutil.copy2(unselected_raw, raw_out_dir / unselected_raw.name)
    shutil.copy2(unselected_xmp, raw_out_dir / unselected_xmp.name)

    catalog = SimpleNamespace()
    catalog.root = str(root)
    catalog.photos = [
        _make_photo(str(selected_raw), True),
        _make_photo(str(unselected_raw), False),
    ]

    culling_widget = SimpleNamespace()
    culling_widget.catalog = catalog
    culling_widget.save_all_dirty_files = lambda: None
    app.culling_widget = culling_widget

    app._perform_export()

    assert (raw_out_dir / selected_raw.name).exists()
    assert (raw_out_dir / selected_xmp.name).exists()
    assert not (raw_out_dir / unselected_raw.name).exists()
    assert not (raw_out_dir / unselected_xmp.name).exists()
