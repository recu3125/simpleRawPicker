#!/usr/bin/env python3

import os, sys, io, json, time, shutil, argparse, traceback
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import threading
from collections import OrderedDict
from queue import PriorityQueue, Empty
from contextlib import contextmanager
from html import escape as _h

try:
    import exiv2
except Exception:
    exiv2 = None

import numpy as np
from PIL import Image, ImageQt, ImageOps
import rawpy

from PySide6.QtCore import Qt, QSize, QRect, QRectF, QPoint, QTimer, QObject, Signal, Slot, QEvent, QElapsedTimer, QStandardPaths, QThread
from PySide6.QtGui import QPixmap, QKeySequence, QAction, QPainter, QPen, QColor, QFontDatabase, QFont, QIcon, QImage, QPolygon, QPainterPath, QBrush
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QFileDialog, QMessageBox, QFrame,
    QStatusBar, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QStackedWidget,
    QToolBar, QDialog, QFormLayout, QSpinBox, QLineEdit, QDialogButtonBox,
    QSizePolicy, QGroupBox, QGraphicsDropShadowEffect, QRadioButton, QSpacerItem,
    QProgressDialog
)

try:
    import psutil
except Exception:
    psutil = None

def _prof_enabled():
    app = QApplication.instance()
    return bool(getattr(app, "_profile_enabled", False)) if app else False

def _plog(msg: str):
    if _prof_enabled():
        print(f"[{time.perf_counter():.3f}] [{threading.current_thread().name}] {msg}")

@contextmanager
def _ptime(label: str, warn_ms: float = 16.0):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = (time.perf_counter() - t0) * 1000.0
        if _prof_enabled() and dt > warn_ms:
            print(f"[PROF] {label}: {dt:.1f} ms")

SUPPORTED_EXTS = {
    '.cr3', '.cr2', '.nef', '.arw', '.raf', '.dng', '.orf', '.rw2', '.srw', '.pef'
}

@dataclass
class ExportJob:
    root: str
    selected_raw_paths: List[str]
    raw_output_folder_name: str
    jpeg_output_folder_name: str


class ExportCancelledError(Exception):
    pass


class ExportWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(tuple)
    failed = Signal(str, str)
    cancelled = Signal()

    def __init__(self, job: ExportJob, perform_export_fn):
        super().__init__()
        self.job = job
        self._perform_export_fn = perform_export_fn
        self._cancel_event = threading.Event()

    @Slot()
    def run(self):
        try:
            result = self._perform_export_fn(
                self.job,
                self._emit_progress,
                self._cancel_event.is_set
            )
            if self._cancel_event.is_set():
                self.cancelled.emit()
                return
            self.finished.emit(result)
        except ExportCancelledError:
            self.cancelled.emit()
        except Exception as e:
            self.failed.emit(str(e), traceback.format_exc())

    def cancel(self):
        self._cancel_event.set()

    def _emit_progress(self, current: int, total: int, message: str):
        self.progress.emit(current, total, message)

_XMP_GLOBAL_LOCK = threading.Lock()
def read_xmp_sidecar(path: str) -> Dict:
    """Reads rating, label, and pick status from an XMP sidecar file."""
    if exiv2 is None: return {}
    xmp_path = os.path.splitext(path)[0] + '.xmp'

    with _XMP_GLOBAL_LOCK:
        try:
            if not os.path.exists(xmp_path) or os.path.getsize(xmp_path) == 0:
                return {}
        except FileNotFoundError:
            return {} 

        for attempt in range(3):
            try:
                img = exiv2.ImageFactory.open(xmp_path)
                img.readMetadata()
                xmp = img.xmpData()
                
                data = {}
                if 'Xmp.xmp.Rating' in xmp:
                    data['rating'] = int(xmp['Xmp.xmp.Rating'].print())
                if 'Xmp.xmp.Label' in xmp:
                    data['color_label'] = xmp['Xmp.xmp.Label'].print()
                if 'Xmp.photoshop.Urgency' in xmp:
                    urgency_val = int(xmp['Xmp.photoshop.Urgency'].print())
                    data['selected'] = urgency_val == 1
                return data
            except exiv2.Exiv2Error as e:
                if e.code == 21 and attempt < 2: 
                    time.sleep(0.05)
                    continue
                print(f"Warning: Could not read XMP for {os.path.basename(path)}: {e}")
                return {}
            except Exception as e:
                print(f"Warning: Unexpected error reading XMP for {os.path.basename(path)}: {e}")
                return {}
        return {}

def write_xmp_sidecar(path: str, data: Dict):
    """Writes rating, label, or pick status to an XMP sidecar file."""
    if exiv2 is None: return False
    xmp_path = os.path.splitext(path)[0] + '.xmp'

    with _XMP_GLOBAL_LOCK:
        try:
            raw_img = exiv2.ImageFactory.open(path)
            raw_img.readMetadata()
            
            if os.path.exists(xmp_path):
                try:
                    sidecar_img = exiv2.ImageFactory.open(xmp_path)
                    sidecar_img.readMetadata()
                    raw_img.setXmpData(sidecar_img.xmpData())
                except Exception:
                    pass

            xmp = raw_img.xmpData()

            if 'rating' in data and data['rating'] is not None:
                rating_val = int(data['rating'])
                if rating_val == 0 and 'Xmp.xmp.Rating' in xmp:
                     xmp.erase(xmp.findKey(exiv2.XmpKey('Xmp.xmp.Rating')))
                elif rating_val > 0:
                    xmp['Xmp.xmp.Rating'] = str(rating_val)
            
            if 'color_label' in data:
                label_val = data['color_label']
                if label_val:
                    xmp['Xmp.xmp.Label'] = str(label_val)
                elif 'Xmp.xmp.Label' in xmp:
                    xmp.erase(xmp.findKey(exiv2.XmpKey('Xmp.xmp.Label')))

            if 'selected' in data and data['selected'] is not None:
                is_selected = data['selected']
                urgency_key_str = 'Xmp.photoshop.Urgency'
                
                if is_selected:
                    xmp[urgency_key_str] = '1'
                elif urgency_key_str in xmp:
                    xmp[urgency_key_str] = '0'

            with open(xmp_path, 'w', encoding='utf-8') as f:
                f.write(raw_img.xmpPacket())

        except Exception as e:
            print(f"Error writing XMP for {os.path.basename(path)}: {e}")
            return False

    return True

def read_exif_datetime(path: str, st: Optional[os.stat_result] = None) -> Optional[datetime]:
    try:
        st = st or os.stat(path)
        ts = getattr(st, 'st_mtime_ns', None)
        if ts is not None:
            return datetime.fromtimestamp(ts / 1e9)
        return datetime.fromtimestamp(st.st_mtime)
    except Exception:
        return None

def _pow2(x):
    try:
        return 2.0 ** float(x)
    except (ValueError, TypeError):
        return None

def _ratarr_to_tuple(v):
    try:
        return tuple(float(x) for x in v)
    except Exception:
        try:
            return tuple(x.numerator / x.denominator for x in v)
        except Exception:
            return None

def _exif_to_meta(exif, meta: Dict[str, str]):
    def get(tagid):
        return exif.get(tagid)
    def updf(key, val):
        if val is None: return
        if not meta.get(key): meta[key] = val

    TAG = {
        'Make': 271, 'Model': 272,
        'LensModel': 0xA434, 'LensInfo': 0xA432,
        'FNumber': 0x829D, 'ExposureTime': 0x829A,
        'ShutterSpeedValue': 0x9201, 'ApertureValue': 0x9202,
        'ISOSpeedRatings': 0x8827, 'PhotographicSensitivity': 0x8833,
        'FocalLength': 0x920A,
        'DateTimeOriginal': 0x9003, 'DateTimeDigitized': 0x9004, 'DateTime': 0x0132,
        'SubSecTimeOriginal': 0x9291,'SubSecTimeDigitized': 0x9292,'SubSecTime': 0x9290,
    }

    updf('make',  get(TAG['Make'])  and str(get(TAG['Make'])))
    updf('model', get(TAG['Model']) and str(get(TAG['Model'])))

    lens_model = get(TAG['LensModel'])
    if lens_model:
        updf('lens', str(lens_model))
    else:
        li = get(TAG['LensInfo'])
        li = _ratarr_to_tuple(li) if li is not None else None
        if li and len(li) >= 4:
            fl_min, fl_max, f_at_min, f_at_max = li[:4]
            if abs(fl_min - fl_max) < 1e-3:
                updf('lens', f"{fl_min:g}mm f/{f_at_min:.1f}")
            else:
                updf('lens', f"{fl_min:g}-{fl_max:g}mm f/{min(f_at_min, f_at_max):.1f}")

    fno = get(TAG['FNumber'])
    if fno is not None:
        try:
            f = float(fno) if isinstance(fno, (int, float)) else fno.numerator / fno.denominator
            updf('fnumber', f"{f:.1f}")
        except Exception:
            pass
    if not meta.get('fnumber'):
        av = get(TAG['ApertureValue'])
        try:
            avf = float(av) if isinstance(av, (int, float)) else av.numerator / av.denominator
            f = _pow2(avf / 2.0)
            if f is not None:
                updf('fnumber', f"{f:.1f}")
        except Exception:
            pass

    exp = get(TAG['ExposureTime'])
    if exp is not None:
        try:
            e = float(exp) if isinstance(exp, (int, float)) else exp.numerator / exp.denominator
            if e >= 1:
                s = f"{e:.3f}".rstrip('0').rstrip('.')
            else:
                denom = int(round(1.0 / e)) if e > 0 else None
                s = f"1/{denom}" if denom else f"{e:.4f}".rstrip('0').rstrip('.')
            updf('exp', s)
        except Exception:
            pass
    if not meta.get('exp'):
        tv = get(TAG['ShutterSpeedValue'])
        try:
            tvf = float(tv) if isinstance(tv, (int, float)) else tv.numerator / tv.denominator
            t = _pow2(-tvf)
            if t is not None:
                if t >= 1:
                    s = f"{t:.3f}".rstrip('0').rstrip('.')
                else:
                    denom = int(round(1.0 / t)) if t > 0 else None
                    s = f"1/{denom}" if denom else f"{t:.4f}".rstrip('0').rstrip('.')
                updf('exp', s)
        except Exception:
            pass

    iso = get(TAG['PhotographicSensitivity']) or get(TAG['ISOSpeedRatings'])
    if iso is not None:
        updf('iso', str(iso))

    fl = get(TAG['FocalLength'])
    if fl is not None:
        try:
            val = float(fl) if isinstance(fl, (int, float)) else fl.numerator / fl.denominator
            updf('fl', f"{val:g}mm")
        except Exception:
            pass

    base = get(TAG['DateTimeOriginal']) or get(TAG['DateTimeDigitized']) or get(TAG['DateTime'])
    sub  = get(TAG['SubSecTimeOriginal']) or get(TAG['SubSecTimeDigitized']) or get(TAG['SubSecTime'])
    if base and not meta.get('dt'):
        base_s = str(base)
        if sub:
            ssub = ''.join(ch for ch in str(sub) if ch.isdigit())[:6]
            if ssub:
                try:
                    dt = datetime.strptime(base_s, '%Y:%m:%d %H:%M:%S')
                    dt = dt.replace(microsecond=int((ssub + '000000')[:6]))
                    updf('dt', dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
                except Exception:
                    updf('dt', base_s)
        else:
            updf('dt', base_s)

def _open_jpeg_transposed(path: str) -> Image.Image:
    img = Image.open(path); img.load()
    try: img = ImageOps.exif_transpose(img)
    except Exception: pass
    return img

def load_half_pil(path: str) -> Optional[Image.Image]:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in {'.jpg', '.jpeg'}:
            pil = _open_jpeg_transposed(path)
            if max(pil.size) > 2400:
                w, h = pil.size
                pil = pil.resize((max(1,w//2), max(1,h//2)), Image.BILINEAR)
            return pil
        with _ptime(f"rawpy half postprocess {os.path.basename(path)}", warn_ms=40):
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True, no_auto_bright=True, half_size=True,
                    output_bps=8, gamma=(2.222, 4.5),
                    output_color=rawpy.ColorSpace.sRGB,
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR
                )
            return Image.fromarray(rgb)
    except Exception:
        return None

def load_full_pil(path: str) -> Optional[Image.Image]:
    ext = os.path.splitext(path)[1]
    try:
        if ext.lower() in {'.jpg', '.jpeg'}:
            return _open_jpeg_transposed(path)
        with _ptime(f"rawpy full postprocess {os.path.basename(path)}", warn_ms=80):
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True, no_auto_bright=True, half_size=False,
                    output_bps=8, gamma=(2.222, 4.5),
                    output_color=rawpy.ColorSpace.sRGB,
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR
                )
            return Image.fromarray(rgb)
    except Exception:
        return None

def _estimate_pil_bytes(pil: Optional[Image.Image]) -> int:
    if pil is None:
        return 0
    try:
        w, h = pil.size
    except Exception:
        return 0
    if w <= 0 or h <= 0:
        return 0
    try:
        bands = len(pil.getbands())
    except Exception:
        bands = 3
    bands = max(1, bands)
    return int(w * h * bands)

def _jpeg_exif_as_meta(path: str) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    img = _open_jpeg_transposed(path)
    exif = img.getexif()
    _exif_to_meta(exif, meta)
    return meta

def _exiv2_read_meta_text(path: str) -> Dict[str, str]:
    if exiv2 is None:
        return {}
    try:
        img = exiv2.ImageFactory.open(path)
        img.readMetadata()
        exif = img.exifData()
        xmp  = img.xmpData()
        def gp(*keys) -> str:
            for k in keys:
                try:
                    if k in exif:
                        return exif[k].print()
                    if k in xmp:
                        return xmp[k].print()
                except Exception:
                    pass
            return ""
        meta: Dict[str, str] = {}
        meta['make']   = gp("Exif.Image.Make")
        meta['model']  = gp("Exif.Image.Model")
        meta['lens']   = gp("Exif.Photo.LensModel", "Xmp.aux.Lens", "Xmp.exifEX.LensModel", "Xmp.aux.LensID")
        meta['fnumber_text'] = gp("Exif.Photo.FNumber", "Exif.Photo.ApertureValue")
        meta['exp_text']     = gp("Exif.Photo.ExposureTime", "Xmp.exif.ExposureTime")
        meta['iso']    = gp("Exif.Photo.PhotographicSensitivity", "Exif.Photo.ISOSpeedRatings", "Xmp.exif.ISOSpeedRatings")
        meta['fl']     = gp("Exif.Photo.FocalLength", "Xmp.exif.FocalLength")
        meta['dt']     = gp("Exif.Photo.DateTimeOriginal", "Exif.Image.DateTime", "Xmp.exif.DateTimeOriginal", "Xmp.xmp.CreateDate")
        return {k: v for k, v in meta.items() if v}
    except Exception:
        return {}

def _raw_meta_as_meta(path: str) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    try:
        with _ptime(f"raw meta read {os.path.basename(path)}", warn_ms=40):
            with rawpy.imread(path) as raw:
                m = getattr(raw, 'metadata', None)
                if m:
                    make  = getattr(m, 'make', None) or getattr(m, 'camera_make', None)
                    model = getattr(m, 'model', None) or getattr(m, 'camera_model', None)
                    lens  = getattr(m, 'lens', None) or getattr(m, 'lens_model', None)
                    fno   = getattr(m, 'aperture', None)
                    exp   = getattr(m, 'shutter', None)
                    iso   = getattr(m, 'iso_speed', None) or getattr(m, 'iso', None)
                    fl    = getattr(m, 'focal_length', None)
                    ts    = getattr(m, 'timestamp', None)
                    if make:  meta['make']  = str(make)
                    if model: meta['model'] = str(model)
                    if lens:  meta['lens']  = str(lens)
                    if fno and float(fno) > 0:   meta['fnumber'] = f"{float(fno):.1f}"
                    if exp and float(exp) > 0:   meta['exp'] = f"{float(exp):.6f}".rstrip('0').rstrip('.')
                    if iso and float(iso) > 0:   meta['iso'] = str(int(float(iso)))
                    if fl  and float(fl)  > 0:   meta['fl']  = f"{float(fl):g}mm"
                    if ts:
                        try:
                            meta.setdefault('dt', datetime.fromtimestamp(int(ts)).strftime('%Y-%m-%d %H:%M:%S'))
                        except Exception:
                            pass
                try:
                    thumb = raw.extract_thumb()
                    if getattr(thumb, 'format', None) == rawpy.ThumbFormat.JPEG:
                        exif = Image.open(io.BytesIO(thumb.data)).getexif()
                        _exif_to_meta(exif, meta)
                except Exception:
                    pass
    except Exception:
        pass
    ext = _exiv2_read_meta_text(path)
    for k in ('make','model','lens','iso','fl','dt'):
        if not meta.get(k) and ext.get(k):
            meta[k] = ext[k]
    if ext.get('fnumber_text'):
        meta['fnumber_text'] = ext['fnumber_text']
    if ext.get('exp_text'):
        meta['exp_text'] = ext['exp_text']
    return meta

@dataclass
class Photo:
    path: str
    timestamp: datetime
    filesize: int
    selected: bool = False
    rating: int = 0
    color_label: str = ""
    
    xmp_loaded: bool = False  
    is_dirty: bool = False    
    is_saving: bool = False   
    version: int = 0          
    saving_version: int = 0   
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def update_xmp(self, data: Dict):
        """메모리 상태를 변경하고 dirty 플래그를 설정합니다."""
        with self.lock:
            changed = False
            if 'rating' in data and self.rating != data['rating']:
                self.rating = data['rating']
                changed = True
            if 'color_label' in data and self.color_label != data['color_label']:
                self.color_label = data['color_label']
                changed = True
            if 'selected' in data and self.selected != data['selected']:
                self.selected = data['selected']
                changed = True

            if changed:
                self.version += 1
                self.is_dirty = True

class Catalog:
    def __init__(self, root: str):
        self.root = root
        self.photos: List[Photo] = []
        self._index()

    def _iter_files(self):
        try:
            with os.scandir(self.root) as it:
                for entry in it:
                    try:
                        if not entry.is_file():
                            continue
                    except OSError:
                        continue
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext in SUPPORTED_EXTS:
                        yield entry
        except FileNotFoundError:
            print(f"Warning: Directory not found: {self.root}")
            return

    def _index(self):
        entries = list(self._iter_files())
        total = len(entries)
        progress = None
        app = QApplication.instance()
        if app and total >= 200:
            parent = app.activeWindow()
            progress = QProgressDialog("Indexing photos…", "", 0, total, parent)
            progress.setCancelButton(None)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            progress.show()
            QApplication.processEvents()

        items: List[Tuple[datetime, str, int]] = []
        for idx, entry in enumerate(entries, start=1):
            path = entry.path
            try:
                st = entry.stat(follow_symlinks=False)
            except Exception:
                st = None
            dt = read_exif_datetime(path, st)
            if not dt:
                continue
            try:
                sz = st.st_size if st is not None else os.path.getsize(path)
            except Exception:
                sz = 0
            items.append((dt, path, sz))
            if progress:
                if progress.wasCanceled():
                    break
                progress.setValue(idx)
                QApplication.processEvents()

        if progress:
            progress.close()

        items.sort(key=lambda x: x[0])
        self.photos = [Photo(path=p, timestamp=dt, filesize=sz) for dt, p, sz in items]

    def total_photos(self) -> int:
        return len(self.photos)

class ImageView(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setText("Loading…")
        self._rendered_pixmap: Optional[QPixmap] = None
        self._pm_rect: QRect = QRect()
        self.loading: bool = False
        self._selected: bool = False
        self._mode: str = 'fit'
        self._zoom: float = 1.0
        self._top_left: QPoint = QPoint(0, 0)
        self._dragging: bool = False
        self._drag_start: Optional[QPoint] = None
        self._top_left_start: Optional[QPoint] = None
        self._pil_full: Optional[Image.Image] = None
        self.show_zebra: bool = False
        self.show_hdr: bool = False
        self._wheel_queue: int = 0
        self._wheel_pos: Optional[QPoint] = None
        self._zebra_phase: int = 0
        self._zebra_elapsed = QElapsedTimer()
        self._zebra_speed_px_s = 20.0
        self._zebra_timer: QTimer = QTimer(self)
        self._zebra_timer.setTimerType(Qt.PreciseTimer)
        self._zebra_timer.setInterval(60)
        self._zebra_timer.timeout.connect(self._advance_zebra_phase)
        self._hist_cache: Optional[np.ndarray] = None
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self._overlay_cache: Optional[Dict] = None
        self._pil_half: Optional[Image.Image] = None
        self._pil_base: Optional[Image.Image] = None
        self._hdr_lut = None

    def set_rendered_pixmap(self, pixmap: Optional[QPixmap]):
        self._rendered_pixmap = pixmap
        if pixmap and not pixmap.isNull():
            area = self.contentsRect()
            dpr = self.devicePixelRatioF()
            pm_w = pixmap.width() / dpr
            pm_h = pixmap.height() / dpr
            
            x = area.x() + (area.width() - pm_w) // 2
            y = area.y() + (area.height() - pm_h) // 2
            self._pm_rect = QRect(int(x), int(y), int(pm_w), int(pm_h))
        else:
            self._pm_rect = QRect()
        
        self.update()

    def set_pils(self, pil_full: Optional[Image.Image], pil_half: Optional[Image.Image]):
        old_base = self._pil_base
        old_base_size = old_base.size if old_base else None

        self._pil_full = pil_full
        self._pil_half = pil_half
        self._pil_base = pil_full or pil_half
        new_base = self._pil_base
        new_base_size = new_base.size if new_base else None

        if self._mode == 'zoom' and old_base_size and new_base_size and old_base_size != new_base_size:
            old_w, old_h = old_base_size
            if abs(self._zoom) > 1e-6 and old_w > 0:
                view_center = self.rect().center()
                
                img_point_x = (view_center.x() - self._top_left.x()) / self._zoom
                img_point_y = (view_center.y() - self._top_left.y()) / self._zoom

                new_w, new_h = new_base_size
                scale_factor = new_w / old_w

                new_img_point_x = img_point_x * scale_factor
                new_img_point_y = img_point_y * scale_factor

                new_zoom = self._zoom / scale_factor

                new_top_left_x = view_center.x() - new_img_point_x * new_zoom
                new_top_left_y = view_center.y() - new_img_point_y * new_zoom

                self._zoom = new_zoom
                self._top_left = QPoint(int(round(new_top_left_x)), int(round(new_top_left_y)))

        self._hist_cache = None
        self._overlay_cache = None

    def _advance_zebra_phase(self):
        dt_ms = self._zebra_elapsed.restart()
        dt = max(0.0, dt_ms / 1000.0)
        period = max(8, int(getattr(self, "zebra_period", 16)))

        self._zebra_phase = (self._zebra_phase + self._zebra_speed_px_s * dt) % period
        self.update()

    def _src_pix(self) -> Optional[QPixmap]:
        return self._rendered_pixmap

    def _src_pil(self) -> Optional[Image.Image]:
        return self._pil_full

    def _pick_src_pil(self) -> Optional[Image.Image]:
        base = self._pil_base or self._pil_full or self._pil_half
        if base is None:
            return None
        if self._mode == 'fit' and self._pm_rect.isValid():
            s = self._pm_rect.height() / float(base.height) if base.height > 0 else 1.0
        else:
            s = self._zoom
        if s <= 0.75 and self._pil_half is not None:
            return self._pil_half
        return self._pil_full or self._pil_half

    def set_zebra_enabled(self, enabled: bool):
        if self.show_zebra == enabled:
            return
        self.show_zebra = enabled
        if enabled:
            self._zebra_phase = 0
            self._zebra_elapsed.restart()
            self._zebra_timer.start()
        else:
            self._zebra_timer.stop()
        self.update()

    def set_hdr_enabled(self, enabled: bool):
        if self.show_hdr == enabled:
            return
        self.show_hdr = enabled
        self.update()
        
    def set_fit(self):
        self._mode = 'fit'
        self._zoom = 1.0
        self._top_left = QPoint(0, 0)
        self._apply_deferred_wheel()
        self.update()

    def set_loading(self, on: bool):
        self.loading = on; self.update()

    def set_selected(self, on: bool):
        self._selected = on; self.update()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self._src_pix() is not None and self._mode == 'fit':
            self._update_scaled()

    def _update_scaled(self):
        pm = self._src_pix()
        if not pm:
            super().setPixmap(QPixmap())
            self._pm_rect = QRect()
            return

        area = self.contentsRect()
        dpr = self.devicePixelRatioF()
        target_h_px = max(1, int(area.height() * dpr))

        with _ptime(f"pm.scaledToHeight target_h={target_h_px}", warn_ms=10):
            pm2 = pm.scaledToHeight(target_h_px, Qt.SmoothTransformation)
            pm2.setDevicePixelRatio(dpr)

        disp_w = max(1, int(round(pm2.width()  / dpr)))
        disp_h = max(1, int(round(pm2.height() / dpr)))

        x = area.x() + (area.width() - disp_w) // 2
        y = area.y()
        self._pm_rect = QRect(x, y, disp_w, disp_h)
        super().setPixmap(pm2)

    def _draw_histogram_overlay(self, painter: QPainter):
        if not self.show_zebra: return
        pil = self._pil_full or self._pil_half
        if pil is None: return
        if self._hist_cache is None:
            with _ptime("histogram compute (lazy, 512 cap)", warn_ms=12):
                pil_small = pil.convert('L')
                maxdim = max(pil_small.size)
                if maxdim > 512:
                    scale = 512.0 / maxdim
                    pil_small = pil_small.resize((int(pil_small.width * scale), int(pil_small.height * scale)), Image.BILINEAR)
                gray = np.array(pil_small, dtype=np.uint8)
                hist, _ = np.histogram(gray, bins=256, range=(0, 255))
                self._hist_cache = hist
        h_img, w_img = 80, 256
        r = self.rect()
        x0 = r.right() - w_img - 14
        y0 = r.top() + 14
        maxv = max(1, int(self._hist_cache.max()))
        painter.fillRect(QRect(x0-4, y0-4, w_img+8, h_img+8), QColor(0,0,0,160))
        painter.setPen(QPen(QColor("#f0f0f0")))
        for x in range(w_img):
            v = int(self._hist_cache[x] / maxv * h_img)
            painter.drawLine(x0 + x, y0 + h_img, x0 + x, y0 + h_img - v)

    def _visible_geometry(self, src_pil: Optional[Image.Image] = None):
        base = self._pil_base or src_pil or self._pil_full or self._pil_half
        src  = src_pil or (self._pil_full or self._pil_half)
        if base is None or src is None:
            return None

        base_w, base_h = base.width, base.height
        if base_w <= 0 or base_h <= 0 or src.width <= 0 or src.height <= 0:
            return None

        if self._mode == 'fit':
            if self._pm_rect.isValid():
                pm_rect = QRect(self._pm_rect)
                s = pm_rect.height() / float(base_h) if base_h > 0 else 1.0
            else:
                area = self.rect()
                scale_w = area.width() / base_w if base_w > 0 else 1.0
                scale_h = area.height() / base_h if base_h > 0 else 1.0
                s = min(scale_w, scale_h)
                disp_w, disp_h = int(base_w * s), int(base_h * s)
                x = area.x() + (area.width() - disp_w) // 2
                y = area.y() + (area.height() - disp_h) // 2
                pm_rect = QRect(x, y, disp_w, disp_h)
        else:
            pm_rect = QRect(self._top_left, QSize(int(base_w * self._zoom), int(base_h * self._zoom)))
            s = self._zoom

        area = self.rect()
        vis = pm_rect.intersected(area)
        if not vis.isValid():
            return None

        sx = (vis.left() - pm_rect.left()) / s
        sy = (vis.top()  - pm_rect.top())  / s
        sw = vis.width()  / s
        sh = vis.height() / s

        sx = max(0.0, min(sx, base_w))
        sy = max(0.0, min(sy, base_h))
        sw = max(0.0, min(sw, base_w - sx))
        sh = max(0.0, min(sh, base_h - sy))

        scale_x = src.width  / float(base_w) if base_w > 0 else 1.0
        scale_y = src.height / float(base_h) if base_h > 0 else 1.0
        pil_box = (
            int(round(sx * scale_x)),
            int(round(sy * scale_y)),
            int(round((sx + sw) * scale_x)),
            int(round((sy + sh) * scale_y)),
        )

        return {
            "pm_rect": pm_rect,
            "target": QRectF(vis),
            "src_pil": pil_box,
            "scale": s,
            "base_w": base_w,
            "base_h": base_h,
            "src_w": src.width,
            "src_h": src.height,
        }

    def paintEvent(self, ev):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1e1e1e"))

        if self._selected:
            rect_to_draw = self._pm_rect
            if self._mode == 'zoom' and self._pil_base and self._pil_base.width > 0 and self._pil_base.height > 0:
                rect_to_draw = QRect(
                    self._top_left,
                    QSize(int(self._pil_base.width * self._zoom), int(self._pil_base.height * self._zoom))
                )
            
            if rect_to_draw.isValid():
                padding = 6 
                background_frame = rect_to_draw.adjusted(-padding, -padding, padding, padding)
                painter.fillRect(background_frame, QColor("#4CEF50"))

        if (self._rendered_pixmap and not self._rendered_pixmap.isNull() 
                and not self.show_hdr):
            if self._mode == 'fit':
                painter.drawPixmap(self._pm_rect, self._rendered_pixmap)
            else:
                base = self._pil_base
                if base and base.width > 0 and base.height > 0:
                    target_rect = QRect(
                        self._top_left,
                        QSize(int(base.width * self._zoom), int(base.height * self._zoom))
                    )
                    painter.drawPixmap(target_rect, self._rendered_pixmap, self._rendered_pixmap.rect())

        else:
            src = self._pil_half or self._pil_full
            if src:
                geom = self._visible_geometry(src_pil=src)
                if geom:
                    try:
                        tgt = geom["target"]
                        box = geom["src_pil"]
                        dpr = painter.device().devicePixelRatioF()
                        if self._mode == 'fit':
                          self._pm_rect = QRect(int(tgt.x()), int(tgt.y()), int(tgt.width()), int(tgt.height()))

                        quality_scaler = 1.0
                        tw = max(1, int(tgt.width() * dpr * quality_scaler))
                        th = max(1, int(tgt.height() * dpr * quality_scaler))

                        resize_filter = Image.BILINEAR
                        roi = src.crop(box).resize((tw, th), resize_filter)

                        if self.show_hdr:
                            if self._hdr_lut is None:
                                gamma = 0.6
                                lut_array = np.zeros(256, dtype=np.float32)
                                for i in range(256):
                                    x = i / 255.0
                                    if x <= 0.5: lut_array[i] = 0.5 * pow(x / 0.5, gamma)
                                    else: lut_array[i] = 1.0 - 0.5 * pow((1.0 - x) / 0.5, gamma)
                                self._hdr_lut = (lut_array * 255.0).astype(np.uint8).tolist()

                            ycbcr = roi.convert('YCbCr')
                            y, cb, cr = ycbcr.split()
                            y = y.point(self._hdr_lut)
                            roi = Image.merge('YCbCr', (y, cb, cr)).convert('RGB')

                        qimg = ImageQt.ImageQt(roi)
                        painter.drawImage(tgt, qimg)
                    except Exception as e:
                        print(f"Error during temporary paint: {e}")


        self._draw_overlays(painter)
        self._draw_histogram_overlay(painter)

        if self.loading:
            painter.fillRect(self.rect(), QColor(0, 0, 0, 120))
            painter.setPen(Qt.white); f = painter.font(); f.setPointSize(f.pointSize()+6); f.setBold(True); painter.setFont(f)
            painter.drawText(self.rect(), Qt.AlignCenter, "Loading…")
               
    def _draw_overlays(self, painter: QPainter):
        src = self._pick_src_pil()
        if src is None:
            return
        geom = self._visible_geometry(src_pil=src)
        if geom is None:
            return

        tgt: QRectF = geom["target"]; box = geom["src_pil"]
        tw, th = int(tgt.width()), int(tgt.height())
        if tw < 2 or th < 2:
            return

        need_zebra = self.show_zebra
        need_peaking = False
        if not (need_zebra or need_peaking):
            return

        down = 1

        ds_w = max(1, tw // down); ds_h = max(1, th // down)

        roi_key = (box[0], box[1], box[2]-box[0], box[3]-box[1],
                  ds_w, ds_h, self._mode, need_peaking, ('H' if src is self._pil_half else 'F'))

        if not self._overlay_cache or self._overlay_cache.get("key") != roi_key:
            with _ptime(f"overlay ROI prepare {ds_w}x{ds_h}", warn_ms=5):
                roi_for_mask = src.crop(box).resize((ds_w, ds_h), Image.NEAREST)

            with _ptime("overlay mask compute(gray+hi/lo)", warn_ms=6):
                gray_pil = roi_for_mask.convert('L')
                gray_arr = np.array(gray_pil, dtype=np.uint8)
                
                cache = {"key": roi_key, "shape": gray_arr.shape}
                cache["mask_hi"] = (gray_arr >= 251)
                cache["mask_lo"] = (gray_arr <= 4)
                self._overlay_cache = cache

        H, W = self._overlay_cache["shape"][:2]
        overlay = np.zeros((H, W, 4), dtype=np.uint8)
        if need_zebra:
            yy, xx = np.indices((H, W))
            period = 16; duty = 8; phase = self._zebra_phase
            stripe = ((xx + yy + phase) % period) < duty
            mh = self._overlay_cache["mask_hi"]
            ml = self._overlay_cache["mask_lo"]
            overlay[mh & stripe] = [130, 0, 0, 255]
            overlay[(ml & stripe) & ~mh] = [0, 0, 130, 255]
            overlay[(ml & mh) & stripe]  = [255, 0, 255, 180]

        if overlay.any():
            with _ptime("overlay compose+draw", warn_ms=8):
                ov_img_small = Image.fromarray(overlay, mode='RGBA')
                ov_img = ov_img_small.resize((tw, th), Image.NEAREST)
                qimg = ImageQt.ImageQt(ov_img)
                painter.drawImage(tgt, qimg)

    def wheelEvent(self, e):
        pm = self._src_pix()
        delta = e.angleDelta().y()
        if delta == 0:
            return

        if pm is None and self._pil_base is None:
            step = 1 if delta > 0 else -1
            self._wheel_queue += step
            self._wheel_pos = e.position().toPoint()
            return

        if self._mode != 'zoom':
            base = self._pil_base or self._pil_full or self._pil_half
            if base and self._pm_rect.isValid():
                s_fit = self._pm_rect.height() / float(base.height if base.height > 0 else 1)
            else:
                s_fit = 1.0

            self._zoom = max(0.01, s_fit)
            area = self.rect()
            if base:
                img_w = int(base.width * self._zoom)
                img_h = int(base.height * self._zoom)
            else:
                img_w = int(self.width() * self._zoom)
                img_h = int(self.height() * self._zoom)


            self._top_left = QPoint(
                int(area.center().x() - img_w / 2),
                int(area.center().y() - img_h / 2)
            )
            self._mode = 'zoom'
            self.set_rendered_pixmap(None)

        step = 1 if delta > 0 else -1
        factor = 1.25 ** step
        new_zoom = min(12.0, max(0.05, self._zoom * factor))
        c = e.position().toPoint()
        s_old = self._zoom
        T_old = self._top_left

        u_img_x = (c.x() - T_old.x()) / s_old
        u_img_y = (c.y() - T_old.y()) / s_old
        self._zoom = new_zoom
        T_new_x = int(c.x() - u_img_x * self._zoom)
        T_new_y = int(c.y() - u_img_y * self._zoom)
        self._top_left = QPoint(T_new_x, T_new_y)
        self._overlay_cache = None
        self.update()

    def _apply_deferred_wheel(self):
        if not self._wheel_queue or (self._src_pix() is None and self._pil_base is None):
            return

        c = self._wheel_pos or self.rect().center()

        if self._mode != 'zoom':
            base = self._pil_base or self._pil_full or self._pil_half
            if base and self._pm_rect.isValid():
                s_fit = self._pm_rect.height() / float(base.height if base.height > 0 else 1)
            else:
                s_fit = 1.0

            self._zoom = max(0.01, s_fit)
            area = self.rect()
            if base:
                img_w = int(base.width * self._zoom)
                img_h = int(base.height * self._zoom)
            else:
                pm = self._src_pix()
                img_w = int((pm.width() if pm else area.width()) * self._zoom)
                img_h = int((pm.height() if pm else area.height()) * self._zoom)

            self._top_left = QPoint(
                int(area.center().x() - img_w / 2),
                int(area.center().y() - img_h / 2)
            )
            self._mode = 'zoom'
            self.set_rendered_pixmap(None)

        steps = self._wheel_queue
        self._wheel_queue = 0
        step_unit = 1 if steps > 0 else -1

        for _ in range(abs(steps)):
            factor = 1.25 ** step_unit
            s_old = self._zoom
            T_old = self._top_left
            u_img_x = (c.x() - T_old.x()) / s_old
            u_img_y = (c.y() - T_old.y()) / s_old
            self._zoom = min(12.0, max(0.05, self._zoom * factor))
            T_new_x = int(c.x() - u_img_x * self._zoom)
            T_new_y = int(c.y() - u_img_y * self._zoom)
            self._top_left = QPoint(T_new_x, T_new_y)

        self._overlay_cache = None
        self.update()

    def mousePressEvent(self, e):
        if self._mode == 'zoom' and e.button() == Qt.LeftButton:
            self._dragging = True; self._drag_start = e.pos(); self._top_left_start = QPoint(self._top_left)
        super().mousePressEvent(e)
    def mouseMoveEvent(self, e):
        if self._mode == 'zoom' and self._dragging and self._drag_start is not None:
            delta = e.pos() - self._drag_start
            self._top_left = self._top_left_start + delta
            self._overlay_cache = None
            self.update()
        super().mouseMoveEvent(e)
    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._dragging = False
        super().mouseReleaseEvent(e)
                   
class Filmstrip(QWidget):
    COLOR_LABEL_MAP = {
        "Red":    QColor("#fa4a45"),
        "Yellow": QColor("#fbeb1c"),
        "Green":  QColor("#46cf4e"),
        "Blue":   QColor("#0085ff"),
        "Purple": QColor("#ad5bff"),
    }
  
    def __init__(self, loader, on_click, height=120, parent=None):
        super().__init__(parent)
        self.loader = loader
        self.on_click = on_click
        self.setMinimumHeight(height)
        self.items = []
        self.rects = []
        self.setMouseTracking(True)

    def set_items(self, items):
        self.items = items or []
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#1e1e1e"))
        self.rects = []
        if not self.items:
            return
        margin = 8
        h = self.height() - margin*2
        w = int(h * 1.5)

        thumbs: List[Optional[QPixmap]] = []
        for it in self.items:
            pm = self.loader(it['path'], h)
            thumbs.append(pm)

        widths = [w] * len(self.items)

        cur_idx = next((i for i, it in enumerate(self.items) if it.get('current')), 0)
        cur_w = widths[cur_idx]
        center_x = self.rect().center().x()
        y = margin

        cur_left = center_x - cur_w // 2
        cur_rect = QRect(cur_left, y, cur_w, h)

        x_left = cur_left - margin
        for i in range(cur_idx-1, -1, -1):
            w = widths[i]
            r = QRect(x_left - w, y, w, h)
            self._draw_thumb(p, r, thumbs[i], self.items[i])
            self.rects.append((r, self.items[i]['path']))
            x_left = r.left() - margin

        self._draw_thumb(p, cur_rect, thumbs[cur_idx], self.items[cur_idx])
        self.rects.append((cur_rect, self.items[cur_idx]['path']))

        x_right = cur_left + cur_w + margin
        for i in range(cur_idx+1, len(self.items)):
            w = widths[i]
            r = QRect(x_right, y, w, h)
            self._draw_thumb(p, r, thumbs[i], self.items[i])
            self.rects.append((r, self.items[i]['path']))
            x_right = r.right() + 1 + margin

    def _draw_thumb(self, p: QPainter, r: QRect, pm: Optional[QPixmap], it: Dict):
        p.fillRect(r, QColor("#222222"))

        if pm is not None and not pm.isNull():
            scaled_size = pm.size().scaled(r.size(), Qt.KeepAspectRatio)
            
            x = r.x() + (r.width() - scaled_size.width()) / 2
            y = r.y() + (r.height() - scaled_size.height()) / 2
            target_rect = QRect(int(x), int(y), scaled_size.width(), scaled_size.height())
            
            p.drawPixmap(target_rect, pm, pm.rect())
        
        self._draw_rating(p, r, it.get('rating', 0))
        
        labelBorderColor = QColor("#1e1e1e")
        selectedBorderWidth = 2
        currentLineWidth = 6
        
        if it.get('current'):
            pen = QPen(QColor("#aaaaaa")); pen.setWidth(currentLineWidth); p.setPen(pen); p.drawRect(r.adjusted(-3,-3,3,3))
            labelBorderColor = QColor("#aaaaaa")
        
        if it.get('selected'):
            pen = QPen(QColor("#4ffd65")); pen.setWidth(selectedBorderWidth); p.setPen(pen); p.drawRect(r)
            labelBorderColor = QColor("#4ffd65")
            
        self._draw_color_label(p, r, it.get('color_label', ''),labelBorderColor,selectedBorderWidth)
        
    def _draw_color_label(self, p: QPainter, r: QRect, label: str, borderColor: QColor, borderWidth: int):
        if not label or label not in self.COLOR_LABEL_MAP:
            return
        color = self.COLOR_LABEL_MAP[label]
        
        size = r.height() // 3 

        p1 = QPoint(r.left(), r.top())
        p2 = QPoint(r.left() + size, r.top())
        p3 = QPoint(r.left(), r.top() + size)
        triangle = QPolygon([p1, p2, p3])

        p.save()
        p.setRenderHint(QPainter.Antialiasing) 
        
        p.setBrush(color)
        p.setPen(Qt.NoPen) 
        p.drawPolygon(triangle)

        p.setPen(QPen(borderColor,borderWidth)) 
        p.drawLine(p2, p3)
        
        p.restore()

    def _draw_rating(self, p: QPainter, r: QRect, rating: int):
        if rating <= 0:
            return

        star_str_full = "★" * rating + "☆" * (5 - rating)
        available_width = r.width() - 4  

        font = p.font()
        font_size = 12  
        while font_size > 6:  
            font.setPointSize(font_size)
            p.setFont(font)
            fm = p.fontMetrics()
            text_width = fm.horizontalAdvance(star_str_full)
            if text_width <= available_width:
                break  
            font_size -= 1

        fm = p.fontMetrics()
        
        filled_str = "★" * rating
        filled_width = fm.horizontalAdvance(filled_str)
        
        empty_str = "☆" * (5 - rating)
        
        total_width = fm.horizontalAdvance(filled_str + empty_str)
        
        start_x = r.left() + (r.width() - total_width) // 2
        text_rect = QRect(start_x, r.bottom() - fm.height() - 2, total_width, fm.height())

        if filled_str:
            path = QPainterPath()
            path.addText(text_rect.left(), text_rect.top() + fm.ascent(), font, filled_str)

            pen = QPen(QColor("black"), 1)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush) 
            p.drawPath(path)

            brush = QBrush(QColor("#FFD700"))
            p.setBrush(brush)
            p.setPen(Qt.NoPen) 
            p.drawPath(path)
            p.setBrush(Qt.NoBrush)

        empty_rect = QRect(text_rect.left() + filled_width, text_rect.top(), text_rect.width() - filled_width, text_rect.height())
        p.setPen(QColor("#808080"))
        p.drawText(empty_rect, Qt.AlignLeft | Qt.AlignVCenter, empty_str)
    def mousePressEvent(self, e):
        pos = e.position().toPoint()
        for r, path in self.rects:
            if r.contains(pos):
                if callable(self.on_click): self.on_click(path)
                break
        super().mousePressEvent(e)

class LoaderSignals(QObject):
    loaded = Signal(str, str)
    meta = Signal(str, dict)
    xmp = Signal(str, dict)
    load_failed = Signal(str, str)
    xmp_saved = Signal(str)
    xmp_save_failed = Signal(str)
    thumb_ready = Signal(str, int, object)
    resized_pixmap_ready = Signal(str, QSize, object)

@dataclass
class AppSettings:
    autosave_interval_min: int = 1
    output_folder_name: str = "selected"
    raw_output_folder_name: str = "_selected_raw"
    jpeg_output_folder_name: str = "_selected_jpeg"
    hotkeys: Dict[str, str] = field(default_factory=lambda: {
        'next': 'D, Right',
        'prev': 'A, Left',
        'toggle_select': 'Space, S',
        'unselect': 'X',
        'toggle_zebra': 'Q',
        'toggle_hdr': 'E',
        'save': '',
        'export': 'Ctrl+S',
        'help': 'F1',
        'quit': 'Esc'
    })

class CullingWidget(QWidget):
    status_message = Signal(str, int)
    export_requested = Signal()

    def __init__(self, root: str, settings: AppSettings, workers: Optional[int] = None, parent=None):
        super().__init__(parent)
        self.settings = settings

        self.view = ImageView()
        self.filmstrip = Filmstrip(loader=self._filmstrip_loader, on_click=self._on_filmstrip_click, height=104)

        self.meta_bar = QWidget(); self.meta_bar.setObjectName("metaBar")
        mb = QHBoxLayout(self.meta_bar); mb.setContentsMargins(12,8,12,8); mb.setSpacing(8)

        self.meta_left = QLabel(); self.meta_left.setObjectName('metaLeft')
        self.meta_left.setText("")

        self.badge_selected = QLabel("Selected: 0"); self.badge_selected.setObjectName("badge")
        self.badge_zebra =   QLabel("Zebra OFF");    self.badge_zebra.setObjectName("badgeGhost")
        self.badge_hdr   =   QLabel("HDR OFF");      self.badge_hdr.setObjectName("badgeGhost")

        mb.addWidget(self.meta_left, 1)
        mb.addSpacing(8)
        mb.addWidget(self.badge_selected, 0, Qt.AlignRight)
        mb.addWidget(self.badge_zebra,   0, Qt.AlignRight)
        mb.addWidget(self.badge_hdr,     0, Qt.AlignRight)
        

        root_box = QVBoxLayout(self); root_box.setContentsMargins(0,0,0,0); root_box.setSpacing(0)

        card = QWidget(); card.setObjectName("card")
        card_l = QVBoxLayout(card); card_l.setContentsMargins(0,0,0,0); card_l.setSpacing(0)
        self.view.setMinimumHeight(300); card_l.addWidget(self.view, 1)

        film_wrap = QWidget(); film_wrap.setObjectName("filmWrap")
        film_l = QVBoxLayout(film_wrap); film_l.setContentsMargins(0,0,0,0); film_l.setSpacing(0)
        film_l.addWidget(self.filmstrip)
        card_l.addWidget(film_wrap, 0)

        card_l.addWidget(self.meta_bar, 0)
        root_box.addWidget(card, 1)

        self.catalog = Catalog(root)
        self.idx = 0

        self.pil_full_cache: OrderedDict[str, Image.Image] = OrderedDict()
        self.pil_half_cache: OrderedDict[str, Image.Image] = OrderedDict()
        self.resized_pixmap_cache: OrderedDict[Tuple[str, int, int], QPixmap] = OrderedDict()
        self.cache_resized_limit = 32
        self.cache_full_limit = 32
        self.cache_half_limit = 64
        self.cache_lock = threading.Lock()
        self._cache_estimated_bytes: int = 0
        self._cache_item_sizes: Dict[Tuple[str, str], int] = {}
        self._pm_thumb_cache: Dict[Tuple[str,int], QPixmap] = {}
        self._pm_thumb_limit = 256
        self._load_failures: set[str] = set()

        self.signals = LoaderSignals()
        self.signals.loaded.connect(self._on_loaded)
        self.signals.meta.connect(self._on_meta_ready)
        self.signals.xmp.connect(self._on_xmp_ready)
        self.signals.load_failed.connect(self._on_load_failed)
        self.signals.xmp_saved.connect(self._on_xmp_saved)
        self.signals.xmp_save_failed.connect(self._on_xmp_save_failed)
        self.signals.thumb_ready.connect(self._on_thumb_ready)
        self.signals.resized_pixmap_ready.connect(self._on_resized_pixmap_ready)

        self._taskq: PriorityQueue = PriorityQueue()
        self._task_counter = 0
        self._loader_stop = False
        default_workers = max(2, min(8, (os.cpu_count() or 4) - 1))
        self._num_workers = int(workers) if (workers and workers > 0) else default_workers
        self._loader_threads: List[threading.Thread] = []
        for i in range(self._num_workers):
            t = threading.Thread(target=self._loader_loop, daemon=True, name=f"loader-{i}")
            t.start(); self._loader_threads.append(t)

        self._full_lock = threading.Lock()
        self._full_running = 0

        self.selections_path = os.path.join(root, 'selections.json')
        self._load_selections() 

        self.autosave_timer = QTimer(self); self.autosave_timer.setSingleShot(True)
        self.autosave_interval_timer = QTimer(self)
        self.update_autosave_interval()
        self.autosave_timer.timeout.connect(self.save_all_dirty_files)
        self.autosave_interval_timer.timeout.connect(self.save_all_dirty_files)
        self.autosave_interval_timer.start()

        self._pending_tasks = set()
        self._pending_lock = threading.Lock()


        self._schedule_timer = QTimer(self); self._schedule_timer.setSingleShot(True)
        self._schedule_timer.setInterval(80); self._schedule_timer.timeout.connect(self._schedule_loading_plan_fire)

        self._full_delay_timer = QTimer(self); self._full_delay_timer.setSingleShot(True)
        self._full_delay_timer.setInterval(1000); self._full_delay_timer.timeout.connect(self._on_full_delay_fire)
        self._full_wait_target: Optional[str] = None

        self._heavy_load_scheduler = QTimer(self); self._heavy_load_scheduler.setSingleShot(True)
        self._heavy_load_scheduler.setInterval(80); self._heavy_load_scheduler.timeout.connect(self._schedule_heavy_load)

        self._last_input_ts = 0.0
        self._meta_cache: Dict[str, Dict[str,str]] = {}

        self.zebra_toggled = False
        self.hdr_toggled = False
        self._zebra_toggle_key: Optional[Qt.Key] = None
        self._hdr_toggle_key: Optional[Qt.Key] = None

        self.status_restore_timer = QTimer(self); self.status_restore_timer.setSingleShot(True)
        self.status_restore_timer.timeout.connect(self._refresh_statusbar)

        self._create_actions()
        self.update_settings()
        QApplication.instance().installEventFilter(self)

        self._show_current()
        self._schedule_heavy_load()

        if not self.catalog.photos:
            QMessageBox.information(self, "Information", "No supported photo files found.")
        elif exiv2 is None:
            QMessageBox.warning(self, "XMP 기능 비활성화",
                                "py3exiv2 라이브러리를 찾을 수 없습니다.\n"
                                "등급 및 색상 라벨 기능이 비활성화됩니다.\n"
                                "`pip install py3exiv2`를 실행하여 설치해주세요.")

        self.setStyleSheet("""
            QWidget#card {
                background:#2b2b2b;
                border:1px solid #3a3a3a;
                border-radius:0px;
            }
            QWidget#filmWrap {
                background:#242424;
                border-top:1px solid #3a3a3a;
            }
            QWidget#metaBar {
                background:#252525;
                border-top: 1px solid #3b3b3b;
            }
            QLabel#metaLeft {
                color:#e6e6e6;
            }
            QLabel#badge, QLabel#badgeGhost {
                padding:4px 10px;
                border-radius:999px;
                font-size:10pt;
            }
            QLabel#badge {
                background:#1e3a28;
                color:#b7f3c9;
                border:1px solid #2d6a40;
            }
            QLabel#badgeGhost {
                background:#2b2b2b;
                color:#bdbdbd;
                border:1px solid #3a3a3a;
            }
        """)

    def update_settings(self):
        self.update_autosave_interval()
        self._apply_hotkeys()
        self._update_toggle_keys()

    def _update_toggle_keys(self):
        try:
            zebra_key_str = self.settings.hotkeys.get('toggle_zebra', '').split(',')[0].strip()
            self._zebra_toggle_key = QKeySequence(zebra_key_str)[0].key() if zebra_key_str else None
        except Exception:
            self._zebra_toggle_key = None
        try:
            hdr_key_str = self.settings.hotkeys.get('toggle_hdr', '').split(',')[0].strip()
            self._hdr_toggle_key = QKeySequence(hdr_key_str)[0].key() if hdr_key_str else None
        except Exception:
            self._hdr_toggle_key = None

    def update_autosave_interval(self):
        self.autosave_interval_timer.setInterval(self.settings.autosave_interval_min * 60 * 1000)

    def _create_actions(self):
        self.actions: Dict[str, QAction] = {}
        action_map = {
            'save': self.save_all_dirty_files, 
            'quit': self.window().close,
            'next': self.next_photo,
            'prev': self.prev_photo,
            'toggle_select': self.toggle_select,
            'unselect': self.unselect_current,
            'export': self.export_selected,
            'help': self.show_help
        }
        for name, callback in action_map.items():
            action = QAction(self)
            action.triggered.connect(callback)
            self.actions[name] = action
            self.addAction(action)
        self._apply_hotkeys()

    def _apply_hotkeys(self):
        for name, key_sequence_str in self.settings.hotkeys.items():
            if name in self.actions:
                sequences = [QKeySequence(s.strip()) for s in key_sequence_str.split(',') if s.strip()]
                self.actions[name].setShortcuts(sequences)

    def eventFilter(self, obj, ev: QEvent):
        if ev.type() in (QEvent.Type.KeyPress, QEvent.Type.Wheel, QEvent.Type.MouseButtonPress):
            self._note_user_input()
        if ev.type() == QEvent.Type.KeyPress and not ev.isAutoRepeat():
            key = ev.key()
            if self._zebra_toggle_key is not None and key == self._zebra_toggle_key:
                self.toggle_zebra(); return True
            if self._hdr_toggle_key is not None and key == self._hdr_toggle_key:
                self.toggle_hdr(); return True
            
            if Qt.Key_1 <= key <= Qt.Key_5:
                self.set_rating(key - Qt.Key_0)
                return True
            if key == Qt.Key_6:
                self.set_color_label("Red")
                return True
            if key == Qt.Key_7:
                self.set_color_label("Yellow")
                return True
            if key == Qt.Key_8:
                self.set_color_label("Green")
                return True
            if key == Qt.Key_9:
                self.set_color_label("Blue")
                return True
            if key == Qt.Key_0:
                self.set_color_label("Purple")
                return True

        return super().eventFilter(obj, ev)

    def _note_user_input(self):
        now = time.monotonic()
        self._last_input_ts = now

    def _flush_queue(self):
        flushed = 0
        try:
            while True:
                self._taskq.get_nowait()
                self._taskq.task_done(); flushed += 1
        except Empty:
            pass
        with self._pending_lock:
            p = len(self._pending_tasks); self._pending_tasks.clear()
        _plog(f"flush queue: removed={flushed}, cleared_pending={p}")
        
    def _update_selected_badge_fast(self):
        total_sel = sum(1 for x in self.catalog.photos if x.selected)
        self.badge_selected.setText(f"Selected: {total_sel}")
        self.badge_selected.repaint()

    def _loader_loop(self):
        while not self._loader_stop:
            try:
                prio, _, fn, args = self._taskq.get(timeout=0.2)
            except Empty:
                continue
            try:
                fn(*args)
            except Exception as e:
                print("Loader error:", e)
            finally:
                try: self._taskq.task_done()
                except Exception: pass

    def _post_task(self, priority: int, fn, *args):
        self._task_counter += 1
        self._taskq.put((priority, self._task_counter, fn, args))
        if _prof_enabled():
            try: qsz = self._taskq.qsize()
            except Exception: qsz = -1
            _plog(f"enqueue prio={priority} qsize={qsz}")

    def _touch(self, od: OrderedDict, key, limit: int):
        if key in od: od.move_to_end(key)
        while len(od) > limit:
            od.popitem(last=False)

    def _enforce_cache_limits_locked(self, kind: str, limit: int):
        cache = self.pil_full_cache if kind == 'full' else self.pil_half_cache
        while len(cache) > limit:
            key, pil = cache.popitem(last=False)
            removed_size = self._cache_item_sizes.pop((kind, key), None)
            if removed_size is None:
                removed_size = _estimate_pil_bytes(pil)
            self._cache_estimated_bytes = max(0, self._cache_estimated_bytes - removed_size)
        self._enforce_memory_budget_locked()

    def _enforce_memory_budget_locked(self):
        if psutil is None:
            return
        try:
            vm = psutil.virtual_memory()
        except Exception:
            return
        if not vm:
            return
        total = getattr(vm, 'total', None) or 0
        available = getattr(vm, 'available', None) or 0
        if total <= 0 or available <= 0:
            return

        target = max(64 * 1024 * 1024, int(min(total * 0.15, available * 0.5)))

        while self._cache_estimated_bytes > target and self.pil_full_cache:
            key, pil = self.pil_full_cache.popitem(last=False)
            removed_size = self._cache_item_sizes.pop(('full', key), None)
            if removed_size is None:
                removed_size = _estimate_pil_bytes(pil)
            self._cache_estimated_bytes = max(0, self._cache_estimated_bytes - removed_size)
            self.cache_full_limit = max(8, min(self.cache_full_limit, len(self.pil_full_cache)))

        while self._cache_estimated_bytes > target and self.pil_half_cache:
            key, pil = self.pil_half_cache.popitem(last=False)
            removed_size = self._cache_item_sizes.pop(('half', key), None)
            if removed_size is None:
                removed_size = _estimate_pil_bytes(pil)
            self._cache_estimated_bytes = max(0, self._cache_estimated_bytes - removed_size)
            self.cache_half_limit = max(16, min(self.cache_half_limit, len(self.pil_half_cache)))

        self._cache_estimated_bytes = max(0, self._cache_estimated_bytes)

    def _get_pil_full_cached(self, path: str) -> Optional[Image.Image]:
        with self.cache_lock: return self.pil_full_cache.get(path)
    def _get_pil_half_cached(self, path: str) -> Optional[Image.Image]:
        with self.cache_lock: return self.pil_half_cache.get(path)
    def _put_pil_full(self, path: str, pil: Image.Image):
        if pil is None:
            return
        size = _estimate_pil_bytes(pil)
        with self.cache_lock:
            prev = self.pil_full_cache.pop(path, None)
            if prev is not None:
                prev_size = self._cache_item_sizes.pop(('full', path), None)
                if prev_size is not None:
                    self._cache_estimated_bytes = max(0, self._cache_estimated_bytes - prev_size)
                else:
                    self._cache_estimated_bytes = max(0, self._cache_estimated_bytes - _estimate_pil_bytes(prev))
            self.pil_full_cache[path] = pil
            self._cache_item_sizes[('full', path)] = size
            self._cache_estimated_bytes += size
            self._enforce_cache_limits_locked('full', self.cache_full_limit)
    def _put_pil_half(self, path: str, pil: Image.Image):
        if pil is None:
            return
        size = _estimate_pil_bytes(pil)
        with self.cache_lock:
            prev = self.pil_half_cache.pop(path, None)
            if prev is not None:
                prev_size = self._cache_item_sizes.pop(('half', path), None)
                if prev_size is not None:
                    self._cache_estimated_bytes = max(0, self._cache_estimated_bytes - prev_size)
                else:
                    self._cache_estimated_bytes = max(0, self._cache_estimated_bytes - _estimate_pil_bytes(prev))
            self.pil_half_cache[path] = pil
            self._cache_item_sizes[('half', path)] = size
            self._cache_estimated_bytes += size
            self._enforce_cache_limits_locked('half', self.cache_half_limit)
    def _put_resized_pixmap(self, path: str, size: QSize, pixmap: QPixmap):
        key = (path, size.width(), size.height())
        with self.cache_lock:
            self.resized_pixmap_cache[key] = pixmap; self._touch(self.resized_pixmap_cache, key, self.cache_resized_limit)

    def _enqueue(self, priority: int, key: Tuple, fn, *args):
        with self._pending_lock:
            if key in self._pending_tasks: return
            self._pending_tasks.add(key)
        def _wrap(*a):
            try: fn(*a)
            finally:
                with self._pending_lock:
                    self._pending_tasks.discard(key)
        self._post_task(priority, _wrap, *args)

    def _enqueue_load(self, path: str, kind: str, priority: int):
        key = (path, kind); self._enqueue(priority, key, self._worker_entry, path, kind)
    def _enqueue_meta(self, path: str):
        key = (path, 'meta'); self._enqueue(-89, key, self._worker_entry, path, 'meta')
    def _enqueue_xmp(self, path: str, priority: int):
        key = (path, 'xmp'); self._enqueue(priority, key, self._worker_entry, path, 'xmp')
    def _enqueue_thumb(self, path: str, target_h: int, priority: int):
        key = (path, 'thumb', target_h); self._enqueue(priority, key, self._worker_build_thumb, path, target_h)
    def _enqueue_extract_thumb(self, path: str, target_h: int, priority: int):
        key = (path, 'extract_thumb', target_h); self._enqueue(priority, key, self._worker_extract_thumb, path, target_h)
    def _enqueue_build_resized_pixmap(self, path: str, size: QSize, priority: int):
        key = (path, 'resized_pixmap', size.width(), size.height())
        self._enqueue(priority, key, self._worker_build_resized_pixmap, path, size)

    def _worker_entry(self, path: str, kind: str):
        if kind == 'half':
            with _ptime(f"worker half postprocess {os.path.basename(path)}", warn_ms=40):
                pil = load_half_pil(path)
            if pil is None:
                self.signals.load_failed.emit(path, 'half')
            else:
                self._put_pil_half(path, pil)
            self.signals.loaded.emit(path, 'half')
        elif kind == 'full':
            self._acquire_full_slot()
            try:
                with _ptime(f"worker full postprocess {os.path.basename(path)}", warn_ms=80):
                    pil = load_full_pil(path)
                if pil is None:
                    self.signals.load_failed.emit(path, 'full')
                else:
                    self._put_pil_full(path, pil)
                self.signals.loaded.emit(path, 'full')
            finally:
                self._release_full_slot()
        elif kind == 'meta':
            with _ptime(f"worker meta {os.path.basename(path)}", warn_ms=40):
                m = self._read_metadata_heavy(path)
            pil = self._get_pil_full_cached(path) or self._get_pil_half_cached(path)
            if pil and 'size' not in m: m['size'] = f"{pil.size[0]}x{pil.size[1]}"
            self.signals.meta.emit(path, m)
        elif kind == 'xmp':
            with _ptime(f"worker xmp {os.path.basename(path)}", warn_ms=10):
                xmp_data = read_xmp_sidecar(path)
            if xmp_data:
                self.signals.xmp.emit(path, xmp_data)

    def _acquire_full_slot(self):
        while not self._loader_stop:
            with self._full_lock:
                limit = 1 if self._is_user_hot(300.0) else max(1, min(3, self._num_workers // 2))
                if self._full_running < limit:
                    self._full_running += 1; return
            time.sleep(0.03)

    def _release_full_slot(self):
        with self._full_lock: self._full_running = max(0, self._full_running - 1)

    def _worker_build_resized_pixmap(self, path: str, bounding_box_size: QSize):
        pil = self._get_pil_half_cached(path) or self._get_pil_full_cached(path)
        if pil is None:
            _plog(f"[!!!] build_resized_pixmap FAILED for {os.path.basename(path)}")
            return

        box_w, box_h = bounding_box_size.width(), bounding_box_size.height()
        if box_w <= 0 or box_h <= 0:
            return

        img_w, img_h = pil.size
        if img_w <= 0 or img_h <= 0:
            return

        scale = min(box_w / img_w, box_h / img_h)
        target_w = int(img_w * scale)
        target_h = int(img_h * scale)
        if target_w <= 0 or target_h <= 0:
            return

        with _ptime(f"worker pixmap resize {os.path.basename(path)} -> {target_w}x{target_h}", warn_ms=20):
            try:
                resize_filter = getattr(Image, 'LANCZOS', Image.BILINEAR)
                pil_resized = pil.resize((target_w, target_h), resize_filter)

                qimg = ImageQt.ImageQt(pil_resized.convert("RGBA"))
                qimg = qimg.convertToFormat(QImage.Format_ARGB32_Premultiplied)
            except Exception as e:
                print(f"Error resizing image for {path}: {e}")
                return

        self.signals.resized_pixmap_ready.emit(path, bounding_box_size, qimg)


    def _worker_build_thumb(self, path: str, target_h: int):
        pil = self._get_pil_half_cached(path) or self._get_pil_full_cached(path)
        if pil is None: return
        pw, ph = pil.size
        if ph <= 0: return
        ratio = target_h / float(ph); tw, th = max(1, int(pw * ratio)), target_h
        with _ptime(f"worker thumb resize {os.path.basename(path)} -> {target_h}px", warn_ms=10):
            try: thumb = pil.resize((tw, th), Image.BILINEAR)
            except Exception: thumb = pil
        self.signals.thumb_ready.emit(path, target_h, thumb)

    def _worker_extract_thumb(self, path: str, target_h: int):
        try:
            with rawpy.imread(path) as raw:
                thumb = raw.extract_thumb()
                if thumb.format != rawpy.ThumbFormat.JPEG: return
                pil_thumb = Image.open(io.BytesIO(thumb.data))
                pil_thumb.thumbnail((pil_thumb.width, target_h), Image.BILINEAR)
                self.signals.thumb_ready.emit(path, target_h, pil_thumb)
        except Exception as e:
            if 'No thumbnail' in str(e): pass
            else: print(f"Error extracting thumb for {os.path.basename(path)}: {e}")

    def _read_metadata_heavy(self, path: str) -> Dict[str, str]:
        ext = os.path.splitext(path)[1].lower(); meta: Dict[str, str] = {}
        try:
            if ext in {'.jpg', '.jpeg'}: meta.update(_jpeg_exif_as_meta(path))
            else: meta.update(_raw_meta_as_meta(path))
        except Exception: pass
        return meta

    def _set_meta_label(self, m: Dict[str,str]):
        aperture = m.get('fnumber_text') or (f"f/{m.get('fnumber')}" if m.get('fnumber') else '')
        shutter  = m.get('exp_text')     or (f"{m.get('exp')}s" if m.get('exp') else '')
        iso_val = m.get('iso'); iso_str = f"ISO {iso_val}" if iso_val else ''
        parts = [x for x in [m.get('fl'), aperture, shutter, iso_str, m.get('size')] if x]
        main_line = '  |  '.join(parts)
        cam_parts = [x for x in [m.get('make'), m.get('model')] if x]; cam_line = ' '.join(cam_parts)
        lens_line = m.get('lens', ''); dt_line = m.get('dt', '')
        html = f"""
            <div style="margin:0; padding:0;">
              <div style="margin:0; padding:0; font-weight:600; color:#E6E6E6;">{_h(main_line)}
                <span style="color:#9aa0a6;">&nbsp;&nbsp;|&nbsp;&nbsp;{_h(dt_line)}</span>
              </div>
              <div style="margin:0; padding:0; font-size:9pt; color:#B0B0B0;">{_h(cam_line)}
                <span ...>&nbsp;&nbsp;|&nbsp;&nbsp;{_h(lens_line)}</span>
              </div>
            </div>"""
        self.meta_left.setText(html)

    def _update_metadata(self, path: str):
        m_dyn: Dict[str, str] = {}
        pil_full = self._get_pil_full_cached(path); pil_half = self._get_pil_half_cached(path)
        if pil_full is not None: m_dyn['size'] = f"{pil_full.size[0]}x{pil_full.size[1]}"
        elif pil_half is not None: m_dyn['size'] = f"(Half) {pil_half.size[0]}x{pil_half.size[1]}"
        try:
            st = os.stat(path)
            m_dyn.setdefault('dt', datetime.fromtimestamp(getattr(st, 'st_mtime', st.st_mtime)).strftime('%Y-%m-%d %H:%M:%S'))
        except Exception: pass
        m_heavy = self._meta_cache.get(path)
        if not m_heavy: self._enqueue_meta(path); m_heavy = {}
        m_final = dict(m_heavy); m_final.pop('size', None); m_final.update(m_dyn)
        if m_heavy.get('dt'): m_final['dt'] = m_heavy['dt']
        self._set_meta_label(m_final)

    def show_help(self):
        hotkeys = self.settings.hotkeys
        xmp_hotkeys = (
            "<b>XMP/Rating:</b>\n"
            "  1-5: Set star rating\n"
            "  6: Red, 7: Yellow, 8: Green, 9: Blue\n"
            "  0: Clear color label"
        )
        QMessageBox.information(self, "Key Bindings",
            f"Navigate: {hotkeys.get('next','')} (Next), {hotkeys.get('prev','')} (Previous)\n"
            f"Mouse: Wheel to zoom, Drag to pan (when zoomed)\n"
            f"Select: {hotkeys.get('toggle_select','')} (Toggle), {hotkeys.get('unselect','')} (Unselect)\n"
            f"View Modes: {hotkeys.get('toggle_zebra','')} (Toggle Zebra/Histogram), {hotkeys.get('toggle_hdr','')} (Toggle Faux HDR Preview)\n"
            f"Save/Export: {hotkeys.get('save','')} (Save selections.json), {hotkeys.get('export','')} (Export)\n"
            f"Exit: {hotkeys.get('quit','')}\n\n"
            f"{xmp_hotkeys}"
        )

    def _current(self) -> Optional[Photo]:
        if not self.catalog.photos: return None
        return self.catalog.photos[self.idx]

    def next_photo(self):
        if self.idx + 1 < len(self.catalog.photos):
            self.idx += 1; self._show_current(); self._heavy_load_scheduler.start()

    def prev_photo(self):
        if self.idx > 0:
            self.idx -= 1; self._show_current(); self._heavy_load_scheduler.start()

    def _load_xmp_if_needed(self, photo: Photo):
        """Photo 객체의 XMP 데이터를 아직 읽지 않았다면 파일에서 읽기 작업을 예약합니다."""
        with photo.lock:
            if photo.xmp_loaded:
                return
            photo.xmp_loaded = True
        self._enqueue_xmp(photo.path, priority=-95)

    def toggle_select(self):
        p = self._current()
        if not p: return
        
        self._load_xmp_if_needed(p) 
        p.update_xmp({'selected': not p.selected}) 
        
        self._update_selected_badge_fast()
        self._show_current() 
        self.autosave_timer.start(1500) 

    def unselect_current(self):
        p = self._current()
        if not p: return
        if p.selected:
            self._load_xmp_if_needed(p)
            p.update_xmp({'selected': False})
            
            self._update_selected_badge_fast()
            self._show_current()
            self.autosave_timer.start(1500)

    def set_rating(self, rating: int):
        p = self._current()
        if not p or not (0 <= rating <= 5): return
        
        self._load_xmp_if_needed(p)
        
        new_rating = 0 if p.rating == rating else rating
        p.update_xmp({'rating': new_rating})
        
        self._show_temporary_status(f"Rating set to {p.rating} star(s)", 1000)
        self._update_filmstrip()
        self.autosave_timer.start(1500)

    def set_color_label(self, label: Optional[str]):
        p = self._current()
        if not p: return
        
        self._load_xmp_if_needed(p)
        
        new_label = "" if p.color_label == label else (label or "")
        p.update_xmp({'color_label': new_label})
        
        self._show_temporary_status(f"Color label set to {p.color_label or 'None'}", 1000)
        self._update_filmstrip()
        self.autosave_timer.start(1500)

    def toggle_zebra(self):
        self.zebra_toggled = not self.zebra_toggled
        self.view.set_zebra_enabled(self.zebra_toggled)
        self.badge_zebra.setText("Zebra ON" if self.zebra_toggled else "Zebra OFF")
        self.badge_zebra.setObjectName("badge" if self.zebra_toggled else "badgeGhost")
        self.badge_zebra.style().unpolish(self.badge_zebra); self.badge_zebra.style().polish(self.badge_zebra)
        self._show_temporary_status(f"Zebra/Histogram: {'ON' if self.zebra_toggled else 'OFF'}", 1000)

    def toggle_hdr(self):
        self.hdr_toggled = not self.hdr_toggled
        self.view.set_hdr_enabled(self.hdr_toggled)
        self.badge_hdr.setText("Faux HDR ON" if self.hdr_toggled else "Faux HDR OFF")
        self.badge_hdr.setObjectName("badge" if self.hdr_toggled else "badgeGhost")
        self.badge_hdr.style().unpolish(self.badge_hdr); self.badge_hdr.style().polish(self.badge_hdr)
        self._show_temporary_status(f"Faux HDR Preview: {'ON' if self.hdr_toggled else 'OFF'}", 1000)

    def _show_temporary_status(self, message: str, timeout: int = 1000):
        self.status_restore_timer.stop()
        self.status_message.emit(message, 0)
        self.status_restore_timer.start(timeout)
        
    def _show_toast(self, text: str, ms: int = 1500):
        window = self.window() or self.parent()
        show_toast = getattr(window, "_show_toast", None) if window else None
        if callable(show_toast):
            show_toast(text, ms)
        else:
            self.status_message.emit(text, ms)

    def export_selected(self):
        self.save_all_dirty_files() 
        self.export_requested.emit()

    def _on_filmstrip_click(self, path: str):
        for i, p in enumerate(self.catalog.photos):
            if p.path == path:
                if self.idx == i: return
                self.idx = i; self._show_current(); self._heavy_load_scheduler.start()
                break

    def _filmstrip_loader(self, path: str, target_h: int) -> Optional[QPixmap]:
        key = (path, target_h)
        pm = self._pm_thumb_cache.get(key)
        if pm is not None:
            return pm
        if self._get_pil_half_cached(path) is not None:
            self._enqueue_thumb(path, target_h, priority=-80)
        else:
            self._enqueue_load(path, 'half', priority=-100)
        return None


    def _update_filmstrip(self, k_forward: int = 5, k_backward: int = 5):
        if not self.catalog.photos: self.filmstrip.set_items([]); return
        items = []
        start_back = max(0, self.idx - k_backward)
        for i in range(start_back, self.idx):
            ph = self.catalog.photos[i]
            items.append({'path': ph.path, 'selected': ph.selected, 'current': False, 'rating': ph.rating, 'color_label': ph.color_label})
        
        phc = self.catalog.photos[self.idx]
        items.append({'path': phc.path, 'selected': phc.selected, 'current': True, 'rating': phc.rating, 'color_label': phc.color_label})
        
        end_forward = min(len(self.catalog.photos), self.idx + 1 + k_forward)
        for i in range(self.idx + 1, end_forward):
            ph = self.catalog.photos[i]
            items.append({'path': ph.path, 'selected': ph.selected, 'current': False, 'rating': ph.rating, 'color_label': ph.color_label})
        
        self.filmstrip.set_items(items)

        target_h = max(32, self.filmstrip.height() - 0)
        for it in items:
            self._enqueue_thumb(it['path'], target_h, priority=-96)

        sched = []
        for d in range(1, 5 + 1):
            j_f = self.idx + d; j_b = self.idx - d
            if j_f < len(self.catalog.photos): sched.append((self.catalog.photos[j_f].path, d))
            if j_b >= 0: sched.append((self.catalog.photos[j_b].path, d))
        sched.sort(key=lambda x: x[1])

        FILMSTRIP_BASE = -96
        for pth, d in sched:
            self._enqueue_thumb(pth, target_h, priority=FILMSTRIP_BASE + d)

    @Slot(str, dict)
    def _on_meta_ready(self, path: str, meta: Dict[str, str]):
        self._meta_cache[path] = meta
        cur = self._current()
        if cur and cur.path == path:
            self._update_metadata(path)
    
    @Slot(str, dict)
    def _on_xmp_ready(self, path: str, data: Dict):
        photo = next((p for p in self.catalog.photos if p.path == path), None)
        if not photo:
            return

        with photo.lock:
            photo.xmp_loaded = True
            if photo.is_dirty or photo.is_saving:
                return

            rating_val = data.get('rating')
            label_val = data.get('color_label')
            selected_val = data.get('selected')

            rating_changed = rating_val is not None and photo.rating != rating_val
            label_changed = label_val is not None and photo.color_label != label_val
            selected_changed = selected_val is not None and photo.selected != selected_val

            if rating_changed:
                photo.rating = rating_val
            if label_changed:
                photo.color_label = label_val
            if selected_changed:
                photo.selected = selected_val

            if not (rating_changed or label_changed or selected_changed):
                return

            current_selected = photo.selected

        cur = self._current()
        if cur and cur.path == path:
            self.view.set_selected(current_selected)
            self._update_filmstrip()
        else:
            self.filmstrip.update()

        self._update_selected_badge_fast()
        
    def _refresh_statusbar(self):
        if self.status_restore_timer.isActive(): return
        p = self._current()
        if not p: return
        total = len(self.catalog.photos)
        total_sel = sum(1 for x in self.catalog.photos if x.selected)
        self.badge_selected.setText(f"Selected: {total_sel}")
        msg = f"[{self.idx+1}/{total}]  Selected: {total_sel}  {os.path.basename(p.path)}  |  workers: {self._num_workers}"
        self.status_message.emit(msg, 0)

    def _show_current(self):
        p = self._current()
        if not p:
            self.meta_left.setText("")
            self.view.setText("No images"); return

        self._load_xmp_if_needed(p) 

        total = len(self.catalog.photos)
        visible_range = range(max(0, self.idx - 5), min(total, self.idx + 6))  
        for i in visible_range:
            if i<0: break
            visiblePhoto = self.catalog.photos[i]
            self._load_xmp_if_needed(visiblePhoto)
        
        self.view.set_selected(p.selected)
        self._update_metadata(p.path)
        self._refresh_statusbar()
        self._update_filmstrip(k_forward=5, k_backward=5)

        area = self.view.contentsRect(); dpr = self.view.devicePixelRatioF()
        target_size = QSize(int(area.width() * dpr), int(area.height() * dpr))
        cache_key = (p.path, target_size.width(), target_size.height())

        with self.cache_lock:
            pixmap = self.resized_pixmap_cache.get(cache_key)
            pil_half = self.pil_half_cache.get(p.path)

        if pixmap:
            self.view.set_pils(self._get_pil_full_cached(p.path), pil_half)
            self.view.set_rendered_pixmap(pixmap); self.view.set_loading(False)
        elif pil_half:
            self.view.set_pils(self._get_pil_full_cached(p.path), pil_half)
            self.view.set_rendered_pixmap(None); self.view.set_loading(False)
            self._enqueue_build_resized_pixmap(p.path, target_size, -99)
        else:
            self.view.set_pils(None, None)
            self.view.set_rendered_pixmap(None); self.view.set_loading(True)

        self.view.set_fit()
        self._cancel_full_delay_timer()
        if not pil_half or pixmap:
            self._heavy_load_scheduler.start()

    @Slot(str, str)
    def _on_loaded(self, path: str, kind: str):
        _plog(f"[SLOT] _on_loaded received for path={os.path.basename(path)}, kind={kind}")
        if self._get_pil_full_cached(path) or self._get_pil_half_cached(path):
            self._load_failures.discard(path)
        cur = self._current(); is_current = (cur and cur.path == path)
        if is_current:
            self.view.set_pils(self._get_pil_full_cached(path), self._get_pil_half_cached(path))
            if kind == 'half':
                area = self.view.contentsRect(); dpr = self.view.devicePixelRatioF()
                target_size = QSize(int(area.width() * dpr), int(area.height() * dpr))
                target_h = max(32, (self.filmstrip.height() or self.filmstrip.sizeHint().height() or 104))
                self._enqueue_thumb(path, target_h, priority=-79)
                self._enqueue_build_resized_pixmap(path, target_size, -99)
            if self.view.loading: self.view.set_loading(False)
            else: self.view.update()
            self._update_metadata(path)
        self._refresh_statusbar(); self.filmstrip.update()

    @Slot(str, str)
    def _on_load_failed(self, path: str, kind: str):
        cur = self._current()
        if cur and cur.path == path:
            self.view.set_loading(False)
        if path in self._load_failures:
            return
        self._load_failures.add(path)
        self._show_toast(f"Failed to load image: {os.path.basename(path)}", 2000)

    @Slot(str)
    def _on_xmp_saved(self, path: str):
        self._refresh_statusbar()
        cur = self._current()
        if cur and cur.path == path:
            self._update_metadata(path)
        self.status_message.emit(f"Metadata saved for {os.path.basename(path)}", 2000)

    @Slot(str)
    def _on_xmp_save_failed(self, path: str):
        self.status_message.emit(f"Failed to save metadata for {os.path.basename(path)}", 3000)
        self._show_toast(f"Failed to save metadata: {os.path.basename(path)}", 2000)

    @Slot(str, QSize, object)
    def _on_resized_pixmap_ready(self, path: str, size: QSize, qimg_obj):
        _plog(f"[SLOT] _on_resized_pixmap_ready received for path={os.path.basename(path)}")
        if qimg_obj is None:
            return

        try:
            qimg: QImage = qimg_obj
            dpr = self.view.devicePixelRatioF()
            qimg.setDevicePixelRatio(dpr)
            pm = QPixmap.fromImage(qimg, Qt.NoFormatConversion)
        except Exception as e:
            print(f"Error converting QImage to QPixmap for {path}: {e}")
            return

        self._put_resized_pixmap(path, size, pm)
        cur = self._current()
        if cur and cur.path == path:
            self.view.set_rendered_pixmap(pm)
            self.view.set_loading(False)


    @Slot(str, int, object)
    def _on_thumb_ready(self, path: str, target_h: int, pil_small_obj):
        if pil_small_obj is None: return
        key = (path, target_h)
        pm_new = QPixmap.fromImage(ImageQt.ImageQt(pil_small_obj.convert('RGB')))
        
        if key in self._pm_thumb_cache:
            self._pm_thumb_cache.pop(key, None)
        self._pm_thumb_cache[key] = pm_new
        
        while len(self._pm_thumb_cache) > self._pm_thumb_limit:
            try:
                oldest_key = next(iter(self._pm_thumb_cache))
            except StopIteration:
                break
            self._pm_thumb_cache.pop(oldest_key, None)

        self.filmstrip.update()

    def _is_user_hot(self, ms=300.0) -> bool:
        return (time.monotonic() - self._last_input_ts) * 1000.0 < ms

    def _schedule_heavy_load(self):
        _plog(f"User input settled. Scheduling heavy load for index {self.idx}.")
        self._flush_queue(); self._schedule_loading_plan_fire()

    def _schedule_loading_plan_fire(self):
        cur = self._current()
        if not cur: return
        cur_path = cur.path; total = len(self.catalog.photos)

        area = self.view.contentsRect(); dpr = self.view.devicePixelRatioF()
        target_size = QSize(int(area.width() * dpr), int(area.height() * dpr))

        if self._get_pil_half_cached(cur_path) is None:
            self._enqueue_load(cur_path, 'half', -100)
        
        self._enqueue_xmp(cur_path, -95)

        self._start_full_delay_timer(cur_path)

        neighbors: list[tuple[str,int]] = []
        for d in range(1, 10 + 1):
            j = self.idx + d
            if j < total: neighbors.append((self.catalog.photos[j].path, d))
        for d in range(1, 3 + 1):
            j = self.idx - d
            if j >= 0: neighbors.append((self.catalog.photos[j].path, d))
        neighbors.sort(key=lambda x: x[1])

        HALF_BASE = -60
        for pth, d in neighbors:
            self._enqueue_xmp(pth, HALF_BASE + d * 2)
            if self._get_pil_half_cached(pth) is not None:
                self._enqueue_build_resized_pixmap(pth, target_size, HALF_BASE + d * 2 + 1)
            else:
                self._enqueue_load(pth, 'half', HALF_BASE + d * 2)

        neighbors_full: list[tuple[str,int]] = []
        for d in range(1, 5 + 1):
            j = self.idx + d
            if j < total: neighbors_full.append((self.catalog.photos[j].path, d))
        for d in range(1, 2 + 1):
            j = self.idx - d
            if j >= 0: neighbors_full.append((self.catalog.photos[j].path, d))
        neighbors_full.sort(key=lambda x: x[1])

        FULL_BASE = -40
        for pth, d in neighbors_full:
            if self._get_pil_full_cached(pth) is None:
                self._enqueue_load(pth, 'full', FULL_BASE + d)

    def _start_full_delay_timer(self, path: str):
        if self._get_pil_full_cached(path) is not None: return
        self._full_delay_timer.stop(); self._full_wait_target = path; self._full_delay_timer.start()

    def _cancel_full_delay_timer(self):
        self._full_delay_timer.stop(); self._full_wait_target = None

    def _on_full_delay_fire(self):
        if not self._full_wait_target: return
        cur = self._current()
        if not cur or cur.path != self._full_wait_target: return
        if self._get_pil_full_cached(cur.path) is None:
            self._enqueue_load(cur.path, 'full', -90)

    def _load_all_xmp_data(self):
        """Enqueue loading of XMP data for all photos at startup."""
        for i, photo in enumerate(self.catalog.photos):
            self._enqueue_xmp(photo.path, priority=100 + i)

    def _load_selections(self):
        try:
            with open(self.selections_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            selected_set = set(data.get('selected_paths', []))
            for p in self.catalog.photos:
                p.selected = (p.path in selected_set)
        except Exception:
            pass

    def save_all_dirty_files(self):
        """메모리에서 변경된 모든 사항(selections.json 및 XMP)을 파일에 저장합니다."""
        self.autosave_timer.stop()

        selected_paths = [p.path for p in self.catalog.photos if p.selected]
        data = {'root': self.catalog.root, 'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'selected_paths': selected_paths}
        try:
            with open(self.selections_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving selections.json: {e}")

        tasks: List[Tuple[Photo, Dict, int]] = []
        for photo in self.catalog.photos:
            with photo.lock:
                if not photo.is_dirty or photo.is_saving:
                    continue
                data_to_write = {
                    'rating': photo.rating,
                    'color_label': photo.color_label,
                    'selected': photo.selected
                }
                version = photo.version
                photo.is_dirty = False
                photo.is_saving = True
                photo.saving_version = version
            tasks.append((photo, data_to_write, version))

        if not tasks:
            return

        def _write_task_with_cleanup(path, data, photo_obj, version):
            success = False
            try:
                success = bool(write_xmp_sidecar(path, data))
            except Exception as e:
                print(f"Unexpected error saving XMP for {os.path.basename(path)}: {e}")
                success = False
            finally:
                signal_to_emit = None
                with photo_obj.lock:
                    if photo_obj.saving_version != version:
                        return
                    photo_obj.is_saving = False
                    if photo_obj.version > version:
                        photo_obj.is_dirty = True
                        photo_obj.saving_version = version
                        return
                    if not success:
                        photo_obj.is_dirty = True
                        photo_obj.saving_version = version
                        signal_to_emit = 'failed'
                    else:
                        photo_obj.saving_version = version
                        signal_to_emit = 'saved'
                if signal_to_emit == 'saved':
                    self.signals.xmp_saved.emit(path)
                elif signal_to_emit == 'failed':
                    self.signals.xmp_save_failed.emit(path)

        for photo, payload, version in tasks:
            self._post_task(20, _write_task_with_cleanup, photo.path, payload, photo, version)

        self._show_temporary_status(f"Auto-saved metadata for {len(tasks)} photos.", 2000)
    def cleanup(self):
        self.save_all_dirty_files() 
        self._loader_stop = True
        try:
            while True:
                self._taskq.get_nowait(); self._taskq.task_done()
        except Empty:
            pass
        try:
            for _ in range(len(self._loader_threads)):
                self._taskq.put((9999, 0, lambda: None, ()))
            for t in self._loader_threads:
                t.join(timeout=0.5)
        except Exception:
            pass
        QApplication.instance().removeEventFilter(self)

              
class KeySequenceEdit(QLineEdit):
    def __init__(self, key_sequence_str: str, parent=None):
        super().__init__(key_sequence_str, parent)
        self.setPlaceholderText("Click to set a new shortcut")
        self._is_capturing = False

    def _enter_capture_mode(self):
        self.setText("")
        self.setPlaceholderText("Press a key or key combination...")
        self._is_capturing = True

    def _exit_capture_mode(self):
        self._is_capturing = False
        self.setPlaceholderText("Click to set a new shortcut")

    def mousePressEvent(self, event: QEvent):
        self._enter_capture_mode()
        super().mousePressEvent(event)

    def focusOutEvent(self, event: QEvent):
        self._exit_capture_mode()
        super().focusOutEvent(event)

    def keyPressEvent(self, event: QEvent):
        if not self._is_capturing:
            super().keyPressEvent(event)
            return

        key = event.key()
        
        if key in (Qt.Key_unknown, Qt.Key_Control, Qt.Key_Shift, Qt.Key_Alt, Qt.Key_Meta):
            return

        key_sequence = QKeySequence(event.keyCombination())
        text = key_sequence.toString(QKeySequence.NativeText)

        if not text:
            return

        current_text = self.text()
        if current_text:
            self.setText(f"{current_text}, {text}")
        else:
            self.setText(text)

        event.accept()

class SettingsDialog(QDialog):
    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.settings = settings
        self.hotkey_edits: Dict[str, QLineEdit] = {}

        main_layout = QVBoxLayout(self)
        
        general_group = QGroupBox("General")
        form_layout = QFormLayout()
        self.autosave_spinbox = QSpinBox()
        self.autosave_spinbox.setRange(1, 60)
        self.autosave_spinbox.setValue(self.settings.autosave_interval_min)
        self.autosave_spinbox.setSuffix(" minute(s)")
        form_layout.addRow("Autosave Interval:", self.autosave_spinbox)

        self.output_folder_edit = QLineEdit(self.settings.output_folder_name)
        form_layout.addRow("Output Folder Name:", self.output_folder_edit)
        
        self.raw_output_folder_edit = QLineEdit(self.settings.raw_output_folder_name)
        form_layout.addRow("Raw Output Folder Name:", self.raw_output_folder_edit)

        self.jpeg_output_folder_edit = QLineEdit(self.settings.jpeg_output_folder_name)
        form_layout.addRow("JPEG Output Folder Name:", self.jpeg_output_folder_edit)
        
        general_group.setLayout(form_layout)
        main_layout.addWidget(general_group)

        hotkey_group = QGroupBox("Hotkeys")
        hotkey_layout = QFormLayout()
        
        hotkey_labels = {
            'next': 'Next Image:',
            'prev': 'Previous Image:',
            'toggle_select': 'Toggle Select:',
            'unselect': 'Unselect Image:',
            'toggle_zebra': 'Toggle Zebra/Histogram:',
            'toggle_hdr': 'Toggle Faux HDR Preview:',
            'save': 'Save Selections:',
            'export': 'Export Selected:',
            'help': 'Show Help:',
            'quit': 'Quit Application:'
        }

        for action, key_sequence_str in self.settings.hotkeys.items():
            label_text = hotkey_labels.get(action, f"{action.replace('_', ' ').title()}:")
            edit = KeySequenceEdit(key_sequence_str)
            hotkey_layout.addRow(label_text, edit)
            self.hotkey_edits[action] = edit
        hotkey_group.setLayout(hotkey_layout)
        main_layout.addWidget(hotkey_group)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        main_layout.addWidget(self.button_box)

    def accept(self):
        self.settings.autosave_interval_min = self.autosave_spinbox.value()
        self.settings.output_folder_name = self.output_folder_edit.text() or "selected"
        self.settings.raw_output_folder_name = self.raw_output_folder_edit.text() or "_selected_raw"
        self.settings.jpeg_output_folder_name = self.jpeg_output_folder_edit.text() or "_selected_jpeg"
        
        for action, edit in self.hotkey_edits.items():
            self.settings.hotkeys[action] = edit.text()

        super().accept()

class CompletionDialog(QDialog):
    def __init__(self, title: str,
                selected_count: int,
                raw_path: str, dest_raw_count: int,
                jpeg_path: str, dest_jpeg_count: int,
                parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(560)

        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        layout = QVBoxLayout(self)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(16)

        title_label = QLabel(title)
        f = title_label.font(); f.setPointSize(18); f.setBold(True); title_label.setFont(f)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        group = QGroupBox("Summary")
        form = QFormLayout(); form.setSpacing(10); form.setLabelAlignment(Qt.AlignRight)

        sel_lbl = QLabel(f"<b>{selected_count}</b> selected")
        form.addRow("Selected:", sel_lbl)

        raw_row = QLabel(f"<b>{dest_raw_count}</b> files  |  <i>{_h(raw_path)}</i>")
        raw_row.setWordWrap(True)
        form.addRow("RAW:", raw_row)

        if dest_jpeg_count > 0:
            jpg_row = QLabel(f"<b>{dest_jpeg_count}</b> files  |  <i>{_h(jpeg_path)}</i>")
            jpg_row.setWordWrap(True)
            form.addRow("JPEG:", jpg_row)

        group.setLayout(form)
        layout.addWidget(group)

        btn_row = QHBoxLayout()
        open_raw_btn = QPushButton("Open RAW Folder")
        open_raw_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(raw_path)))
        btn_row.addWidget(open_raw_btn)

        if dest_jpeg_count > 0:
            open_jpg_btn = QPushButton("Open JPEG Folder")
            open_jpg_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(jpeg_path)))
            btn_row.addWidget(open_jpg_btn)

        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        layout.addStretch(1)

        btns = QDialogButtonBox(QDialogButtonBox.Ok)
        btns.accepted.connect(self.accept)
        btns.setCenterButtons(True)
        layout.addWidget(btns)

class WelcomeWidget(QWidget):
    folder_selected = Signal(str)

    _PRIMARY_BG      = "#1b6d35"
    _PRIMARY_BG_HOV  = "#2a8b4a"
    _PRIMARY_BG_PR   = "#15562a"
    _PRIMARY_BORDER  = "#57b87a"
    _PRIMARY_SHADOW  = QColor(43, 150, 85, 110)

    _RECENT_TEXT     = "#b8b8b8"
    _RECENT_BORDER   = "#3a3a3a"
    _RECENT_BG       = "transparent"
    _RECENT_BG_HOV   = "#343434"
    _RECENT_BG_PR    = "#2f2f2f"
    _RECENT_DOT      = "#7e7e7e"

    GAP_TITLE_BTN    = 36
    GAP_CENTER_BOTTOM= 36
    CONTAINER_W      = 560
    FOOTER_MARGIN_TOP= 16

    def __init__(self, on_select_folder, recent_folders: List[str], parent=None):
        super().__init__(parent)
        self.on_select_folder = on_select_folder

        root_h = QHBoxLayout(self)
        root_h.setContentsMargins(0, 0, 0, 0); root_h.setSpacing(0); root_h.setAlignment(Qt.AlignCenter)

        col = QWidget(self); col.setFixedWidth(self.CONTAINER_W)
        root_h.addWidget(col, 0, Qt.AlignCenter)

        v = QVBoxLayout(col); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)

        self._sp_top    = QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Fixed)
        self._sp_bottom = QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        v.addItem(self._sp_top)

        self.center_group = QWidget(col)
        cg = QVBoxLayout(self.center_group); cg.setContentsMargins(0, 0, 0, 0); cg.setSpacing(0); cg.setAlignment(Qt.AlignHCenter)
        app = QApplication.instance()
        font_family = getattr(app, "_custom_font_family", "Arial")

        self.title = QLabel(self.center_group)
        self.title.setText(f'simple <span style="color:{self._PRIMARY_BORDER}">raw</span> picker')
        self.title.setTextFormat(Qt.RichText); self.title.setAlignment(Qt.AlignCenter)
        tf = QFont(font_family, 40); tf.setBold(True); self.title.setFont(tf)
        cg.addWidget(self.title, 0, Qt.AlignHCenter)

        cg.addSpacing(self.GAP_TITLE_BTN)

        self.select_btn = QPushButton("Select Folder to Start", self.center_group)
        self.select_btn.setObjectName("WelcomeButton")
        self.select_btn.setMinimumHeight(56)
        self.select_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.select_btn.setCursor(Qt.PointingHandCursor)
        self.select_btn.clicked.connect(on_select_folder)
        self.select_btn.setStyleSheet(f"""
            QPushButton#WelcomeButton {{
                background-color: {self._PRIMARY_BG};
                border: 1px solid {self._PRIMARY_BORDER};
                color: #ffffff;
                font-size: 12pt;
                font-weight: bold;
                padding: 12px 18px;
                border-radius: 10px;
            }}
            QPushButton#WelcomeButton:hover {{ background-color: {self._PRIMARY_BG_HOV}; }}
            QPushButton#WelcomeButton:pressed {{ background-color: {self._PRIMARY_BG_PR}; }}
        """)
        shadow = QGraphicsDropShadowEffect(self); shadow.setBlurRadius(24); shadow.setOffset(0, 4); shadow.setColor(self._PRIMARY_SHADOW)
        self.select_btn.setGraphicsEffect(shadow)
        cg.addWidget(self.select_btn, 0, Qt.AlignHCenter)

        v.addWidget(self.center_group, 0, Qt.AlignHCenter)

        self._gap_between = QWidget(col); self._gap_between.setFixedHeight(self.GAP_CENTER_BOTTOM)
        v.addWidget(self._gap_between)

        self.bottom_section = QWidget(col)
        bl = QVBoxLayout(self.bottom_section); bl.setContentsMargins(0, 0, 0, 20); bl.setSpacing(0)

        divider = QFrame(self.bottom_section)
        divider.setFrameShape(QFrame.HLine); divider.setFrameShadow(QFrame.Sunken)
        divider.setStyleSheet("background-color:#444; color:#444;"); divider.setFixedHeight(2)
        bl.addWidget(divider)

        bl.addSpacing(36)

        self.recent_header = QLabel("Recent Folders", self.bottom_section)
        self.recent_header.setAlignment(Qt.AlignLeft)
        hf = QFont(font_family, 10); hf.setLetterSpacing(QFont.PercentageSpacing, 102)
        self.recent_header.setFont(hf); self.recent_header.setStyleSheet("color:#9a9a9a;")
        bl.addWidget(self.recent_header)

        self.recent_container = QWidget(self.bottom_section)
        self.recent_layout = QVBoxLayout(self.recent_container); self.recent_layout.setContentsMargins(0, 0, 0, 0); self.recent_layout.setSpacing(6)
        bl.addWidget(self.recent_container, 0, Qt.AlignTop)

        support = QLabel('<a href="http://donate.recu3125.com">Support the developer</a>', self.bottom_section)
        support.setOpenExternalLinks(True); support.setAlignment(Qt.AlignCenter)
        support.setStyleSheet(f"QLabel {{ color:#8fbf92; font-size:10pt; margin-top:{self.FOOTER_MARGIN_TOP}px; }}")

        bl.addStretch(1)
        bl.addWidget(support, 0, Qt.AlignBottom)

        v.addWidget(self.bottom_section, 0)
        v.addItem(self._sp_bottom)

        self.update_recent_folders(recent_folders)
        QTimer.singleShot(0, self._reflow)

    def _reflow(self):
        H = self.height()
        gh = self.center_group.sizeHint().height()
        gap = self._gap_between.height()
        bh = self.bottom_section.sizeHint().height()

        total = gh + gap + bh
        if H >= total:
            top_h = (H - total) // 2
        else:
            top_h = max(0, H - total)

        self._sp_top.changeSize(0, top_h, QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.layout().invalidate(); self.layout().activate()

    def resizeEvent(self, e):
        super().resizeEvent(e); self._reflow()

    def _make_recent_row(self, display_text: str, full_path: str) -> QWidget:
        row = QWidget(self.bottom_section)
        hl = QHBoxLayout(row); hl.setContentsMargins(0, 0, 0, 0); hl.setSpacing(8)

        dot = QLabel(row); dot.setFixedSize(8, 8)
        dot.setStyleSheet(f"background:{self._RECENT_DOT}; border-radius:4px;")
        hl.addWidget(dot, 0, Qt.AlignVCenter)

        btn = QPushButton(display_text, row)
        btn.setObjectName("RecentButton"); btn.setFlat(True)
        btn.setMinimumHeight(30); btn.setCursor(Qt.PointingHandCursor)
        btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn.setToolTip(full_path); btn.setFocusPolicy(Qt.NoFocus)
        btn.clicked.connect(lambda _=False, p=full_path: self.folder_selected.emit(p))
        btn.setStyleSheet(f"""
            QPushButton#RecentButton {{
                background: {self._RECENT_BG};
                color: {self._RECENT_TEXT};
                border: 1px solid {self._RECENT_BORDER};
                padding: 6px 10px;
                border-radius: 8px;
                text-align: left;
            }}
            QPushButton#RecentButton:hover {{ background: {self._RECENT_BG_HOV}; border-color: {self._RECENT_BORDER}; }}
            QPushButton#RecentButton:pressed {{ background: {self._RECENT_BG_PR}; }}
        """)
        hl.addWidget(btn, 1)
        return row

    def update_recent_folders(self, folders: List[str]):
        while self.recent_layout.count():
            it = self.recent_layout.takeAt(0)
            w = it.widget()
            if w: w.deleteLater()

        valid = [f for f in folders if os.path.isdir(f)][:3]
        if valid:
            for p in valid:
                name = os.path.basename(os.path.normpath(p)) or p
                self.recent_layout.addWidget(self._make_recent_row(name, p))
            self.recent_header.setVisible(True); self.recent_container.setVisible(True)
        else:
            self.recent_header.setVisible(False); self.recent_container.setVisible(False)

        QTimer.singleShot(0, self._reflow)


class AppWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("simple raw picker")
        self.resize(1200, 800)
        self.args = args
        self.settings = AppSettings()
        self.culling_widget: Optional[CullingWidget] = None
        self._export_worker: Optional[ExportWorker] = None
        self._export_thread: Optional[QThread] = None
        self._export_dialog: Optional[QProgressDialog] = None
        self._export_intent: Optional[str] = None

        self.app_data_path = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        os.makedirs(self.app_data_path, exist_ok=True)
        self.app_state_file = os.path.join(self.app_data_path, "app_state.json")
        self.recent_folders = []
        self._load_app_state()

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.welcome_screen = WelcomeWidget(
            on_select_folder=self.select_folder,
            recent_folders=self.recent_folders
        )
        self.welcome_screen.folder_selected.connect(self.start_culling_session)
        self.stack.addWidget(self.welcome_screen)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self._create_toolbar()
        self.update_toolbar_state(is_culling=False)

    def _load_app_state(self):
        try:
            with open(self.app_state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
                self.recent_folders = state.get("recent_folders", [])
        except (FileNotFoundError, json.JSONDecodeError):
            self.recent_folders = []

    def _save_app_state(self):
        try:
            state = {"recent_folders": self.recent_folders}
            with open(self.app_state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"Error saving app state: {e}")

    def _create_toolbar(self):
        self.toolbar = QToolBar("Main Toolbar")
        self.addToolBar(self.toolbar)

        self.act_select_folder = QAction("Select Folder", self)
        self.act_select_folder.triggered.connect(self.select_folder)
        self.toolbar.addAction(self.act_select_folder)

        self.act_settings = QAction("Settings", self)
        self.act_settings.triggered.connect(self.open_settings)
        self.toolbar.addAction(self.act_settings)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.toolbar.addWidget(spacer)

        self.act_help = QAction("Help", self)
        self.act_help.triggered.connect(self.open_help)
        self.toolbar.addAction(self.act_help)

        self.act_complete = QAction("Complete", self)
        self.act_complete.triggered.connect(self.complete_culling)
        self.toolbar.addAction(self.act_complete)

    def update_toolbar_state(self, is_culling: bool):
        self.act_settings.setEnabled(True)
        self.act_complete.setEnabled(is_culling)

    def select_folder(self):
        root = QFileDialog.getExistingDirectory(self, "Select Photo Folder")
        if root:
            self.start_culling_session(root)

    def select_folder_from_arg(self, path):
        if os.path.isdir(path):
            self.start_culling_session(path)
        else:
            QMessageBox.warning(self, "Invalid Folder", f"The provided path is not a valid directory:\n{path}")

    def start_culling_session(self, root):
        if self.culling_widget:
            self.culling_widget.cleanup()
            self.stack.removeWidget(self.culling_widget)
            self.culling_widget.deleteLater()

        if root in self.recent_folders:
            self.recent_folders.remove(root)
        self.recent_folders.insert(0, root)
        self.recent_folders = self.recent_folders[:5]
        self._save_app_state()

        self.culling_widget = CullingWidget(
            root=root,
            settings=self.settings,
            workers=self.args.workers,
            parent=self
        )
        self.culling_widget.status_message.connect(self.status.showMessage)
        self.culling_widget.export_requested.connect(self.handle_export)
        self.stack.addWidget(self.culling_widget)
        self.stack.setCurrentWidget(self.culling_widget)
        self.update_toolbar_state(is_culling=True)

    def open_settings(self):
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec():
            if self.culling_widget:
                self.culling_widget.update_settings()
            self.status.showMessage("Settings updated.", 2000)
            
    def open_help(self):
        if self.culling_widget:
            self.culling_widget.show_help()
        else:
            QMessageBox.information(self, "Help",
                "Open a folder to start.\n\n"
                "Shortcuts will be available during culling.")

    @staticmethod
    def _perform_export(job: ExportJob,
                        progress_callback=None,
                        is_cancelled=None) -> Tuple[int, str, int, str, int, int, int]:
        is_cancelled = is_cancelled or (lambda: False)

        def _check_cancel():
            if is_cancelled():
                raise ExportCancelledError()

        def _emit(current: int, total: int, message: str):
            if progress_callback:
                progress_callback(current, total, message)

        def _needs_copy(src: str, dst: str) -> bool:
            try:
                s, d = os.stat(src), os.stat(dst)
                if s.st_size != d.st_size:
                    return True
                s_m = getattr(s, "st_mtime_ns", int(s.st_mtime * 1e9))
                d_m = getattr(d, "st_mtime_ns", int(d.st_mtime * 1e9))
                return s_m > d_m
            except FileNotFoundError:
                return True
            except Exception:
                return True

        def _list_by_basename(folder: str, exts: set) -> Dict[str, str]:
            out = {}
            try:
                for fn in os.listdir(folder):
                    p = os.path.join(folder, fn)
                    if not os.path.isfile(p):
                        continue
                    base, ext = os.path.splitext(fn)
                    if ext.lower() in exts:
                        out[base] = p
            except Exception:
                pass
            return out

        selected_raw_paths = list(job.selected_raw_paths)
        selected_count = len(selected_raw_paths)
        root = job.root

        raw_out_dir = os.path.join(root, job.raw_output_folder_name)
        jpeg_out_dir = os.path.join(root, job.jpeg_output_folder_name)

        _check_cancel()
        os.makedirs(raw_out_dir, exist_ok=True)
        os.makedirs(jpeg_out_dir, exist_ok=True)

        selected_raw_by_base = {os.path.splitext(os.path.basename(p))[0]: p for p in selected_raw_paths}
        root_jpegs_by_base = _list_by_basename(root, {'.jpg', '.jpeg'})

        _check_cancel()
        dest_raw_by_base = _list_by_basename(raw_out_dir, SUPPORTED_EXTS.union({'.xmp'}))
        dest_jpg_by_base = _list_by_basename(jpeg_out_dir, {'.jpg', '.jpeg'})

        for base, dst_path in list(dest_raw_by_base.items()):
            _check_cancel()
            if base not in selected_raw_by_base:
                try:
                    os.remove(dst_path)
                except Exception:
                    pass

        desired_jpg_bases = {b for b in selected_raw_by_base if b in root_jpegs_by_base}

        for base, dst_path in list(dest_jpg_by_base.items()):
            _check_cancel()
            if base not in desired_jpg_bases:
                try:
                    os.remove(dst_path)
                except Exception:
                    pass

        copy_tasks: List[Tuple[str, str, str]] = []
        for base, src_path in selected_raw_by_base.items():
            _check_cancel()
            dst_path = os.path.join(raw_out_dir, os.path.basename(src_path))
            if _needs_copy(src_path, dst_path):
                copy_tasks.append(("raw", src_path, dst_path))
            src_xmp = os.path.splitext(src_path)[0] + '.xmp'
            dst_xmp = os.path.splitext(dst_path)[0] + '.xmp'
            if os.path.exists(src_xmp) and _needs_copy(src_xmp, dst_xmp):
                copy_tasks.append(("xmp", src_xmp, dst_xmp))

        for base in desired_jpg_bases:
            _check_cancel()
            src_jpg = root_jpegs_by_base[base]
            dst_jpg = os.path.join(jpeg_out_dir, os.path.basename(src_jpg))
            if _needs_copy(src_jpg, dst_jpg):
                copy_tasks.append(("jpeg", src_jpg, dst_jpg))

        errors: List[str] = []
        copied_raw = 0
        copied_jpg = 0
        total_tasks = len(copy_tasks)
        progress_total = max(total_tasks, 1)
        if total_tasks == 0:
            _emit(0, progress_total, "No files required copying")
        else:
            _emit(0, progress_total, "Preparing export...")

        for idx, (kind, src, dst) in enumerate(copy_tasks, start=1):
            _check_cancel()
            filename = os.path.basename(src)
            _emit(idx - 1, progress_total, f"Copying {filename}")
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                if kind == 'raw':
                    errors.append(f"[RAW sync] copy failed: {src} -> {dst}: {e}")
                elif kind == 'jpeg':
                    errors.append(f"[JPEG sync] copy failed: {src} -> {dst}: {e}")
                else:
                    errors.append(f"[XMP sync] copy failed: {src} -> {dst}: {e}")
            else:
                if kind == 'raw':
                    copied_raw += 1
                elif kind == 'jpeg':
                    copied_jpg += 1
            finally:
                _emit(idx, progress_total, f"Copying {filename}")

        _emit(progress_total, progress_total, "Finalizing export...")
        _check_cancel()

        dest_raw_count = len(_list_by_basename(raw_out_dir, SUPPORTED_EXTS))
        _check_cancel()
        dest_jpeg_count = len(_list_by_basename(jpeg_out_dir, {'.jpg', '.jpeg'}))

        if errors:
            raise RuntimeError("Export encountered errors:\n" + "\n".join(errors))

        return (copied_raw, raw_out_dir, copied_jpg, jpeg_out_dir,
                selected_count, dest_raw_count, dest_jpeg_count)

    def _set_export_controls_enabled(self, enabled: bool):
        if self.culling_widget:
            export_action = self.culling_widget.actions.get('export')
            if export_action:
                export_action.setEnabled(enabled)
        if enabled:
            self.update_toolbar_state(is_culling=bool(self.culling_widget))
        else:
            self.act_complete.setEnabled(False)

    def _start_export(self, mode: str):
        if not self.culling_widget:
            return
        if self._export_worker is not None:
            self._show_toast("Export already in progress", 2000)
            return

        cw = self.culling_widget
        cw.save_all_dirty_files()

        job = ExportJob(
            root=cw.catalog.root,
            selected_raw_paths=[p.path for p in cw.catalog.photos if p.selected],
            raw_output_folder_name=self.settings.raw_output_folder_name,
            jpeg_output_folder_name=self.settings.jpeg_output_folder_name
        )

        self._export_intent = mode
        self._set_export_controls_enabled(False)

        dialog = QProgressDialog("Preparing export...", "Cancel", 0, 1, self)
        dialog.setWindowTitle("Exporting")
        dialog.setWindowModality(Qt.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setValue(0)
        dialog.setLabelText("Preparing export...")
        dialog.canceled.connect(self._cancel_export)
        self._export_dialog = dialog

        worker = ExportWorker(job, AppWindow._perform_export)
        thread = QThread(self)
        self._export_worker = worker
        self._export_thread = thread
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_export_progress)
        worker.finished.connect(self._on_export_finished)
        worker.failed.connect(self._on_export_failed)
        worker.cancelled.connect(self._on_export_cancelled)
        thread.start()
        dialog.show()

    def _cancel_export(self):
        if self._export_worker:
            self._export_worker.cancel()
        if self._export_dialog:
            self._export_dialog.setLabelText("Cancelling...")
            self._export_dialog.setCancelButton(None)

    def _on_export_progress(self, current: int, total: int, message: str):
        if not self._export_dialog:
            return
        total = max(total, 1)
        self._export_dialog.setRange(0, total)
        self._export_dialog.setValue(min(current, total))
        if message:
            self._export_dialog.setLabelText(message)

    def _on_export_finished(self, result):
        mode = self._export_intent
        self._export_intent = None
        self._teardown_export_worker()
        self._finalize_export_ui()
        self._handle_export_success(result, mode)

    def _on_export_failed(self, message: str, details: str):
        self._export_intent = None
        self._teardown_export_worker()
        self._finalize_export_ui()
        if details:
            print(details, file=sys.stderr)
        box = QMessageBox(self)
        box.setWindowTitle("Export Failed")
        box.setIcon(QMessageBox.Critical)
        text = message or "Export failed."
        box.setText(text)
        if details:
            box.setDetailedText(details)
        box.exec()
        self._show_toast("Export failed", 2000)

    def _on_export_cancelled(self):
        self._export_intent = None
        self._teardown_export_worker()
        self._finalize_export_ui()
        self._show_toast("Export cancelled", 2000)

    def _teardown_export_worker(self):
        worker, thread = self._export_worker, self._export_thread
        self._export_worker = None
        self._export_thread = None
        if worker is not None:
            worker.deleteLater()
        if thread is not None:
            thread.quit()
            thread.wait()
            thread.deleteLater()

    def _finalize_export_ui(self):
        if self._export_dialog:
            self._export_dialog.hide()
            self._export_dialog.deleteLater()
            self._export_dialog = None
        self._set_export_controls_enabled(True)

    def _handle_export_success(self, result: Tuple[int, str, int, str, int, int, int], mode: Optional[str]):
        (copied_raw, raw_path, copied_jpg, jpeg_path,
         selected_count, dest_raw_count, dest_jpeg_count) = result

        if mode == "complete":
            dlg = CompletionDialog("Culling Complete",
                                  selected_count,
                                  raw_path, dest_raw_count,
                                  jpeg_path, dest_jpeg_count,
                                  self)
            dlg.exec()

            if self.culling_widget:
                self.culling_widget.cleanup()
                self.stack.removeWidget(self.culling_widget)
                self.culling_widget.deleteLater()
                self.culling_widget = None

            self._load_app_state()
            self.welcome_screen.update_recent_folders(self.recent_folders)
            self.stack.setCurrentWidget(self.welcome_screen)
            self.update_toolbar_state(is_culling=False)
            self.status.clearMessage()
        else:
            msg = (f"Sync complete · {selected_count} selected → "
                   f"{dest_raw_count} RAW, {dest_jpeg_count} JPEG")
            self._show_toast(msg, 2000)

    def handle_export(self):
        self._start_export(mode="sync")

    def complete_culling(self):
        self._start_export(mode="complete")

    def _show_toast(self, text: str, ms: int = 1500):
        if not hasattr(self, "_toast_label"):
            self._toast_label = QLabel(self)
            self._toast_label.setObjectName("toastLabel")
            self._toast_label.setStyleSheet("""
                QLabel#toastLabel {
                    background: rgba(0,0,0,180);
                    color: #ffffff;
                    border-radius: 8px;
                    padding: 10px 14px;
                    font-weight: 600;
                }
            """)
            self._toast_label.setAlignment(Qt.AlignCenter)
            self._toast_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            self._toast_timer = QTimer(self)
            self._toast_timer.setSingleShot(True)
            self._toast_timer.timeout.connect(self._toast_label.hide)

        self._toast_label.setText(text)
        self._toast_label.adjustSize()

        w, h = self._toast_label.width(), self._toast_label.height()
        x = max(0, (self.width() - w) // 2)
        y = max(0, (self.height() - h) //2)
        self._toast_label.setGeometry(x, y, w, h)
        self._toast_label.raise_()
        self._toast_label.show()
        self._toast_timer.start(ms)
        
    def closeEvent(self, event):
        if self.culling_widget:
            self.culling_widget.cleanup()
        event.accept()

def main():
    parser = argparse.ArgumentParser(description='simple raw picker - A fast photo culling tool.')
    parser.add_argument('root', nargs='?', help='Path to the photo folder (optional)')
    parser.add_argument('--workers', type=int, default=None, help='Number of loader threads (default: CPU-1, max 8, min 2)')
    parser.add_argument('--profile', action='store_true', help='Enable performance profiling logs')
    args = parser.parse_args()

    app = QApplication.instance() or QApplication(sys.argv)
    app.setOrganizationName("simple-raw-picker")
    app.setApplicationName("simple-raw-picker")
    
    setattr(app, "_profile_enabled", bool(args.profile))

    font_path = "Gantari-Regular.ttf"
    if hasattr(sys, '_MEIPASS'):
        font_path = os.path.join(sys._MEIPASS, font_path)

    font_id = QFontDatabase.addApplicationFont(font_path)
    if font_id == -1:
        print(f"Warning: Could not load font '{font_path}'. Falling back to default.")
        setattr(app, "_custom_font_family", "Arial")
    else:
        font_families = QFontDatabase.applicationFontFamilies(font_id)
        if font_families:
            font_family = font_families[0]
            setattr(app, "_custom_font_family", font_family)
            default_font = QFont(font_family, 10) 
            app.setFont(default_font)
            print(f"Successfully loaded font '{font_family}'")
        else:
            print(f"Warning: Could not find font family in '{font_path}'.")
            setattr(app, "_custom_font_family", "Arial")

    app.setStyleSheet("""
        QWidget {
            background-color: #2b2b2b;
            color: #f0f0f0;
        }
        QMainWindow, QDialog {
            background-color: #2b2b2b;
        }
        QToolBar {
            background: #3c3c3c;
            border: none;
            padding: 2px;
        }
        QToolBar QToolButton {
            color: #f0f0f0;
            background: transparent;
            padding: 6px;
            margin: 2px;
        }
        QToolBar QToolButton:hover {
            background: #555;
            border-radius: 3px;
        }
        QStatusBar {
            background: #3c3c3c;
            color: #d0d0d0;
        }
        QGroupBox {
            border: 1px solid #4a4a4a;
            border-radius: 5px;
            margin-top: 1ex;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 3px;
        }
        QPushButton {
            background-color: #555;
            color: #f0f0f0;
            border: 1px solid #666;
            padding: 8px 16px;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #6a6a6a;
            border-color: #777;
        }
        QPushButton:pressed {
            background-color: #4a4a4a;
        }
        QPushButton:focus {
            outline: none;
        }
        QPushButton#WelcomeButton {
            background-color: #2C6F40;
            border-color: #66BB6A;
            font-size: 11pt;
            font-weight: bold;
        }
        QPushButton#WelcomeButton:hover {
            background-color: #66BB6A;
        }
        QPushButton#WelcomeButton:pressed {
            background-color: #286E1C;
        }
        QLineEdit, QSpinBox {
            background-color: #3c3c3c;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 5px;
        }
        QLabel a {
            color: #4CAF50;
            text-decoration: none;
        }
        QMessageBox {
            background-color: #3c3c3c;
        }
    """)


    win = AppWindow(args)
    win.show()

    if args.root:
        win.select_folder_from_arg(args.root)
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()