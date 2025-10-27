#!/usr/bin/env python3

import os, sys, io, json, time, shutil, argparse, traceback, subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Optional, Tuple, Set
from datetime import datetime
import threading
from collections import OrderedDict
from queue import PriorityQueue, Empty
from contextlib import contextmanager
from html import escape as _h

import numpy as np
from PIL import Image, ImageQt, ImageOps
import rawpy
try:
    import cv2
except Exception:
    cv2 = None

try:
    import mediapipe as mp
except Exception:
    mp = None

from PySide6.QtCore import (
    Qt,
    QSize,
    QRect,
    QRectF,
    QPoint,
    QTimer,
    QObject,
    Signal,
    Slot,
    QEvent,
    QElapsedTimer,
    QStandardPaths,
    QThread,
)
from PySide6.QtGui import (
    QPixmap,
    QKeySequence,
    QAction,
    QPainter,
    QPen,
    QColor,
    QFontDatabase,
    QFont,
    QIcon,
    QImage,
    QPolygon,
    QPainterPath,
    QBrush,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QFileDialog, QMessageBox, QFrame,
    QStatusBar, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QStackedWidget,
    QToolBar, QDialog, QFormLayout, QGridLayout, QSpinBox, QLineEdit, QDialogButtonBox,
    QSizePolicy, QGroupBox, QGraphicsDropShadowEffect, QRadioButton, QSpacerItem,
    QProgressDialog, QScrollArea, QCheckBox
)

try:
    import psutil
except Exception:
    psutil = None


def theme_color(path: str) -> str:
    group, key = path.split('.')
    return THEME_COLORS[group][key]


def theme_qcolor(path: str) -> QColor:
    return QColor(theme_color(path))

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

DEFAULT_HOTKEYS = OrderedDict([
    ('next', 'D, Right'),
    ('prev', 'A, Left'),
    ('toggle_select', 'Space, S'),
    ('unselect', 'X'),
    ('toggle_zebra', 'Q'),
    ('toggle_hdr', 'E'),
    ('show_filter_dialog', 'F'),
    # ('save', ''),
    ('export', 'Ctrl+S'),
    ('help', 'F1'),
    ('quit', ''),
    ('rate_1', '1'),
    ('rate_2', '2'),
    ('rate_3', '3'),
    ('rate_4', '4'),
    ('rate_5', '5'),
    ('label_red', '6'),
    ('label_yellow', '7'),
    ('label_green', '8'),
    ('label_blue', '9'),
    ('label_purple', '0'),
])

OPEN_EYE_BUCKETS: List[Tuple[str, str]] = [
    ("error", "Processing failed"),
    ("no_faces", "No faces detected"),
    ("eyes_failed", "Faces detected but eyes missing"),
    ("0_50", "0–50% eyes open"),
    ("50_80", "50–80% eyes open"),
    ("80_90", "80–90% eyes open"),
    ("90_99", "90–99% eyes open"),
    ("100", "100% eyes open"),
]
OPEN_EYE_BUCKET_LABELS = {key: label for key, label in OPEN_EYE_BUCKETS}

SHARPNESS_BUCKETS: List[Tuple[str, str]] = [
    ("very_blurry", "Very blurry"),
    ("blurry", "Blurry"),
    ("moderate", "Moderate"),
    ("sharp", "Sharp"),
    ("very_sharp", "Very sharp"),
]
SHARPNESS_BUCKET_LABELS = {key: label for key, label in SHARPNESS_BUCKETS}

ANALYSIS_VERSION = 2


def _resolve_exiftool_path() -> Optional[str]:
    candidates: List[str] = []
    env_path = os.environ.get('EXIFTOOL_PATH') or os.environ.get('exiftool_path')
    if env_path:
        candidates.append(env_path)
    candidates.extend([
        'exiftool',
        'exiftool.exe',
        'exiftool(-k).exe',
    ])
    for cand in candidates:
        if not cand:
            continue
        if os.path.isabs(cand):
            if os.path.exists(cand) and os.access(cand, os.X_OK):
                return cand
        else:
            resolved = shutil.which(cand)
            if resolved:
                return resolved
    return None


EXIFTOOL_PATH = _resolve_exiftool_path()


def classify_eye_bucket(open_people: int, face_people: int, valid_people: Optional[int] = None) -> Tuple[str, float]:
    if face_people <= 0:
        return "no_faces", 0.0
    if valid_people is None:
        valid_people = face_people
    if valid_people <= 0:
        return "eyes_failed", 0.0
    denominator = max(valid_people, 1)
    ratio = max(0.0, min(1.0, open_people / denominator))
    if ratio >= 0.999 and valid_people == face_people and open_people >= face_people:
        return "100", 1.0
    if ratio >= 0.9:
        bucket = "90_99"
    elif ratio >= 0.8:
        bucket = "80_90"
    elif ratio >= 0.5:
        bucket = "50_80"
    else:
        bucket = "0_50"
    return bucket, ratio


def classify_sharpness_bucket(score: float) -> str:
    if score < 5.0:
        return "very_blurry"
    if score < 12.0:
        return "blurry"
    if score < 22.0:
        return "moderate"
    if score < 35.0:
        return "sharp"
    return "very_sharp"


_HOTKEY_STAR_COLOR = "#f5c518"
_HOTKEY_COLOR_SWATCHES = {
    'label_red': ("Red", "#ff5f5f"),
    'label_yellow': ("Yellow", "#f6d354"),
    'label_green': ("Green", "#7dd57d"),
    'label_blue': ("Blue", "#6aa5ff"),
    'label_purple': ("Purple", "#c98dff"),
}


THEME_COLORS = {
    'bg': {
        'base': '#36363a',
        'surface': '#3f3f44',
        'elevated': '#49494f',
        'photo': '#808080',
    },
    'border': {
        'default': '#5c5c63',
    },
    'text': {
        'primary': '#f1f1f3',
        'secondary': '#cbcbcf',
        'tertiary': '#a0a0a6',
        'on_accent': '#f7f7f9',
    },
    'accent': {
        'super': '#41E27F',
        'primary': '#39a96b',
        'hover': '#49bd7d',
        'active': '#2f8d59',
        'muted': '#1a3425',
    },
    'badge': {
        'text': '#dedee2',
    },
    'scrollbar': {
        'track': '#2f2f33',
        'thumb': '#4a4a50',
        'thumb_hover': '#55555c',
        'thumb_active': '#606068',
    },
}


def _styled_hotkey_label(action: str, fallback_text: str) -> Tuple[str, bool]:
    if action.startswith('rate_') and action[-1].isdigit():
        rating = action[-1]
        return f"{rating}<span style=\"color:{_HOTKEY_STAR_COLOR};\">★</span> Rating:", True

    swatch = _HOTKEY_COLOR_SWATCHES.get(action)
    if swatch is not None:
        color_name, color_value = swatch
        return (
            f"Label {color_name}<span style=\"color:{color_value};\">●</span>:",
            True,
        )

    return fallback_text, False

_XMP_GLOBAL_LOCK = threading.Lock()

_XMP_MINIMAL_TEMPLATE = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" \
    "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\">\n" \
    "  <rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n" \
    "    <rdf:Description rdf:about=\"\"\n" \
    "      xmlns:xmp=\"http://ns.adobe.com/xap/1.0/\"\n" \
    "      xmlns:photoshop=\"http://ns.adobe.com/photoshop/1.0/\">\n" \
    "    </rdf:Description>\n" \
    "  </rdf:RDF>\n" \
    "</x:xmpmeta>\n"""
class HotkeyDialog(QDialog):
    def __init__(self, parent: QWidget, hotkeys: Dict[str, str]):
        super().__init__(parent)
        self.setObjectName("HotkeyDialog")
        self.setWindowTitle("Quick Start Guide")
        self.setModal(True)
        self.setMinimumWidth(560)
        self.setMinimumHeight(520)
        self.setMaximumHeight(640)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.hotkeys = hotkeys

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 24)
        layout.setSpacing(18)

        title = QLabel("Welcome to Simple Raw Picker", self)
        title.setObjectName("DialogTitle")
        title.setAlignment(Qt.AlignCenter)
        title.setWordWrap(True)
        layout.addWidget(title)

        subtitle = QLabel(
            "Start by opening a folder of RAW files. These quick tips show how to review, pick, and export efficiently.",
            self,
        )
        subtitle.setObjectName("DialogSubtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        layout.addSpacing(6)

        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(scroll_area, 1)

        content_widget = QWidget(scroll_area)
        scroll_area.setWidget(content_widget)

        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(14)

        content_layout.addWidget(self._create_step_card(
            "1. Prepare your shoot folder",
            self._build_prepare_folder_text(),
        ))

        content_layout.addWidget(self._create_step_card(
            "2. Choose a folder on the start screen",
            "<p>On launch you'll see the welcome screen. Click <b>Select Folder to Start</b> or pick from the recent list to jump back in.</p>",
        ))

        content_layout.addWidget(self._create_step_card(
            "3. Review and cull",
            self._build_review_text(),
        ))

        content_layout.addWidget(self._create_step_card(
            "4. Export your picks",
            self._build_export_text(),
        ))

        content_layout.addStretch(1)

        content_layout.addWidget(self._create_step_card(
            "",
            "Shortcuts can be viewed and customized in Settings."
        ))

        footer = QLabel("Press F1 any time to reopen this guide.", self)
        footer.setObjectName("DialogFooter")
        footer.setAlignment(Qt.AlignCenter)
        footer.setWordWrap(True)
        layout.addWidget(footer)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(12)
        button_row.addStretch(1)

        close_btn = QPushButton("Let's start", self)
        close_btn.setObjectName("DialogPrimaryButton")
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setDefault(True)
        close_btn.clicked.connect(self.accept)
        button_row.addWidget(close_btn, 0, Qt.AlignRight)

        layout.addLayout(button_row)

        self.setStyleSheet(f"""
            QDialog#HotkeyDialog {{
                background-color: {theme_color('bg.elevated')};
                border: 1px solid {theme_color('border.default')};
            }}
            QLabel#DialogTitle {{
                font-size: 18pt;
                font-weight: 700;
                color: {theme_color('text.primary')};
            }}
            QLabel#DialogSubtitle {{
                font-size: 11pt;
                color: {theme_color('text.secondary')};
            }}
            QLabel#DialogFooter {{
                color: {theme_color('text.tertiary')};
                font-size: 9.5pt;
            }}
            QScrollArea {{
                background: transparent;
                border: none;
            }}
            QScrollArea > QWidget > QWidget {{
                background: transparent;
            }}
            QScrollBar:vertical {{
                background-color: {theme_color('scrollbar.track')};
                width: 12px;
                margin: 6px 0 6px 4px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {theme_color('scrollbar.thumb')};
                min-height: 40px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {theme_color('scrollbar.thumb_hover')};
            }}
            QScrollBar::handle:vertical:pressed {{
                background-color: {theme_color('scrollbar.thumb_active')};
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {{
                background: none;
                height: 0px;
            }}
            QWidget#SectionCard {{
                background-color: {theme_color('bg.surface')};
                border-radius: 14px;
                border: 1px solid {theme_color('border.default')};
            }}
            QLabel#SectionHeading {{
                font-size: 11pt;
                font-weight: 600;
                color: {theme_color('accent.primary')};
            }}
            QLabel#ShortcutDescription {{
                color: {theme_color('text.primary')};
                font-size: 10.5pt;
            }}
            QLabel#ShortcutKey {{
                font-size: 10.5pt;
                color: {theme_color('text.secondary')};
            }}
            QPushButton#DialogPrimaryButton {{
                background-color: {theme_color('accent.primary')};
                color: {theme_color('text.on_accent')};
                font-weight: 600;
                border-radius: 9px;
                padding: 8px 20px;
            }}
            QPushButton#DialogPrimaryButton:hover {{
                background-color: {theme_color('accent.hover')};
            }}
            QPushButton#DialogPrimaryButton:pressed {{
                background-color: {theme_color('accent.active')};
            }}
        """)

    def _create_step_card(self, heading: str, body_html: str) -> QWidget:
        card = QWidget(self)
        card.setObjectName("SectionCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)

        if(heading!=""):
            heading_lbl = QLabel(heading, card)
            heading_lbl.setObjectName("SectionHeading")
            heading_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            layout.addWidget(heading_lbl)

        if(body_html!=""):
            body_lbl = QLabel(card)
            body_lbl.setObjectName("ShortcutDescription")
            body_lbl.setTextFormat(Qt.RichText)
            body_lbl.setWordWrap(True)
            body_lbl.setText(body_html)
            layout.addWidget(body_lbl)

        return card

    def _build_prepare_folder_text(self) -> str:
        raw_exts = ", ".join(sorted(ext.upper() for ext in SUPPORTED_EXTS))
        return (
            "<ul style='margin:0; padding-left:18px;'>"
            f"<li>Simple Raw Picker recognizes RAW files with these extensions: {_h(raw_exts)}.</li>"
            "<li>Store the photos directly in the folder you open—subfolders aren't scanned.</li>"
            "<li>Keep any matching JPEG next to the RAW file to have it copied alongside during export.</li>"
            "<li>Ratings, color labels, and picks are written to XMP sidecars.</li>"
            "</ul>"
        )

    def _build_review_text(self) -> str:
        return (
            "<ul style='margin:0; padding-left:18px;'>"
            "<li>Use the arrow keys to move through the roll and press <b>Space</b> to toggle a pick.</li>"
            "<li>Scroll or drag to zoom in for detail checks.</li>"
            "<li>Press <b>F</b> to show only the photos you've picked.</li>"
            "<li>Toggle the zebra/histogram overlay with <b>Q</b> and the Faux HDR preview with <b>E</b>.</li>"
            "<li>Apply star ratings (1–5) and color labels (6–0); they're saved straight into the XMP.</li>"
            "</ul>"
        )

    def _build_export_text(self) -> str:
        return (
            "<ul style='margin:0; padding-left:18px;'>"
            "<li>Hit <b>Complete</b> on the toolbar to export the photos you've picked.</li>"
            "<li>Your RAW selections are copied to <code>_selected_raw</code> together with their XMP.</li>"
            "<li>If a JPEG shares the same name, it's copied to <code>_selected_jpeg</code> automatically.</li>"
            "</ul>"
        )


def _ensure_xmp_file_exists(xmp_path: str):
    if os.path.exists(xmp_path):
        return
    base_dir = os.path.dirname(xmp_path)
    if base_dir and not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    with open(xmp_path, 'w', encoding='utf-8') as f:
        f.write(_XMP_MINIMAL_TEMPLATE)


def read_xmp_sidecar(path: str) -> Dict:
    """Reads rating, label, and pick status from an XMP sidecar file."""
    if EXIFTOOL_PATH is None:
        return {}

    xmp_path = os.path.splitext(path)[0] + '.xmp'

    with _XMP_GLOBAL_LOCK:
        try:
            if not os.path.exists(xmp_path) or os.path.getsize(xmp_path) == 0:
                return {}
        except FileNotFoundError:
            return {}
        except Exception:
            return {}

        try:
            proc = subprocess.run(
                [EXIFTOOL_PATH, '-j', '-XMP:Rating', '-XMP:Label', '-XMP-photoshop:Urgency', xmp_path],
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception as e:
            print(f"Warning: Could not read XMP for {os.path.basename(path)}: {e}")
            return {}

        if proc.returncode != 0 or not proc.stdout.strip():
            return {}

        try:
            payload = json.loads(proc.stdout)
        except Exception as e:
            print(f"Warning: Failed to parse XMP JSON for {os.path.basename(path)}: {e}")
            return {}

        if not isinstance(payload, list) or not payload:
            return {}

        entry = payload[0]
        data: Dict[str, object] = {}

        rating_val = entry.get('XMP:Rating') or entry.get('Rating')
        if rating_val not in (None, ''):
            try:
                data['rating'] = int(str(rating_val))
            except Exception:
                pass

        label_val = entry.get('XMP:Label') or entry.get('Label')
        if label_val not in (None, ''):
            data['color_label'] = str(label_val)

        urgency_val = (
            entry.get('XMP-photoshop:Urgency')
            or entry.get('Photoshop:Urgency')
            or entry.get('Urgency')
        )
        if urgency_val not in (None, ''):
            try:
                urgency_int = int(str(urgency_val))
                data['selected'] = urgency_int == 1 or urgency_int > 0
            except Exception:
                pass

        return data

def write_xmp_sidecar(path: str, data: Dict):
    """Writes rating, label, or pick status to an XMP sidecar file."""
    if EXIFTOOL_PATH is None:
        return False
    xmp_path = os.path.splitext(path)[0] + '.xmp'

    with _XMP_GLOBAL_LOCK:
        try:
            _ensure_xmp_file_exists(xmp_path)
        except Exception as e:
            print(f"Error preparing XMP sidecar for {os.path.basename(path)}: {e}")
            return False

        cmd: List[str] = [
            EXIFTOOL_PATH,
            '-overwrite_original',
            '-P',
            '-m',
        ]

        rating_val = data.get('rating') if 'rating' in data else None
        if rating_val is not None:
            try:
                rating_int = int(rating_val)
                if rating_int <= 0:
                    cmd.append('-XMP:Rating=')
                else:
                    cmd.append(f'-XMP:Rating={rating_int}')
            except Exception:
                pass

        if 'color_label' in data:
            label_val = data.get('color_label')
            if label_val:
                cmd.append(f'-XMP:Label={label_val}')
            else:
                cmd.append('-XMP:Label=')

        if 'selected' in data and data['selected'] is not None:
            cmd.append(f"-XMP-photoshop:Urgency={'1' if data['selected'] else '0'}")

        if len(cmd) == 4:
            return True

        cmd.append(xmp_path)

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except Exception as e:
            print(f"Error writing XMP for {os.path.basename(path)}: {e}")
            return False

        if proc.returncode != 0:
            stderr = proc.stderr.strip() if proc.stderr else ''
            if stderr:
                print(f"Error writing XMP for {os.path.basename(path)}: {stderr}")
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

def _exiftool_read_meta_text(path: str) -> Dict[str, str]:
    if EXIFTOOL_PATH is None:
        return {}

    tags = [
        '-Make',
        '-Model',
        '-LensModel',
        '-Lens',
        '-LensID',
        '-LensSpec',
        '-LensInfo',
        '-FNumber',
        '-ApertureValue',
        '-ExposureTime',
        '-ShutterSpeedValue',
        '-ISO',
        '-PhotographicSensitivity',
        '-FocalLength',
        '-DateTimeOriginal',
        '-CreateDate',
        '-ModifyDate',
    ]

    try:
        proc = subprocess.run(
            [EXIFTOOL_PATH, '-j', *tags, path],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return {}

    if proc.returncode != 0 or not proc.stdout.strip():
        return {}

    try:
        payload = json.loads(proc.stdout)
    except Exception:
        return {}

    if not isinstance(payload, list) or not payload:
        return {}

    entry = payload[0]

    def first(*keys) -> Optional[str]:
        for key in keys:
            val = entry.get(key)
            if val not in (None, ''):
                return str(val)
        return None

    meta: Dict[str, str] = {}
    make = first('Make')
    model = first('Model')
    lens = first('LensModel', 'Lens', 'LensSpec', 'LensID', 'LensInfo')
    fnumber = first('FNumber', 'ApertureValue')
    exposure = first('ExposureTime', 'ShutterSpeedValue')
    iso_val = first('PhotographicSensitivity', 'ISO')
    focal = first('FocalLength')
    dt_val = first('DateTimeOriginal', 'CreateDate', 'ModifyDate')

    if make:
        meta['make'] = make
    if model:
        meta['model'] = model
    if lens:
        meta['lens'] = lens
    if fnumber:
        meta['fnumber_text'] = fnumber
    if exposure:
        meta['exp_text'] = exposure
    if iso_val:
        meta['iso'] = iso_val
    if focal:
        meta['fl'] = focal
    if dt_val:
        meta['dt'] = dt_val

    return meta

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
    ext = _exiftool_read_meta_text(path)
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
    file_mtime_ns: int = 0
    selected: bool = False
    rating: int = 0
    color_label: str = ""

    xmp_loaded: bool = False
    is_dirty: bool = False
    is_saving: bool = False
    version: int = 0
    saving_version: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    analysis_version: int = 0
    analysis_at: float = 0.0
    analysis_status: str = ""
    analysis_error: str = ""
    faces_count: int = 0
    faces_with_eyes: int = 0
    eyes_open_count: int = 0
    eyes_open_ratio: float = 1.0
    eyes_open_bucket: str = "100"
    sharpness_score: float = 0.0
    sharpness_bucket: str = "unknown"


@dataclass
class FilterState:
    selected_only: bool = False
    eye_buckets: Set[str] = field(default_factory=set)
    sharpness_buckets: Set[str] = field(default_factory=set)

    def copy(self) -> 'FilterState':
        return FilterState(
            selected_only=self.selected_only,
            eye_buckets=set(self.eye_buckets),
            sharpness_buckets=set(self.sharpness_buckets),
        )

    def is_active(self) -> bool:
        return self.selected_only or bool(self.eye_buckets) or bool(self.sharpness_buckets)

    def update_xmp(self, data: Dict):
        """Update the in-memory state and set the dirty flag."""
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

    @staticmethod
    def _entry_to_item(entry) -> Optional[Tuple[datetime, str, int, int]]:
        path = entry.path
        try:
            st = entry.stat(follow_symlinks=False)
        except Exception:
            st = None
        dt = read_exif_datetime(path, st)
        if not dt:
            return None
        try:
            sz = st.st_size if st is not None else os.path.getsize(path)
        except Exception:
            sz = 0
        try:
            mtime_ns = getattr(st, 'st_mtime_ns', int(st.st_mtime * 1e9)) if st is not None else int(os.path.getmtime(path) * 1e9)
        except Exception:
            mtime_ns = 0
        return dt, path, sz, mtime_ns

    def _index(self):
        app = QApplication.instance()
        progress = None
        items: List[Tuple[datetime, str, int, int]] = []

        def _create_progress_dialog(total: int = 0) -> QProgressDialog:
            parent = app.activeWindow() if app else None
            dlg = QProgressDialog("Indexing photos…", "", 0, total, parent)
            dlg.setCancelButton(None)
            dlg.setWindowModality(Qt.WindowModal)
            dlg.setMinimumDuration(0)
            return dlg

        if app:
            progress = _create_progress_dialog()
            progress.setRange(0, 0)
            progress.setValue(0)
            progress.show()
            QApplication.processEvents()

        total = 0
        for total, _ in enumerate(self._iter_files(), start=1):
            if progress and progress.maximum() == 0:
                progress.setLabelText(f"Indexing photos… ({total} files found)")
                if total % 25 == 0:
                    QApplication.processEvents()

        if total == 0:
            if progress:
                progress.setRange(0, 1)
                progress.setValue(1)
                QApplication.processEvents()
                progress.close()
            self.photos = []
            return

        if progress:
            progress.setRange(0, total)
            progress.setValue(0)
            QApplication.processEvents()

        processed = 0
        for entry in self._iter_files():
            item = self._entry_to_item(entry)
            if item is not None:
                items.append(item)
            processed += 1
            if progress:
                if progress.maximum() != total:
                    progress.setRange(0, total)
                progress.setValue(processed)
                if processed % 25 == 0 or processed == total:
                    QApplication.processEvents()

        items.sort(key=lambda x: x[0])

        if progress:
            progress.close()

        self.photos = [Photo(path=p, timestamp=dt, filesize=sz, file_mtime_ns=mt) for dt, p, sz, mt in items]

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
        target_w_px = max(1, int(area.width() * dpr))
        target_h_px = max(1, int(area.height() * dpr))

        with _ptime(
            f"pm.scaled target={target_w_px}x{target_h_px}",
            warn_ms=10,
        ):
            pm2 = pm.scaled(
                QSize(target_w_px, target_h_px),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            pm2.setDevicePixelRatio(dpr)

        disp_w = max(1, int(round(pm2.width() / dpr)))
        disp_h = max(1, int(round(pm2.height() / dpr)))

        x = area.x() + (area.width() - disp_w) // 2
        y = area.y() + (area.height() - disp_h) // 2
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
        painter.setPen(QPen(theme_qcolor('text.primary')))
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
        painter.fillRect(self.rect(), theme_qcolor('bg.base'))

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
                painter.fillRect(background_frame, theme_qcolor('accent.super'))

        photo_brush = theme_qcolor('bg.photo')

        if (self._rendered_pixmap and not self._rendered_pixmap.isNull()
                and not self.show_hdr):
            if self._mode == 'fit':
                if self._pm_rect.isValid():
                    painter.fillRect(self._pm_rect, photo_brush)
                painter.drawPixmap(self._pm_rect, self._rendered_pixmap)
            else:
                base = self._pil_base
                if base and base.width > 0 and base.height > 0:
                    target_rect = QRect(
                        self._top_left,
                        QSize(int(base.width * self._zoom), int(base.height * self._zoom))
                    )
                    painter.fillRect(target_rect, photo_brush)
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

                        target_qrect = QRect(int(tgt.x()), int(tgt.y()), int(tgt.width()), int(tgt.height()))
                        painter.fillRect(target_qrect, photo_brush)

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
            overlay[mh & stripe] = [255, 39, 39, 255]
            overlay[(ml & stripe) & ~mh] = [75, 75, 255, 255]
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
        p.fillRect(self.rect(), theme_qcolor('bg.surface'))
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
        p.fillRect(r, theme_qcolor('bg.elevated'))

        if pm is not None and not pm.isNull():
            scaled_size = pm.size().scaled(r.size(), Qt.KeepAspectRatio)
            
            x = r.x() + (r.width() - scaled_size.width()) / 2
            y = r.y() + (r.height() - scaled_size.height()) / 2
            target_rect = QRect(int(x), int(y), scaled_size.width(), scaled_size.height())
            
            p.drawPixmap(target_rect, pm, pm.rect())
        
        self._draw_rating(p, r, it.get('rating', 0))
        
        labelBorderColor = theme_qcolor('border.default')
        selectedBorderWidth = 2
        currentLineWidth = 6

        if it.get('current'):
            pen = QPen(theme_qcolor('accent.muted'))
            pen.setWidth(currentLineWidth)
            p.setPen(pen)
            p.drawRect(r.adjusted(-3, -3, 3, 3))
            labelBorderColor = theme_qcolor('accent.muted')

        if it.get('selected'):
            pen = QPen(theme_qcolor('accent.super'))
            pen.setWidth(selectedBorderWidth)
            p.setPen(pen)
            p.drawRect(r)
            labelBorderColor = theme_qcolor('accent.super')
            
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

            pen = QPen(theme_qcolor('border.default'), 2)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)

            brush = QBrush(QColor(_HOTKEY_STAR_COLOR))
            p.setBrush(brush)
            p.setPen(Qt.NoPen)
            p.drawPath(path)
            p.setBrush(Qt.NoBrush)

        empty_rect = QRect(text_rect.left() + filled_width, text_rect.top(), text_rect.width() - filled_width, text_rect.height())
        p.setPen(theme_qcolor('text.tertiary'))
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
    raw_output_folder_name: str = "_selected_raw"
    jpeg_output_folder_name: str = "_selected_jpeg"
    hotkeys: Dict[str, str] = field(default_factory=lambda: OrderedDict(DEFAULT_HOTKEYS))

    def __post_init__(self):
        ordered = OrderedDict()
        for key, default_value in DEFAULT_HOTKEYS.items():
            ordered[key] = self.hotkeys.get(key, default_value)
        for key, value in self.hotkeys.items():
            if key not in ordered:
                ordered[key] = value
        self.hotkeys = ordered


@dataclass
class ExportResult:
    copied_raw: int
    raw_out_dir: str
    copied_jpg: int
    jpeg_out_dir: str
    selected_count: int
    dest_raw_count: int
    dest_jpeg_count: int


class ExportWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(ExportResult)
    error = Signal(str, str)
    canceled = Signal()

    def __init__(
        self,
        root: str,
        selected_raw_paths: List[str],
        raw_output_folder_name: str,
        jpeg_output_folder_name: str,
    ):
        super().__init__()
        self.root = root
        self.selected_raw_paths = selected_raw_paths
        self.raw_output_folder_name = raw_output_folder_name
        self.jpeg_output_folder_name = jpeg_output_folder_name
        self._cancel_requested = threading.Event()

    @Slot()
    def run(self):
        try:
            result = self._sync()
            if result is None or self._cancel_requested.is_set():
                self.canceled.emit()
            else:
                self.finished.emit(result)
        except Exception as exc:
            tb = traceback.format_exc()
            self.error.emit(str(exc), tb)

    def request_cancel(self):
        self._cancel_requested.set()

    def _sync(self) -> Optional[ExportResult]:
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

        def _list_paths_by_basename(folder: str, exts: set) -> Dict[str, List[str]]:
            out: Dict[str, List[str]] = {}
            try:
                for fn in os.listdir(folder):
                    p = os.path.join(folder, fn)
                    if not os.path.isfile(p):
                        continue
                    base, ext = os.path.splitext(fn)
                    if ext.lower() in exts:
                        out.setdefault(base, []).append(p)
            except Exception:
                pass
            return out

        selected_paths = [p for p in self.selected_raw_paths if os.path.isfile(p)]
        selected_count = len(selected_paths)
        raw_out_dir = os.path.join(self.root, self.raw_output_folder_name)
        jpeg_out_dir = os.path.join(self.root, self.jpeg_output_folder_name)

        os.makedirs(raw_out_dir, exist_ok=True)
        os.makedirs(jpeg_out_dir, exist_ok=True)

        selected_raw_by_base = {
            os.path.splitext(os.path.basename(p))[0]: p for p in selected_paths
        }
        root_jpegs_by_base = _list_paths_by_basename(self.root, {".jpg", ".jpeg"})

        dest_raw_by_base = _list_paths_by_basename(
            raw_out_dir, SUPPORTED_EXTS.union({".xmp"})
        )
        dest_jpg_by_base = _list_paths_by_basename(
            jpeg_out_dir, {".jpg", ".jpeg"}
        )

        if self._cancel_requested.is_set():
            return None

        for base, dst_paths in list(dest_raw_by_base.items()):
            if base not in selected_raw_by_base:
                for dst_path in dst_paths:
                    if self._cancel_requested.is_set():
                        return None
                    try:
                        os.remove(dst_path)
                    except Exception:
                        pass

        desired_jpg_bases = {
            b for b in selected_raw_by_base if b in root_jpegs_by_base
        }

        for base, dst_paths in list(dest_jpg_by_base.items()):
            if base not in desired_jpg_bases:
                for dst_path in dst_paths:
                    if self._cancel_requested.is_set():
                        return None
                    try:
                        os.remove(dst_path)
                    except Exception:
                        pass

        tasks: List[Tuple[str, str, str]] = []  # (src, dst, kind)
        for base, src_path in selected_raw_by_base.items():
            dst_path = os.path.join(raw_out_dir, os.path.basename(src_path))
            if _needs_copy(src_path, dst_path):
                tasks.append((src_path, dst_path, "raw"))
            src_xmp = os.path.splitext(src_path)[0] + ".xmp"
            dst_xmp = os.path.splitext(dst_path)[0] + ".xmp"
            if os.path.exists(src_xmp) and _needs_copy(src_xmp, dst_xmp):
                tasks.append((src_xmp, dst_xmp, "xmp"))

        for base in desired_jpg_bases:
            src_jpg = root_jpegs_by_base[base][0]
            dst_jpg = os.path.join(jpeg_out_dir, os.path.basename(src_jpg))
            if _needs_copy(src_jpg, dst_jpg):
                tasks.append((src_jpg, dst_jpg, "jpeg"))

        total_tasks = len(tasks) if tasks else 1
        self.progress.emit(0, total_tasks, "Preparing export...")

        copied_raw = 0
        copied_jpg = 0
        completed = 0

        for src, dst, kind in tasks:
            if self._cancel_requested.is_set():
                return None
            name = os.path.basename(src)
            self.progress.emit(completed, total_tasks, f"Copying {name}")
            try:
                shutil.copy2(src, dst)
            except Exception as exc:
                raise RuntimeError(f"Failed to copy {src} -> {dst}: {exc}") from exc
            completed += 1
            if kind == "raw":
                copied_raw += 1
            elif kind == "jpeg":
                copied_jpg += 1
            self.progress.emit(completed, total_tasks, f"Copied {name}")

        if self._cancel_requested.is_set():
            return None

        dest_raw_count = len(_list_paths_by_basename(raw_out_dir, SUPPORTED_EXTS))
        dest_jpeg_count = len(
            _list_paths_by_basename(jpeg_out_dir, {".jpg", ".jpeg"})
        )

        self.progress.emit(total_tasks, total_tasks, "Export complete.")

        return ExportResult(
            copied_raw=copied_raw,
            raw_out_dir=raw_out_dir,
            copied_jpg=copied_jpg,
            jpeg_out_dir=jpeg_out_dir,
            selected_count=selected_count,
            dest_raw_count=dest_raw_count,
            dest_jpeg_count=dest_jpeg_count,
        )


class AnalysisStore:
    FILE_NAME = "analysis.json"

    def __init__(self, root: str):
        self.root = root
        self.path = os.path.join(root, self.FILE_NAME)
        self.data: Dict[str, Dict] = {}
        self._load()

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except FileNotFoundError:
            self.data = {}
            return
        except Exception:
            print(f"Failed to load analysis data from {self.path}")
            self.data = {}
            return

        if not isinstance(payload, dict):
            self.data = {}
            return
        if payload.get("version") != ANALYSIS_VERSION:
            self.data = {}
            return

        items = payload.get("items")
        if isinstance(items, dict):
            self.data = items
        else:
            self.data = {}

    def save(self):
        payload = {"version": ANALYSIS_VERSION, "items": self.data}
        tmp_path = self.path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp_path, self.path)
        except Exception:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    def get(self, path: str) -> Optional[Dict]:
        return self.data.get(path)

    def update_many(self, results: Dict[str, Dict]):
        for path, payload in results.items():
            payload = dict(payload)
            payload["version"] = ANALYSIS_VERSION
            self.data[path] = payload
        self.save()

    def prune(self, valid_paths: Set[str]) -> List[str]:
        removed: List[str] = []
        for path in list(self.data.keys()):
            if path not in valid_paths:
                removed.append(path)
                self.data.pop(path, None)
        if removed:
            self.save()
        return removed


class AnalysisWorker(QObject):
    progress = Signal(int, int, float, str)
    finished = Signal(dict, bool)
    error = Signal(str, str)

    def __init__(self, photos: List[Photo]):
        super().__init__()
        self.photos = photos
        self._cancel_requested = threading.Event()
        self._face_mesh = self._create_face_mesh()
        self._ear_threshold = 0.28

    @staticmethod
    def _create_face_mesh():
        if mp is None:
            return None
        try:
            return mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                refine_landmarks=True,
                max_num_faces=12,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception:
            return None

    def request_cancel(self):
        self._cancel_requested.set()

    @Slot()
    def run(self):
        results: Dict[str, Dict] = {}
        total = len(self.photos)
        start = time.perf_counter()
        for index, photo in enumerate(self.photos, start=1):
            if self._cancel_requested.is_set():
                break
            try:
                payload = self._process_photo(photo)
            except Exception as exc:
                tb = traceback.format_exc()
                message = str(exc)
                self.error.emit(message, tb)
                payload = self._build_error_payload(photo, message)
            if payload is not None:
                results[photo.path] = payload
            elapsed = time.perf_counter() - start
            name = os.path.basename(photo.path)
            self.progress.emit(index, total, elapsed, name)
        canceled = self._cancel_requested.is_set()
        self.finished.emit(results, canceled)
        if self._face_mesh is not None:
            try:
                self._face_mesh.close()
            except Exception:
                pass
            self._face_mesh = None

    def _process_photo(self, photo: Photo) -> Optional[Dict]:
        if cv2 is None:
            raise RuntimeError("OpenCV is not available for analysis")
        if mp is None:
            raise RuntimeError("MediaPipe is not available for analysis")
        pil = load_half_pil(photo.path)
        if pil is None:
            pil = load_full_pil(photo.path)
        if pil is None:
            raise RuntimeError(f"Failed to decode image: {photo.path}")

        rgb = np.array(pil.convert('RGB'))
        if rgb.size == 0:
            raise RuntimeError(f"Empty image data: {photo.path}")

        h, w = rgb.shape[:2]
        max_side = max(h, w)
        scale = 1.0
        if max_side > 1280:
            scale = 1280.0 / float(max_side)
        if scale != 1.0:
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = rgb

        faces_count, open_count, valid_faces = self._detect_eyes(resized)

        gray_focus = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        focus_score = self._measure_sharpness(gray_focus)

        eye_bucket, ratio = classify_eye_bucket(open_count, faces_count, valid_faces)
        sharp_bucket = classify_sharpness_bucket(focus_score)

        payload = {
            'filesize': photo.filesize,
            'timestamp': photo.timestamp.isoformat(),
            'mtime_ns': photo.file_mtime_ns,
            'analysis_at': time.time(),
            'status': 'ok',
            'error_message': '',
            'faces_count': faces_count,
            'faces_with_eyes': valid_faces,
            'eyes_open_count': open_count,
            'eyes_open_ratio': ratio,
            'eyes_open_bucket': eye_bucket,
            'sharpness_score': float(focus_score),
            'sharpness_bucket': sharp_bucket,
            'analysis_version': ANALYSIS_VERSION,
        }
        return payload

    def _build_error_payload(self, photo: Photo, message: str) -> Dict:
        return {
            'filesize': photo.filesize,
            'timestamp': photo.timestamp.isoformat(),
            'mtime_ns': photo.file_mtime_ns,
            'analysis_at': time.time(),
            'status': 'error',
            'error_message': message,
            'faces_count': 0,
            'faces_with_eyes': 0,
            'eyes_open_count': 0,
            'eyes_open_ratio': 0.0,
            'eyes_open_bucket': 'error',
            'sharpness_score': 0.0,
            'sharpness_bucket': 'unknown',
            'analysis_version': ANALYSIS_VERSION,
        }

    def _detect_eyes(self, rgb_image: np.ndarray) -> Tuple[int, int, int]:
        if self._face_mesh is None:
            raise RuntimeError("MediaPipe Face Mesh is not available for analysis")

        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            return 0, 0, 0

        # MediaPipe expects RGB float arrays marked as not writeable.
        frame = np.ascontiguousarray(rgb_image)
        frame.flags.writeable = False
        results = self._face_mesh.process(frame)
        if not results.multi_face_landmarks:
            return 0, 0, 0

        h, w = frame.shape[:2]
        total_faces = len(results.multi_face_landmarks)
        valid_faces = 0
        open_faces = 0

        for landmarks in results.multi_face_landmarks:
            left_ear = self._compute_eye_aspect_ratio(landmarks, self._LEFT_EYE_IDXS, w, h)
            right_ear = self._compute_eye_aspect_ratio(landmarks, self._RIGHT_EYE_IDXS, w, h)

            ears = [ear for ear in (left_ear, right_ear) if ear is not None and np.isfinite(ear)]
            if not ears:
                continue

            valid_faces += 1
            avg_ear = float(sum(ears) / len(ears))
            if avg_ear >= self._ear_threshold:
                open_faces += 1

        return total_faces, open_faces, valid_faces

    _LEFT_EYE_IDXS = (33, 160, 158, 133, 153, 144)
    _RIGHT_EYE_IDXS = (362, 385, 387, 263, 373, 380)

    @staticmethod
    def _compute_eye_aspect_ratio(landmarks, indices: Tuple[int, ...], width: int, height: int) -> Optional[float]:
        try:
            pts = [landmarks.landmark[i] for i in indices]
        except Exception:
            return None

        coords = []
        for pt in pts:
            if pt is None:
                return None
            if not np.isfinite(pt.x) or not np.isfinite(pt.y):
                return None
            coords.append(np.array([pt.x * width, pt.y * height], dtype=np.float32))

        if len(coords) != 6:
            return None

        p1, p2, p3, p4, p5, p6 = coords
        horizontal = np.linalg.norm(p1 - p4)
        if horizontal < 1e-4:
            return None
        vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
        ear = vertical / (2.0 * horizontal)
        if not np.isfinite(ear):
            return None
        return float(ear)

    def _measure_sharpness(self, gray_image: np.ndarray) -> float:
        if gray_image.size == 0:
            return 0.0
        max_side = max(gray_image.shape[:2])
        if max_side > 2048:
            scale = 2048.0 / float(max_side)
            new_w = max(1, int(round(gray_image.shape[1] * scale)))
            new_h = max(1, int(round(gray_image.shape[0] * scale)))
            gray_image = cv2.resize(gray_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        gray = gray_image.astype(np.float32) / 255.0
        # Suppress sensor noise before measuring edge strength.
        gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0, sigmaY=1.0)

        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)

        # Focus on the sharpest regions by pooling local maxima.
        magnitude = cv2.GaussianBlur(magnitude, (0, 0), sigmaX=1.5, sigmaY=1.5)
        magnitude = cv2.dilate(magnitude, np.ones((5, 5), dtype=np.float32))

        score = float(np.percentile(magnitude, 99.7)) * 100.0
        if not np.isfinite(score):
            return 0.0
        return score


class FilterDialog(QDialog):
    filter_changed = Signal(FilterState)
    process_requested = Signal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setWindowTitle("Filters")
        self.setModal(False)
        self.setMinimumWidth(420)

        self._state = FilterState()
        self._block_updates = False
        self._analysis_ready = False
        self._processing = False
        self._pending_count = 0
        self._analysis_supported = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setSpacing(16)

        self.selected_checkbox = QCheckBox("Show selected photos only", self)
        self.selected_checkbox.stateChanged.connect(self._on_selected_toggled)
        layout.addWidget(self.selected_checkbox)

        self.eye_group = QGroupBox("Eyes open ratio", self)
        eye_layout = QVBoxLayout()
        self.eye_checkboxes: Dict[str, QCheckBox] = {}
        for bucket, label in OPEN_EYE_BUCKETS:
            cb = QCheckBox(label, self.eye_group)
            cb.stateChanged.connect(lambda state, b=bucket: self._on_eye_toggled(b, state))
            self.eye_checkboxes[bucket] = cb
            eye_layout.addWidget(cb)
        self.eye_group.setLayout(eye_layout)
        layout.addWidget(self.eye_group)

        self.sharp_group = QGroupBox("Sharpness", self)
        sharp_layout = QVBoxLayout()
        self.sharp_checkboxes: Dict[str, QCheckBox] = {}
        for bucket, label in SHARPNESS_BUCKETS:
            cb = QCheckBox(label, self.sharp_group)
            cb.stateChanged.connect(lambda state, b=bucket: self._on_sharpness_toggled(b, state))
            self.sharp_checkboxes[bucket] = cb
            sharp_layout.addWidget(cb)
        self.sharp_group.setLayout(sharp_layout)
        layout.addWidget(self.sharp_group)

        self.analysis_notice = QLabel("", self)
        self.analysis_notice.setWordWrap(True)
        layout.addWidget(self.analysis_notice)

        button_row = QHBoxLayout()
        button_row.setSpacing(8)

        self.process_button = QPushButton("Process", self)
        self.process_button.clicked.connect(self._on_process_clicked)
        button_row.addWidget(self.process_button)

        self.reset_button = QPushButton("Reset Filters", self)
        self.reset_button.clicked.connect(self._on_reset_clicked)
        button_row.addWidget(self.reset_button)

        button_row.addStretch(1)

        close_button = QPushButton("Close", self)
        close_button.clicked.connect(self.close)
        button_row.addWidget(close_button)

        layout.addLayout(button_row)

    def set_state(
        self,
        state: FilterState,
        counts: Dict[str, Dict[str, int]],
        *,
        analysis_ready: bool,
        pending_count: int,
        processing: bool,
        analysis_supported: bool = True,
    ):
        self._state = state.copy()
        self._analysis_ready = analysis_ready
        self._pending_count = pending_count
        self._processing = processing
        self._analysis_supported = analysis_supported
        self._apply_counts(counts)
        self._sync_ui()

    def _apply_counts(self, counts: Dict[str, Dict[str, int]]):
        eye_counts = counts.get('eyes', {}) if counts else {}
        for bucket, cb in self.eye_checkboxes.items():
            label = OPEN_EYE_BUCKET_LABELS.get(bucket, bucket)
            count = eye_counts.get(bucket, 0)
            cb.setText(f"{label} ({count})")

        sharp_counts = counts.get('sharpness', {}) if counts else {}
        for bucket, cb in self.sharp_checkboxes.items():
            label = SHARPNESS_BUCKET_LABELS.get(bucket, bucket)
            count = sharp_counts.get(bucket, 0)
            cb.setText(f"{label} ({count})")

        unprocessed = counts.get('unprocessed', 0) if counts else 0
        error_count = eye_counts.get('error', 0)
        if not self._analysis_supported:
            for cb in self.eye_checkboxes.values():
                cb.setEnabled(False)
            for cb in self.sharp_checkboxes.values():
                cb.setEnabled(False)
            self.analysis_notice.setText(
                "Install opencv-python-headless and mediapipe to enable eye openness and sharpness filters."
            )
            self.process_button.setEnabled(False)
            return

        for cb in self.eye_checkboxes.values():
            cb.setEnabled(True)
        for cb in self.sharp_checkboxes.values():
            cb.setEnabled(True)

        if self._processing:
            self.analysis_notice.setText("Processing analysis…")
        elif not self._analysis_ready:
            if self._pending_count > 0:
                self.analysis_notice.setText(
                    f"{self._pending_count} photo(s) require processing before eye and sharpness filters can be used."
                )
            elif unprocessed > 0:
                self.analysis_notice.setText(
                    f"{unprocessed} photo(s) do not yet have analysis data."
                )
            else:
                self.analysis_notice.setText("Analysis is required before using eye or sharpness filters.")
        else:
            self.analysis_notice.setText("Filters are ready. Adjust the options above.")

        self.process_button.setEnabled(not self._processing)

        if error_count > 0:
            extra = f"{error_count} photo(s) failed to analyze. Use the filters above to review them."
            current = self.analysis_notice.text() or ""
            if extra not in current:
                if current:
                    current = current + "\n" + extra
                else:
                    current = extra
                self.analysis_notice.setText(current)

    def _sync_ui(self):
        self._block_updates = True
        try:
            self.selected_checkbox.setChecked(self._state.selected_only)
            for bucket, cb in self.eye_checkboxes.items():
                cb.setChecked(bucket in self._state.eye_buckets)
            for bucket, cb in self.sharp_checkboxes.items():
                cb.setChecked(bucket in self._state.sharpness_buckets)
        finally:
            self._block_updates = False
        self.reset_button.setEnabled(self._state.is_active())

    def _on_selected_toggled(self, state: int):
        if self._block_updates:
            return
        self._state.selected_only = bool(state)
        self.reset_button.setEnabled(self._state.is_active())
        self.filter_changed.emit(self._state.copy())

    def _on_eye_toggled(self, bucket: str, state: int):
        if self._block_updates:
            return
        if not self._analysis_ready:
            self._prompt_analysis_required()
            self._sync_ui()
            return
        if state:
            self._state.eye_buckets.add(bucket)
        else:
            self._state.eye_buckets.discard(bucket)
        self.reset_button.setEnabled(self._state.is_active())
        self.filter_changed.emit(self._state.copy())

    def _on_sharpness_toggled(self, bucket: str, state: int):
        if self._block_updates:
            return
        if not self._analysis_ready:
            self._prompt_analysis_required()
            self._sync_ui()
            return
        if state:
            self._state.sharpness_buckets.add(bucket)
        else:
            self._state.sharpness_buckets.discard(bucket)
        self.reset_button.setEnabled(self._state.is_active())
        self.filter_changed.emit(self._state.copy())

    def _on_process_clicked(self):
        self.process_requested.emit()

    def _on_reset_clicked(self):
        if not self._state.is_active():
            return
        self._state = FilterState()
        self._sync_ui()
        self.filter_changed.emit(self._state.copy())

    def _prompt_analysis_required(self):
        box = QMessageBox(self)
        box.setWindowTitle("Analysis required")
        box.setIcon(QMessageBox.Information)
        box.setText("Eye openness and sharpness filters require processing. Run analysis now?")
        process_button = box.addButton("Process Now", QMessageBox.AcceptRole)
        box.addButton(QMessageBox.Cancel)
        box.exec()
        if box.clickedButton() is process_button:
            self.process_requested.emit()
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

        self.badge_filter = QPushButton("Filter"); self.badge_filter.setObjectName("badgeGhost")
        self.badge_filter.setFlat(True)
        self.badge_filter.setCursor(Qt.PointingHandCursor)
        self.badge_filter.setFocusPolicy(Qt.NoFocus)
        self.badge_filter.clicked.connect(self.show_filter_dialog)
        self.badge_selected = QLabel("Selected: 0"); self.badge_selected.setObjectName("badge")
        self.badge_zebra =   QLabel("Zebra OFF");    self.badge_zebra.setObjectName("badgeGhost")
        self.badge_hdr   =   QLabel("HDR OFF");      self.badge_hdr.setObjectName("badgeGhost")

        mb.addWidget(self.meta_left, 1)
        mb.addSpacing(8)
        mb.addWidget(self.badge_filter, 0, Qt.AlignRight)
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
        self.filter_state = FilterState()
        self._analysis_store = AnalysisStore(root)
        self._analysis_pending_paths: Set[str] = set()
        self._analysis_thread: Optional[QThread] = None
        self._analysis_worker: Optional[AnalysisWorker] = None
        self._analysis_progress_dialog: Optional[QProgressDialog] = None
        self._analysis_start_time: float = 0.0
        self._analysis_average_seconds: float = 1.2
        self._filter_dialog: Optional[FilterDialog] = None
        self._filter_counts: Dict[str, Dict[str, int]] = {}
        self._photo_by_path: Dict[str, Photo] = {}
        self._analysis_supported = (cv2 is not None) and (mp is not None)

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
        default_workers = max(2, min(32, (os.cpu_count() or 4) - 1))
        self._num_workers = int(workers) if (workers and workers > 0) else default_workers
        self._loader_threads: List[threading.Thread] = []
        for i in range(self._num_workers):
            t = threading.Thread(target=self._loader_loop, daemon=True, name=f"loader-{i}")
            t.start(); self._loader_threads.append(t)

        self._full_lock = threading.Lock()
        self._full_running = 0

        self.selections_path = os.path.join(root, 'selections.json')
        self._load_selections()
        self._initialize_analysis_state()

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
        self._hotkey_bindings: Dict[str, List[QAction]] = {}

        self.status_restore_timer = QTimer(self); self.status_restore_timer.setSingleShot(True)
        self.status_restore_timer.timeout.connect(self._refresh_statusbar)

        self._create_actions()
        self.update_settings()
        QApplication.instance().installEventFilter(self)

        self._update_filter_badge()
        self._show_current()
        self._schedule_heavy_load()

        if not self.catalog.photos:
            QMessageBox.information(self, "Information", "No supported photo files found.")
        elif EXIFTOOL_PATH is None:
            QMessageBox.warning(
                self,
                "XMP Features Disabled",
                "Could not find the ExifTool command-line utility.\n"
                "Rating and color label features are disabled.\n"
                "Please install ExifTool from https://exiftool.org/ and ensure it is on your PATH.",
            )

        self.setStyleSheet(f"""
            QWidget#card {{
                background:{theme_color('bg.surface')};
                border:1px solid {theme_color('border.default')};
                border-radius:0px;
            }}
            QWidget#filmWrap {{
                background:{theme_color('bg.elevated')};
                border-top:1px solid {theme_color('border.default')};
            }}
            QWidget#metaBar {{
                background:{theme_color('bg.surface')};
                border-top: 1px solid {theme_color('border.default')};
            }}
            QLabel#metaLeft {{
                color:{theme_color('text.primary')};
            }}
            QLabel#badge, QLabel#badgeGhost, QPushButton#badge, QPushButton#badgeGhost {{
                padding:4px 10px;
                border-radius:999px;
                font-size:10pt;
            }}
            QLabel#badge, QPushButton#badge {{
                background:{theme_color('accent.muted')};
                color:{theme_color('badge.text')};
                border:1px solid {theme_color('accent.primary')};
            }}
            QLabel#badgeGhost, QPushButton#badgeGhost {{
                background:{theme_color('bg.surface')};
                color:{theme_color('text.secondary')};
                border:1px solid {theme_color('border.default')};
            }}
            QPushButton#badge, QPushButton#badgeGhost {{
                min-height: 0px;
            }}
            QPushButton#badge:hover {{
                background:{theme_color('accent.hover')};
            }}
            QPushButton#badgeGhost:hover {{
                background:{theme_color('bg.elevated')};
            }}
        """)

    def update_settings(self):
        self.update_autosave_interval()
        self._apply_hotkeys()
        self._rebuild_hotkey_bindings()

    def update_autosave_interval(self):
        self.autosave_interval_timer.setInterval(self.settings.autosave_interval_min * 60 * 1000)

    def _create_actions(self):
        self.actions: Dict[str, QAction] = {}
        action_map = {
            # 'save': self.save_all_dirty_files,
            'quit': self.window().close,
            'next': self.next_photo,
            'prev': self.prev_photo,
            'toggle_select': self.toggle_select,
            'unselect': self.unselect_current,
            'export': self.export_selected,
            'help': self.show_help,
            'toggle_zebra': self.toggle_zebra,
            'toggle_hdr': self.toggle_hdr,
            'show_filter_dialog': self.show_filter_dialog,
        }
        for name, callback in action_map.items():
            action = QAction(self)
            action.triggered.connect(callback)
            self.actions[name] = action
            self.addAction(action)

        for rating in range(1, 6):
            action = QAction(self)
            action.triggered.connect(lambda checked=False, r=rating: self.set_rating(r))
            name = f'rate_{rating}'
            self.actions[name] = action
            self.addAction(action)

        color_map = {
            'label_red': 'Red',
            'label_yellow': 'Yellow',
            'label_green': 'Green',
            'label_blue': 'Blue',
            'label_purple': 'Purple',
        }
        for name, label in color_map.items():
            action = QAction(self)
            action.triggered.connect(lambda checked=False, lbl=label: self.set_color_label(lbl))
            self.actions[name] = action
            self.addAction(action)
        self._apply_hotkeys()

    def _apply_hotkeys(self):
        for name, key_sequence_str in self.settings.hotkeys.items():
            if name in self.actions:
                sequences = [QKeySequence(s.strip()) for s in key_sequence_str.split(',') if s.strip()]
                self.actions[name].setShortcuts(sequences)

    def _rebuild_hotkey_bindings(self):
        relevant_names = [
            *(f'rate_{i}' for i in range(1, 6)),
            'label_red', 'label_yellow', 'label_green', 'label_blue', 'label_purple',
            'toggle_zebra', 'toggle_hdr', 'show_filter_dialog',
        ]

        bindings: Dict[str, List[QAction]] = {}
        for name in relevant_names:
            action = self.actions.get(name)
            if not action:
                continue
            try:
                shortcuts = action.shortcuts()
            except Exception:
                shortcuts = []
            for seq in shortcuts:
                text = seq.toString(QKeySequence.PortableText)
                if not text:
                    continue
                bindings.setdefault(text, []).append(action)

        self._hotkey_bindings = bindings

    def _update_filter_badge(self):
        if getattr(self, 'badge_filter', None) is None:
            return
        parts: List[str] = []
        if self.filter_state.selected_only:
            parts.append("Selected")
        if self.filter_state.eye_buckets:
            parts.append("Eyes")
        if self.filter_state.sharpness_buckets:
            parts.append("Sharpness")
        text = "Filter"
        if parts:
            text += ": " + ", ".join(parts)
        if self._analysis_pending_paths:
            text += f" ({len(self._analysis_pending_paths)} to process)"
        self.badge_filter.setText(text)
        self.badge_filter.setObjectName("badge" if self.filter_state.is_active() else "badgeGhost")
        self.badge_filter.style().unpolish(self.badge_filter); self.badge_filter.style().polish(self.badge_filter)

    def _active_indices(self) -> List[int]:
        if not self.catalog.photos:
            return []
        indices: List[int] = []
        for idx, photo in enumerate(self.catalog.photos):
            if self.filter_state.selected_only and not photo.selected:
                continue
            if self.filter_state.eye_buckets:
                if photo.analysis_version != ANALYSIS_VERSION or photo.analysis_at <= 0:
                    continue
                if photo.eyes_open_bucket not in self.filter_state.eye_buckets:
                    continue
            if self.filter_state.sharpness_buckets:
                if photo.analysis_version != ANALYSIS_VERSION or photo.analysis_at <= 0:
                    continue
                if photo.sharpness_bucket not in self.filter_state.sharpness_buckets:
                    continue
            indices.append(idx)
        if indices:
            return indices
        if not self.filter_state.is_active():
            return list(range(len(self.catalog.photos)))
        return []

    def _current_position(self, indices: Optional[List[int]] = None) -> int:
        indices = indices if indices is not None else self._active_indices()
        if not indices:
            return -1
        try:
            return indices.index(self.idx)
        except ValueError:
            return -1

    def _update_view_after_selection_change(self, reference_index: Optional[int] = None):
        indices = self._active_indices()
        if indices:
            if self.idx in indices:
                return
            ref = reference_index if reference_index is not None else self.idx
            if reference_index is not None and reference_index in indices:
                self.idx = reference_index
                return
            next_idx = next((i for i in indices if i >= ref), None)
            self.idx = next_idx if next_idx is not None else indices[0]
            return

        if self.filter_state.selected_only and not any(p.selected for p in self.catalog.photos):
            self.filter_state.selected_only = False
            self._update_filter_badge()
            self._refresh_filter_dialog()
            self._show_toast("No selected photos - showing all photos")
            if self.catalog.photos:
                self.idx = max(0, min(self.idx, len(self.catalog.photos) - 1))

    def eventFilter(self, obj, ev: QEvent):
        if ev.type() in (QEvent.Type.KeyPress, QEvent.Type.Wheel, QEvent.Type.MouseButtonPress):
            self._note_user_input()

        capture_widget = None
        key_sequence_cls = globals().get("KeySequenceEdit")
        if key_sequence_cls is not None:
            capture_widget = key_sequence_cls.active_capture_widget()
        capture_active = capture_widget is not None

        if capture_active and ev.type() in (QEvent.Type.Shortcut, QEvent.Type.ShortcutOverride):
            try:
                ev.accept()
            except Exception:
                pass
            return True

        if ev.type() == QEvent.Type.KeyPress:
            if capture_active:
                return False
            is_capture_active = getattr(obj, "is_hotkey_capture_active", None)
            if callable(is_capture_active) and is_capture_active():
                return False

        if ev.type() == QEvent.Type.KeyPress and not ev.isAutoRepeat():
            key_sequence = QKeySequence(ev.keyCombination())
            portable = key_sequence.toString(QKeySequence.PortableText)
            actions = self._hotkey_bindings.get(portable, []) if portable else []

            if not actions and ev.modifiers() == Qt.NoModifier:
                fallback = QKeySequence(ev.key())
                portable_fallback = fallback.toString(QKeySequence.PortableText)
                if portable_fallback and portable_fallback != portable:
                    actions = self._hotkey_bindings.get(portable_fallback, [])

            handled = False
            for action in actions:
                if action and action.isEnabled():
                    action.trigger()
                    handled = True
            if handled:
                return True

        return super().eventFilter(obj, ev)

    def _note_user_input(self):
        now = time.monotonic()
        self._last_input_ts = now

    def _is_metadata_save_task(self, fn, args) -> bool:
        if getattr(fn, "_srp_metadata_save", False):
            return True
        if getattr(fn, "__name__", "") != "_write_task_with_cleanup":
            return False
        return len(args) >= 4 and isinstance(args[2], Photo)

    def _restore_cancelled_save_task(self, args) -> bool:
        try:
            photo_obj = args[2]
            version = args[3]
        except Exception:
            return False
        if not isinstance(photo_obj, Photo):
            return False
        restored = False
        with photo_obj.lock:
            if photo_obj.saving_version == version and photo_obj.is_saving:
                photo_obj.is_saving = False
                photo_obj.is_dirty = True
                photo_obj.saving_version = version
                restored = True
        return restored

    def _flush_queue(self, preserve_metadata: bool = True, preserve_keys: Optional[Set[Tuple]] = None):
        flushed = 0
        preserved: List[Tuple] = []
        preserved_plan = 0
        preserved_metadata = 0
        restores = 0
        preserve_set = set(preserve_keys) if preserve_keys is not None else None
        with self._pending_lock:
            pending_before = len(self._pending_tasks)
        try:
            while True:
                item = self._taskq.get_nowait()
                _, _, fn, args = item
                is_save_task = self._is_metadata_save_task(fn, args)
                task_key = getattr(fn, "_srp_task_key", None)
                keep_for_plan = preserve_set is not None and task_key in preserve_set
                if keep_for_plan or (preserve_metadata and is_save_task):
                    preserved.append(item)
                    if keep_for_plan:
                        preserved_plan += 1
                    if preserve_metadata and is_save_task:
                        preserved_metadata += 1
                else:
                    if is_save_task and self._restore_cancelled_save_task(args):
                        restores += 1
                    if task_key is not None:
                        with self._pending_lock:
                            self._pending_tasks.discard(task_key)
                    flushed += 1
                try:
                    self._taskq.task_done()
                except Exception:
                    pass
        except Empty:
            pass
        for item in preserved:
            self._taskq.put(item)
        if preserve_set is None:
            cleared_pending = pending_before
            with self._pending_lock:
                self._pending_tasks.clear()
        else:
            with self._pending_lock:
                pending_after = len(self._pending_tasks)
            cleared_pending = max(0, pending_before - pending_after)
        _plog(
            f"flush queue: removed={flushed}, preserved_plan={preserved_plan}, "
            f"preserved_metadata={preserved_metadata}, restored={restores}, "
            f"cleared_pending={cleared_pending}"
        )

    def _initialize_analysis_state(self):
        self._photo_by_path = {p.path: p for p in self.catalog.photos}
        valid_paths = set(self._photo_by_path.keys())
        self._analysis_store.prune(valid_paths)
        pending: Set[str] = set()
        for path, photo in self._photo_by_path.items():
            payload = self._analysis_store.get(path)
            if payload and self._analysis_entry_matches(photo, payload):
                self._apply_analysis_payload(photo, payload)
            else:
                pending.add(path)
        self._analysis_pending_paths = pending
        self._update_filter_counts()

    def _analysis_entry_matches(self, photo: Photo, payload: Dict) -> bool:
        if not isinstance(payload, dict):
            return False
        version = payload.get('analysis_version', payload.get('version'))
        if version != ANALYSIS_VERSION:
            return False
        if int(payload.get('filesize', -1)) != photo.filesize:
            return False
        if int(payload.get('mtime_ns', -1)) != photo.file_mtime_ns:
            return False
        stored_ts = payload.get('timestamp')
        if stored_ts:
            try:
                stored_dt = datetime.fromisoformat(stored_ts)
            except Exception:
                stored_dt = None
            if stored_dt is not None:
                delta = abs((stored_dt - photo.timestamp).total_seconds())
                if delta > 1.0:
                    return False
        return True

    def _apply_analysis_payload(self, photo: Photo, payload: Dict):
        photo.analysis_version = int(payload.get('analysis_version', payload.get('version', 0)) or 0)
        photo.analysis_at = float(payload.get('analysis_at') or 0.0)
        photo.analysis_status = payload.get('status') or 'ok'
        photo.analysis_error = payload.get('error_message') or ''
        photo.faces_count = int(payload.get('faces_count') or 0)
        photo.faces_with_eyes = int(payload.get('faces_with_eyes') or 0)
        photo.eyes_open_count = int(payload.get('eyes_open_count') or 0)
        bucket = payload.get('eyes_open_bucket') or '100'
        ratio = float(payload.get('eyes_open_ratio') or 0.0)
        if photo.analysis_status == 'error':
            bucket = 'error'
            ratio = 0.0
        else:
            bucket, ratio = classify_eye_bucket(
                photo.eyes_open_count,
                photo.faces_count,
                photo.faces_with_eyes,
            )
        photo.eyes_open_bucket = bucket
        photo.eyes_open_ratio = ratio
        photo.sharpness_score = float(payload.get('sharpness_score') or 0.0)
        if photo.analysis_status == 'error':
            photo.sharpness_bucket = payload.get('sharpness_bucket') or 'unknown'
        else:
            photo.sharpness_bucket = payload.get('sharpness_bucket') or classify_sharpness_bucket(photo.sharpness_score)

    def _analysis_ready(self) -> bool:
        return not self._analysis_pending_paths

    def _update_filter_counts(self):
        eye_counts = {key: 0 for key, _ in OPEN_EYE_BUCKETS}
        sharp_counts = {key: 0 for key, _ in SHARPNESS_BUCKETS}
        unprocessed = 0
        for photo in self.catalog.photos:
            if photo.analysis_version != ANALYSIS_VERSION or photo.analysis_at <= 0:
                unprocessed += 1
                continue
            eye_counts[photo.eyes_open_bucket] = eye_counts.get(photo.eyes_open_bucket, 0) + 1
            sharp_counts[photo.sharpness_bucket] = sharp_counts.get(photo.sharpness_bucket, 0) + 1
        self._filter_counts = {
            'eyes': eye_counts,
            'sharpness': sharp_counts,
            'unprocessed': unprocessed,
        }
        self._refresh_filter_dialog()

    def _refresh_filter_dialog(self):
        if not self._filter_dialog:
            return
        self._filter_dialog.set_state(
            self.filter_state,
            self._filter_counts,
            analysis_ready=self._analysis_ready(),
            pending_count=len(self._analysis_pending_paths),
            processing=self._analysis_thread is not None,
            analysis_supported=self._analysis_supported,
        )

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
        setattr(_wrap, "_srp_task_key", key)
        setattr(_wrap, "_srp_task_origin", fn)
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
            <div style=\"margin:0; padding:0;\">
              <div style=\"margin:0; padding:0; font-weight:600; color:{theme_color('text.primary')};\">{_h(main_line)}
                <span style=\"color:{theme_color('text.secondary')};\">&nbsp;&nbsp;|&nbsp;&nbsp;{_h(dt_line)}</span>
              </div>
              <div style=\"margin:0; padding:0; font-size:9pt; color:{theme_color('text.secondary')};\">{_h(cam_line)}
                <span style=\"color:{theme_color('text.tertiary')};\">&nbsp;&nbsp;|&nbsp;&nbsp;{_h(lens_line)}</span>
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
        parent_widget = self.window()
        if not isinstance(parent_widget, QWidget):
            parent_widget = self
        dialog = HotkeyDialog(parent_widget, self.settings.hotkeys)
        dialog.exec()

    def _current(self) -> Optional[Photo]:
        if not self.catalog.photos: return None
        return self.catalog.photos[self.idx]

    def next_photo(self):
        indices = self._active_indices()
        if not indices:
            return
        pos = self._current_position(indices)
        if pos < 0:
            self.idx = indices[0]
            self._show_current(); self._heavy_load_scheduler.start()
            return
        if pos + 1 < len(indices):
            self.idx = indices[pos + 1]
            self._show_current(); self._heavy_load_scheduler.start()

    def prev_photo(self):
        indices = self._active_indices()
        if not indices:
            return
        pos = self._current_position(indices)
        if pos < 0:
            self.idx = indices[0]
            self._show_current(); self._heavy_load_scheduler.start()
            return
        if pos > 0:
            self.idx = indices[pos - 1]
            self._show_current(); self._heavy_load_scheduler.start()

    def _load_xmp_if_needed(self, photo: Photo):
        """Schedule a read from disk if the photo's XMP data is not loaded yet."""
        with photo.lock:
            if photo.xmp_loaded:
                return
            photo.xmp_loaded = True
        self._enqueue_xmp(photo.path, priority=-95)

    def toggle_select(self):
        p = self._current()
        if not p: return

        self._load_xmp_if_needed(p)
        current_index = self.idx
        p.update_xmp({'selected': not p.selected})

        self._update_selected_badge_fast()
        self._update_view_after_selection_change(current_index)
        self._show_current()
        self.autosave_timer.start(1500)

    def unselect_current(self):
        p = self._current()
        if not p: return
        if p.selected:
            self._load_xmp_if_needed(p)
            current_index = self.idx
            p.update_xmp({'selected': False})

            self._update_selected_badge_fast()
            self._update_view_after_selection_change(current_index)
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

    def show_filter_dialog(self):
        if self._filter_dialog is None:
            self._filter_dialog = FilterDialog(self)
            self._filter_dialog.filter_changed.connect(self._on_filter_changed)
            self._filter_dialog.process_requested.connect(self._on_filter_process_requested)
        self._refresh_filter_dialog()
        self._filter_dialog.show()
        self._filter_dialog.raise_()
        self._filter_dialog.activateWindow()

    def _on_filter_changed(self, state: FilterState):
        self.filter_state = state.copy()
        self._update_filter_badge()
        self._update_view_after_selection_change()
        self._show_current()
        self._heavy_load_scheduler.start()
        self._update_filmstrip()
        self._refresh_statusbar()
        self._refresh_filter_dialog()

    def _on_filter_process_requested(self):
        self._start_analysis_processing()

    def _pending_analysis_photos(self) -> List[Photo]:
        return [self._photo_by_path[path] for path in self._analysis_pending_paths if path in self._photo_by_path]

    def _start_analysis_processing(self, *, force: bool = False):
        if self._analysis_thread is not None:
            if self._analysis_progress_dialog:
                self._analysis_progress_dialog.show()
                self._analysis_progress_dialog.raise_()
            return

        if not self._analysis_supported:
            self._show_toast(
                "Analysis requires OpenCV and MediaPipe. Install opencv-python-headless and mediapipe to enable this feature."
            )
            return

        targets = list(self._photo_by_path.values()) if force else self._pending_analysis_photos()
        if not targets:
            self._show_toast("Analysis is already up to date")
            return

        self._analysis_worker = AnalysisWorker(targets)
        self._analysis_thread = QThread(self)
        self._analysis_worker.moveToThread(self._analysis_thread)
        self._analysis_thread.started.connect(self._analysis_worker.run)
        self._analysis_worker.progress.connect(self._on_analysis_progress)
        self._analysis_worker.finished.connect(self._on_analysis_finished)
        self._analysis_worker.error.connect(self._on_analysis_error)

        self._analysis_progress_dialog = QProgressDialog("Processing analysis…", "Cancel", 0, len(targets), self)
        self._analysis_progress_dialog.setWindowModality(Qt.WindowModal)
        self._analysis_progress_dialog.setMinimumDuration(0)
        self._analysis_progress_dialog.setValue(0)
        self._analysis_progress_dialog.canceled.connect(self._cancel_analysis_processing)
        self._analysis_start_time = time.perf_counter()

        self._analysis_thread.start()
        self._update_filter_badge()
        self._refresh_filter_dialog()

    def _cancel_analysis_processing(self):
        if self._analysis_worker:
            self._analysis_worker.request_cancel()

    @Slot(int, int, float, str)
    def _on_analysis_progress(self, completed: int, total: int, elapsed: float, name: str):
        if not self._analysis_progress_dialog:
            return
        dlg = self._analysis_progress_dialog
        dlg.setMaximum(max(total, 1))
        dlg.setValue(min(completed, total))
        if completed <= 0 or total <= 0:
            dlg.setLabelText("Processing analysis…")
            return
        avg = elapsed / max(completed, 1)
        self._analysis_average_seconds = (self._analysis_average_seconds * 0.6) + (avg * 0.4)
        remaining = max(0, total - completed)
        eta = int(round(self._analysis_average_seconds * remaining))
        label = f"Processing analysis… ({completed}/{total})"
        if eta > 0:
            label += f"  ~{eta}s remaining"
        dlg.setLabelText(label)

    @Slot(dict, bool)
    def _on_analysis_finished(self, results: Dict[str, Dict], canceled: bool):
        if self._analysis_progress_dialog:
            try:
                self._analysis_progress_dialog.reset()
            except Exception:
                pass
            self._analysis_progress_dialog = None

        self._finalize_analysis_thread()

        if results:
            self._apply_analysis_results(results)
            self._show_toast(f"Analysis completed for {len(results)} photo(s)")
        elif canceled:
            self._show_toast("Analysis canceled")
        else:
            self._show_toast("Analysis finished")

    @Slot(str, str)
    def _on_analysis_error(self, message: str, tb: str):
        print("Analysis error:", message)
        if tb:
            print(tb)
        self._show_toast("Failed to analyze some photos")

    def _finalize_analysis_thread(self):
        worker = self._analysis_worker
        thread = self._analysis_thread
        self._analysis_worker = None
        self._analysis_thread = None
        if worker:
            try:
                worker.deleteLater()
            except Exception:
                pass
        if thread:
            try:
                thread.quit()
                thread.wait()
                thread.deleteLater()
            except Exception:
                pass
        self._refresh_filter_dialog()
        self._update_filter_badge()

    def _apply_analysis_results(self, results: Dict[str, Dict]):
        if not results:
            return
        updated: Dict[str, Dict] = {}
        for path, payload in results.items():
            photo = self._photo_by_path.get(path)
            if not photo:
                continue
            self._apply_analysis_payload(photo, payload)
            self._analysis_pending_paths.discard(path)
            updated[path] = payload
        if updated:
            self._analysis_store.update_many(updated)
        self._update_filter_counts()
        self._update_filter_badge()
        self._update_view_after_selection_change()
        self._show_current()
        self._update_filmstrip()
        self._refresh_statusbar()

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

    def set_export_enabled(self, enabled: bool):
        action = self.actions.get('export')
        if action:
            action.setEnabled(enabled)

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
        indices = self._active_indices()
        if not indices:
            self.filmstrip.set_items([])
            return
        if self.idx not in indices:
            self.idx = indices[0]
        current_pos = self._current_position(indices)
        if current_pos < 0:
            current_pos = 0
        if not self.catalog.photos:
            self.filmstrip.set_items([])
            return
        items = []
        start_back = max(0, current_pos - k_backward)
        for pos in range(start_back, current_pos):
            idx = indices[pos]
            ph = self.catalog.photos[idx]
            items.append({'path': ph.path, 'selected': ph.selected, 'current': False, 'rating': ph.rating, 'color_label': ph.color_label})

        phc = self.catalog.photos[self.idx]
        items.append({'path': phc.path, 'selected': phc.selected, 'current': True, 'rating': phc.rating, 'color_label': phc.color_label})

        end_forward = min(len(indices), current_pos + 1 + k_forward)
        for pos in range(current_pos + 1, end_forward):
            idx = indices[pos]
            ph = self.catalog.photos[idx]
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

        photo_index = self.catalog.photos.index(photo)
        self._update_view_after_selection_change(photo_index)

        cur = self._current()
        if cur and cur.path == path:
            self.view.set_selected(current_selected)

        self._update_filmstrip()
        self._update_selected_badge_fast()

        if selected_changed:
            self.autosave_timer.start(1500)
        
    def _refresh_statusbar(self):
        if self.status_restore_timer.isActive(): return
        p = self._current()
        if not p: return
        indices = self._active_indices()
        total = len(indices) if indices else len(self.catalog.photos)
        pos = self._current_position(indices)
        current_num = (pos + 1) if pos >= 0 else (self.idx + 1)
        total_sel = sum(1 for x in self.catalog.photos if x.selected)
        self.badge_selected.setText(f"Selected: {total_sel}")
        scope_parts: List[str] = []
        if self.filter_state.selected_only:
            scope_parts.append("sel")
        if self.filter_state.eye_buckets:
            scope_parts.append("eyes")
        if self.filter_state.sharpness_buckets:
            scope_parts.append("sharp")
        view_scope = ",".join(scope_parts) if scope_parts else "all"
        msg = f"[{current_num}/{total} {view_scope}]  Selected: {total_sel}  {os.path.basename(p.path)}  |  workers: {self._num_workers}"
        self.status_message.emit(msg, 0)

    def _show_current(self):
        self._update_view_after_selection_change(self.idx)
        indices = self._active_indices()
        if not indices:
            self.meta_left.setText("")
            if self.filter_state.is_active():
                self.view.setText("No photos match the active filters")
            else:
                self.view.setText("No images")
            self.filmstrip.set_items([])
            self._refresh_statusbar()
            return
        if self.idx not in indices:
            self.idx = indices[0]
        p = self._current()
        if not p:
            self.meta_left.setText("")
            self.view.setText("No images")
            self.filmstrip.set_items([])
            self._refresh_statusbar()
            return

        self._load_xmp_if_needed(p)

        pos = self._current_position(indices)
        if pos < 0:
            pos = indices.index(self.idx) if self.idx in indices else 0
        start = max(0, pos - 5)
        end = min(len(indices), pos + 6)
        for i in range(start, end):
            photo_index = indices[i]
            visiblePhoto = self.catalog.photos[photo_index]
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
        self._schedule_loading_plan_fire()

    def _apply_task_plan(self, plan: List[Tuple[int, Tuple, Callable[..., None], Tuple]]):
        dedup: Dict[Tuple, Tuple[int, Callable[..., None], Tuple]] = {}
        for priority, key, fn, args in plan:
            prev = dedup.get(key)
            if prev is None or priority < prev[0]:
                dedup[key] = (priority, fn, args)
        preserve_keys = set(dedup.keys())
        self._flush_queue(preserve_keys=preserve_keys)
        ordered = sorted(dedup.items(), key=lambda item: item[1][0])
        for key, (priority, fn, args) in ordered:
            self._enqueue(priority, key, fn, *args)

    def _schedule_loading_plan_fire(self):
        cur = self._current()
        if not cur:
            self._flush_queue(preserve_keys=set())
            return
        cur_path = cur.path; total = len(self.catalog.photos)

        area = self.view.contentsRect(); dpr = self.view.devicePixelRatioF()
        target_size = QSize(int(area.width() * dpr), int(area.height() * dpr))

        plan: List[Tuple[int, Tuple, Callable[..., None], Tuple]] = []

        def plan_load(path: str, kind: str, priority: int):
            plan.append((priority, (path, kind), self._worker_entry, (path, kind)))

        def plan_xmp(path: str, priority: int):
            plan.append((priority, (path, 'xmp'), self._worker_entry, (path, 'xmp')))

        def plan_resized(path: str, size: QSize, priority: int):
            plan.append(
                (
                    priority,
                    (path, 'resized_pixmap', size.width(), size.height()),
                    self._worker_build_resized_pixmap,
                    (path, size),
                )
            )

        if self._get_pil_half_cached(cur_path) is None:
            plan_load(cur_path, 'half', -100)

        plan_xmp(cur_path, -95)

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
            plan_xmp(pth, HALF_BASE + d * 2)
            if self._get_pil_half_cached(pth) is not None:
                plan_resized(pth, target_size, HALF_BASE + d * 2 + 1)
            else:
                plan_load(pth, 'half', HALF_BASE + d * 2)

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
                plan_load(pth, 'full', FULL_BASE + d)

        self._apply_task_plan(plan)

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
        total = len(self.catalog.photos)
        if total == 0:
            return

        if os.path.exists(self.selections_path):
            try:
                with open(self.selections_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                selected_set = set(data.get('selected_paths', []))
                for p in self.catalog.photos:
                    p.selected = (p.path in selected_set)
            except Exception:
                pass
            finally:
                self._update_selected_badge_fast()
            return

        if EXIFTOOL_PATH is None:
            self._update_selected_badge_fast()
            return

        app = QApplication.instance()
        progress = None
        if app:
            parent = app.activeWindow()
            progress = QProgressDialog("Reading XMP sidecars…", "", 0, total, parent)
            progress.setCancelButton(None)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            progress.show()
            QApplication.processEvents()

        selected_paths: List[str] = []
        for idx, photo in enumerate(self.catalog.photos, start=1):
            data = read_xmp_sidecar(photo.path)
            rating_val = data.get('rating') if data else None
            label_val = data.get('color_label') if data else None
            selected_val = data.get('selected') if data else None

            with photo.lock:
                if data:
                    photo.xmp_loaded = True
                if rating_val is not None:
                    photo.rating = rating_val
                if label_val is not None:
                    photo.color_label = label_val
                if selected_val is not None:
                    photo.selected = selected_val

            if photo.selected:
                selected_paths.append(photo.path)

            if progress:
                progress.setValue(idx)
                QApplication.processEvents()

        if progress:
            progress.close()

        self._update_selected_badge_fast()

        data = {
            'root': self.catalog.root,
            'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'selected_paths': selected_paths,
        }
        try:
            with open(self.selections_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error creating selections.json: {e}")

    def save_all_dirty_files(self, wait: bool = False):
        """Save all in-memory changes (selections.json and XMP) to disk."""
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

        wait_events: List[threading.Event] = []

        def _write_task_with_cleanup(path, data, photo_obj, version, done_event: Optional[threading.Event]):
            success = False
            try:
                success = bool(write_xmp_sidecar(path, data))
            except Exception as e:
                print(f"Unexpected error saving XMP for {os.path.basename(path)}: {e}")
                success = False
            finally:
                signal_to_emit = None
                try:
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
                finally:
                    if done_event is not None:
                        done_event.set()
                if signal_to_emit == 'saved':
                    self.signals.xmp_saved.emit(path)
                elif signal_to_emit == 'failed':
                    self.signals.xmp_save_failed.emit(path)

        setattr(_write_task_with_cleanup, "_srp_metadata_save", True)

        for photo, payload, version in tasks:
            done_event = threading.Event() if wait else None
            if done_event is not None:
                wait_events.append(done_event)
            self._post_task(20, _write_task_with_cleanup, photo.path, payload, photo, version, done_event)

        self._show_temporary_status(f"Auto-saved metadata for {len(tasks)} photos.", 2000)

        if wait and wait_events:
            for ev in wait_events:
                ev.wait()

    def cleanup(self):
        try:
            self.autosave_timer.stop()
            self.autosave_interval_timer.stop()
        except Exception:
            pass

        try:
            self._cancel_analysis_processing()
        except Exception:
            pass
        if self._analysis_progress_dialog:
            try:
                self._analysis_progress_dialog.reset()
            except Exception:
                pass
            self._analysis_progress_dialog = None
        self._finalize_analysis_thread()

        self.save_all_dirty_files()
        self._flush_queue()

        try:
            self._taskq.join()
        except Exception:
            pass

        self._loader_stop = True
        try:
            for t in self._loader_threads:
                t.join(timeout=0.5)
        except Exception:
            pass
        QApplication.instance().removeEventFilter(self)

              
class KeySequenceEdit(QLineEdit):
    _active_capture_widget: Optional['KeySequenceEdit'] = None
    _collected_keys: List[str] = []  # Store multiple keys

    def __init__(self, key_sequence_str: str, parent=None):
        super().__init__(key_sequence_str, parent)
        self.setPlaceholderText("Click to set a new shortcut")
        self._is_capturing = False

    @classmethod
    def active_capture_widget(cls) -> Optional['KeySequenceEdit']:
        widget = cls._active_capture_widget
        if widget is not None and widget.is_hotkey_capture_active():
            return widget
        return None

    def is_hotkey_capture_active(self) -> bool:
        return self._is_capturing

    def _enter_capture_mode(self):
        if self._is_capturing:
            return
        self.setText("")
        self.setPlaceholderText("Press keys, then Enter to confirm or Escape to cancel...")
        self._is_capturing = True
        self._collected_keys = []  # Reset collected keys
        KeySequenceEdit._active_capture_widget = self
        try:
            self.grabKeyboard()
        except Exception:
            pass

    def _exit_capture_mode(self):
        if not self._is_capturing:
            return
        self._is_capturing = False
        self._collected_keys = []  # Clear collected keys
        if KeySequenceEdit._active_capture_widget is self:
            KeySequenceEdit._active_capture_widget = None
        try:
            self.releaseKeyboard()
        except Exception:
            pass
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

        # Exit on Escape (cancel) or Enter (confirm)
        if key == Qt.Key_Escape:
            self.setText("")  # Clear on cancel
            self._exit_capture_mode()
            return
        if key == Qt.Key_Return or key == Qt.Key_Enter:
            self._exit_capture_mode()
            return

        key_sequence = QKeySequence(event.keyCombination())
        text = key_sequence.toString(QKeySequence.NativeText)

        if not text:
            # Some platforms fail to produce a key combination string for
            # plain number keys (e.g. keypad digits). Try again with just the
            # key code and finally fall back to the text representation so
            # that numeric shortcuts can be captured reliably.
            fallback_sequence = QKeySequence(event.key())
            text = fallback_sequence.toString(QKeySequence.NativeText)

        if not text:
            text = event.text() or ""

        if not text:
            return

        # Add to collected keys and update display
        if text not in self._collected_keys:  # Avoid duplicates
            self._collected_keys.append(text)
            self.setText(", ".join(self._collected_keys))
        
        # Don't exit capture mode - keep collecting keys
        # User must press Enter/Escape or click outside to finish
        event.accept()

    def keyReleaseEvent(self, event: QEvent):
        if self._is_capturing:
            event.accept()
class SettingsDialog(QDialog):
    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.settings = settings
        self.hotkey_edits: Dict[str, 'KeySequenceEdit'] = {}
        self._parent_window = parent  # Store parent for accessing toast method

        main_layout = QVBoxLayout(self)
        
        general_group = QGroupBox("General")
        form_layout = QFormLayout()
        self.autosave_spinbox = QSpinBox()
        self.autosave_spinbox.setRange(1, 60)
        self.autosave_spinbox.setValue(self.settings.autosave_interval_min)
        self.autosave_spinbox.setSuffix(" minute(s)")
        form_layout.addRow("Autosave Interval:", self.autosave_spinbox)
        
        self.raw_output_folder_edit = QLineEdit(self.settings.raw_output_folder_name)
        form_layout.addRow("Raw Output Folder Name:", self.raw_output_folder_edit)

        self.jpeg_output_folder_edit = QLineEdit(self.settings.jpeg_output_folder_name)
        form_layout.addRow("JPEG Output Folder Name:", self.jpeg_output_folder_edit)
        
        general_group.setLayout(form_layout)
        main_layout.addWidget(general_group)

        hotkey_group = QGroupBox("Hotkeys")
        hotkey_layout = QGridLayout()
        hotkey_layout.setHorizontalSpacing(12)
        hotkey_layout.setVerticalSpacing(8)

        hotkey_labels = {
            'next': 'Next Image:',
            'prev': 'Previous Image:',
            'toggle_select': 'Toggle Select:',
            'unselect': 'Unselect Image:',
            'toggle_zebra': 'Toggle Zebra/Histogram:',
            'toggle_hdr': 'Toggle Faux HDR Preview:',
            'show_filter_dialog': 'Open Filter Panel:',
            'rate_1': '1★ Rating:',
            'rate_2': '2★ Rating:',
            'rate_3': '3★ Rating:',
            'rate_4': '4★ Rating:',
            'rate_5': '5★ Rating:',
            'label_red': 'Label Red:',
            'label_yellow': 'Label Yellow:',
            'label_green': 'Label Green:',
            'label_blue': 'Label Blue:',
            'label_purple': 'Label Purple:',
            # 'save': 'Save Selections:',
            'export': 'Save Progress:',
            'help': 'Show Help:',
            'quit': 'Quit Application:'
        }

        hotkey_entries: List[Tuple[str, QLabel, KeySequenceEdit]] = []
        for action, default_value in DEFAULT_HOTKEYS.items():
            key_sequence_str = self.settings.hotkeys.get(action, default_value)
            fallback_label = hotkey_labels.get(action, f"{action.replace('_', ' ').title()}:")
            label_text, is_rich = _styled_hotkey_label(action, fallback_label)
            label = QLabel()
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            accessible_label = fallback_label.rstrip(':')
            label.setAccessibleName(accessible_label)
            label.setToolTip(accessible_label)
            if is_rich:
                label.setTextFormat(Qt.RichText)
            label.setText(label_text)
            edit = KeySequenceEdit(key_sequence_str)
            hotkey_entries.append((action, label, edit))

        for action in self.settings.hotkeys.keys():
            if action in DEFAULT_HOTKEYS:
                continue
            key_sequence_str = self.settings.hotkeys.get(action, '')
            fallback_label = f"{action.replace('_', ' ').title()}:"
            label_text, is_rich = _styled_hotkey_label(action, fallback_label)
            label = QLabel()
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            accessible_label = fallback_label.rstrip(':')
            label.setAccessibleName(accessible_label)
            label.setToolTip(accessible_label)
            if is_rich:
                label.setTextFormat(Qt.RichText)
            label.setText(label_text)
            edit = KeySequenceEdit(key_sequence_str)
            hotkey_entries.append((action, label, edit))

        columns = 2
        rows_per_column = max(1, (len(hotkey_entries) + columns - 1) // columns)
        for index, (action, label, edit) in enumerate(hotkey_entries):
            column = index // rows_per_column
            row = index % rows_per_column
            base_col = column * 2
            hotkey_layout.addWidget(label, row, base_col)
            hotkey_layout.addWidget(edit, row, base_col + 1)
            self.hotkey_edits[action] = edit
            
            # Connect to check for conflicts when editing finishes
            edit.editingFinished.connect(
                lambda action=action: self._check_hotkey_conflicts(action)
            )
            edit.textChanged.connect(lambda text, action=action: self._on_hotkey_changed(action, text))

        for column in range(columns):
            hotkey_layout.setColumnStretch(column * 2, 0)
            hotkey_layout.setColumnStretch(column * 2 + 1, 1)

        hotkey_group.setLayout(hotkey_layout)
        main_layout.addWidget(hotkey_group)


        restore_button = QPushButton("Restore Default Hotkeys")
        restore_button.setObjectName("SecondaryButton")
        restore_button.setStyleSheet(f"""
            QPushButton#SecondaryButton {{
                background-color: {theme_color('bg.surface')};
                color: {theme_color('text.secondary')};
                border: 1px solid {theme_color('border.default')};
                border-radius: 4px;
                padding: 8px 16px;
            }}
            QPushButton#SecondaryButton:hover {{
                background-color: {theme_color('bg.elevated')};
            }}
            QPushButton#SecondaryButton:pressed {{
                background-color: {theme_color('bg.base')};
            }}
        """)
        restore_button.clicked.connect(self.restore_defaults)
        main_layout.addWidget(restore_button)
        
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        self.button_box.accepted.connect(self.accept)
        main_layout.addWidget(self.button_box)

    def _on_hotkey_changed(self, action: str, text: str):
        """Real-time conflict checking as user types"""
        if not text:
            # Clear any warning styling when field is empty
            edit = self.hotkey_edits.get(action)
            if edit:
                edit.setStyleSheet("")
                edit.setToolTip("")
            return
            
        # Check if this shortcut conflicts with other actions
        conflicting_action = self._find_conflicting_action(action, text)
        if conflicting_action:
            # Show conflict warning but don't clear the field yet
            # Let the user decide what to do
            edit = self.hotkey_edits.get(action)
            if edit:
                edit.setStyleSheet("QLineEdit { background-color: #5c3c3c; border: 1px solid #ff5f5f; }")  # Red background
                conflict_label = conflicting_action.replace('_', ' ').title()
                edit.setToolTip(f"Conflicts with '{conflict_label}'")
        else:
            # Clear any warning styling if no conflict
            edit = self.hotkey_edits.get(action)
            if edit:
                edit.setStyleSheet("")
                edit.setToolTip("")

    def _check_hotkey_conflicts(self, current_action: str):
        """Check for conflicts when editing is finished"""
        current_edit = self.hotkey_edits.get(current_action)
        if not current_edit:
            return
            
        current_keys = current_edit.text().strip()
        if not current_keys:
            # Clear styling when empty
            current_edit.setStyleSheet("")
            current_edit.setToolTip("")
            return
            
        # Find conflicts
        conflicting_action = self._find_conflicting_action(current_action, current_keys)
        if conflicting_action:
            # Show toast message about conflict
            self._show_conflict_toast(current_action, conflicting_action, current_keys)
            # Clear the conflicting shortcut
            current_edit.setText("")
            current_edit.setStyleSheet("")
            current_edit.setToolTip("")

    def _find_conflicting_action(self, current_action: str, keys: str) -> Optional[str]:
        """Find if the given keys conflict with any other action"""
        # Split the keys by comma and normalize
        current_key_list = [k.strip() for k in keys.split(',') if k.strip()]
        
        for action, edit in self.hotkey_edits.items():
            if action == current_action:
                continue
                
            other_keys = edit.text().strip()
            if not other_keys:
                continue
                
            # Check for any overlap between key lists
            other_key_list = [k.strip() for k in other_keys.split(',') if k.strip()]
            
            # Check if any key appears in both lists
            for key in current_key_list:
                if key in other_key_list:
                    return action
                    
        return None

    def _show_conflict_toast(self, action1: str, action2: str, keys: str):
        """Show a toast message about shortcut conflict"""
        action2_label = action2.replace('_', ' ').title()
        msg = f"Shortcut '{keys}' conflicts with '{action2_label}'"
        # Try to use parent's toast method if available
        if hasattr(self._parent_window, '_show_toast'):
            self._parent_window._show_toast(msg, 3000)
        else:
            # Fallback to message box if no toast available
            QMessageBox.warning(self, "Shortcut Conflict", msg)

    def restore_defaults(self):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Restore Defaults")
        msg.setText("Are you sure you want to restore all hotkeys to their default values?")

        restore_btn = msg.addButton("Restore", QMessageBox.AcceptRole)
        cancel_btn  = msg.addButton("Cancel",  QMessageBox.RejectRole)

        # Cancel을 덜 강조
        cancel_btn.setObjectName("GhostButton")
        cancel_btn.setAutoDefault(False)
        cancel_btn.setDefault(False)
        cancel_btn.setStyleSheet(f"""
            QPushButton#GhostButton {{
                background-color: {theme_color('bg.surface')};
                color: {theme_color('text.secondary')};
                border: 1px solid {theme_color('border.default')};
                border-radius: 4px;
                padding: 6px 12px;
            }}
            QPushButton#GhostButton:hover {{
                background-color: {theme_color('bg.elevated')};
            }}
            QPushButton#GhostButton:pressed {{
                background-color: {theme_color('bg.base')};
            }}
        """)

        # Restore를 기본값으로
        restore_btn.setDefault(True)
        restore_btn.setAutoDefault(True)

        msg.exec()

        if msg.clickedButton() is restore_btn:
            for action, default_value in DEFAULT_HOTKEYS.items():
                if action in self.hotkey_edits:
                    self.hotkey_edits[action].setText(default_value)
                    
    def accept(self):
        self.settings.autosave_interval_min = self.autosave_spinbox.value()
        self.settings.raw_output_folder_name = self.raw_output_folder_edit.text() or '_selected_raw'
        self.settings.jpeg_output_folder_name = self.jpeg_output_folder_edit.text() or '_selected_jpeg'
        
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
        sel_lbl.setTextFormat(Qt.RichText)
        form.addRow("Selected:", sel_lbl)

        raw_row = QLabel(f"<b>{dest_raw_count}</b> files  |  <i>{_h(raw_path)}</i>")
        raw_row.setTextFormat(Qt.RichText)
        raw_row.setWordWrap(True)
        form.addRow("RAW:", raw_row)

        if dest_jpeg_count > 0:
            jpg_row = QLabel(f"<b>{dest_jpeg_count}</b> files  |  <i>{_h(jpeg_path)}</i>")
            jpg_row.setTextFormat(Qt.RichText)
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


class AboutDialog(QDialog):
    def __init__(self, parent: QWidget, app_data_path: str):
        super().__init__(parent)
        self.setWindowTitle("About simple raw picker")
        self.setModal(True)
        self.setMinimumWidth(460)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(18)

        app = QApplication.instance()
        font_family = getattr(app, "_custom_font_family", "Arial") if app else "Arial"

        title = QLabel(self)
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont(font_family, 18)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setText(
            f"simple <span style=\"color:{theme_color('accent.primary')}\">raw</span> picker"
        )
        title.setTextFormat(Qt.RichText)
        layout.addWidget(title)

        tagline = QLabel("fast, simple, free, RAW.", self)
        tagline.setAlignment(Qt.AlignCenter)
        tagline_font = QFont(font_family, 11)
        tagline.setFont(tagline_font)
        tagline.setStyleSheet(f"color:{theme_color('text.secondary')};")
        layout.addWidget(tagline)

        highlight_card = QFrame(self)
        highlight_card.setObjectName("HighlightCard")
        highlight_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        highlight_card.setStyleSheet(
            f"#HighlightCard {{"
            f"    background-color: {theme_color('bg.surface')};"
            f"    border: 1px solid {theme_color('border.default')};"
            "    border-radius: 12px;"
            "}"
        )
        highlight_layout = QVBoxLayout(highlight_card)
        highlight_layout.setContentsMargins(18, 16, 18, 18)
        highlight_layout.setSpacing(10)

        highlight_title = QLabel("Contacts", highlight_card)
        highlight_title.setAlignment(Qt.AlignLeft)
        highlight_title.setStyleSheet(
            f"color:{theme_color('accent.primary')}; font-size:11pt; font-weight:600; letter-spacing:0.4px; background-color: transparent;"
        )
        highlight_layout.addWidget(highlight_title)
        
        highlight_layout.setSpacing(10)

        email_label = QLabel(
            '<a href="mailto:superhoy0509@gmail.com">superhoy0509@gmail.com</a>',
            highlight_card,
        )
        email_label.setTextFormat(Qt.RichText)
        email_label.setOpenExternalLinks(True)
        email_label.setStyleSheet(
            f"color:{theme_color('text.primary')}; font-size:10pt; background-color: transparent;"
        )
        highlight_layout.addWidget(email_label)
        
        support = QLabel(
            '<a href="http://donate.recu3125.com">Support the developer</a>',
            highlight_card,
        )
        support.setAlignment(Qt.AlignLeft)
        support.setTextFormat(Qt.RichText)
        support.setOpenExternalLinks(True)
        support.setStyleSheet(
            f"color:{theme_color('accent.primary')}; font-size:10pt; background-color: transparent;"
        )
        highlight_layout.addWidget(support)


        feedback_label = QLabel(
            '<a href="https://forms.gle/QopoQ8KCXJoYZHT39">Bug reports & feedback form</a>',
            highlight_card,
        )
        feedback_label.setTextFormat(Qt.RichText)
        feedback_label.setOpenExternalLinks(True)
        feedback_label.setStyleSheet(
            f"color:{theme_color('text.primary')}; font-size:10pt; background-color: transparent;"
        )
        highlight_layout.addWidget(feedback_label)

        layout.addWidget(highlight_card)

        layout.addSpacing(6)

        open_source_section = QFrame(self)
        open_source_section.setObjectName("OpenSourceCard")
        open_source_section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        os_layout = QVBoxLayout(open_source_section)
        os_layout.setContentsMargins(18, 12, 18, 16)
        os_layout.setSpacing(6)
        open_source_section.setStyleSheet(
            f"#OpenSourceCard {{"
            f"    background-color: {theme_color('bg.surface')};"
            f"    border: 1px solid {theme_color('border.default')};"
            "    border-radius: 10px;"
            "}"
        )
        header = QLabel("Open source components", open_source_section)
        header.setAlignment(Qt.AlignLeft)
        header.setStyleSheet(
            f"color:{theme_color('text.tertiary')}; font-size:9pt; font-weight:500; letter-spacing:0.6px; background-color: transparent;"
        )
        os_layout.addWidget(header)

        oss_entries = [
            ("PySide6 (Qt for Python)", "https://www.qt.io/qt-for-python"),
            ("rawpy", "https://github.com/letmaik/rawpy"),
            ("Pillow", "https://python-pillow.org"),
            ("NumPy", "https://numpy.org"),
            ("psutil", "https://github.com/giampaolo/psutil"),
            ("ExifTool", "https://exiftool.org"),
        ]

        oss_links = ", ".join(
            [
                f"<a style=\"color:{theme_color('text.secondary')}; text-decoration:none;\" href=\"{url}\">{name}</a>"
                for name, url in oss_entries
            ]
        )

        oss_label = QLabel(oss_links, open_source_section)
        oss_label.setTextFormat(Qt.RichText)
        oss_label.setOpenExternalLinks(True)
        oss_label.setWordWrap(True)
        oss_label.setStyleSheet(
            f"color:{theme_color('text.secondary')}; font-size:9pt; background-color: transparent;"
        )
        os_layout.addWidget(oss_label)

        layout.addWidget(open_source_section)

        if app_data_path:
            path_label = QLabel(
                f"App data location: <span style=\"color:{theme_color('text.secondary')}\">{_h(app_data_path)}</span>",
                self,
            )
            path_label.setTextFormat(Qt.RichText)
            path_label.setWordWrap(True)
            path_label.setStyleSheet(
                f"color:{theme_color('text.tertiary')}; font-size:9pt; background-color: transparent;"
            )
            layout.addWidget(path_label)


        author_label = QLabel(
            'Made by <a href="https://recu3125.com/">recu3125</a>',
            self,
        )
        author_label.setAlignment(Qt.AlignLeft)
        author_label.setTextFormat(Qt.RichText)
        author_label.setOpenExternalLinks(True)
        author_label.setStyleSheet(
            f"color:{theme_color('text.secondary')}; font-size:10pt; background-color: transparent;"
        )
        layout.addWidget(author_label)

        layout.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

class WelcomeWidget(QWidget):
    folder_selected = Signal(str)

    GAP_TITLE_BTN    = 36
    GAP_CENTER_BOTTOM= 36
    CONTAINER_W      = 560
    FOOTER_MARGIN_TOP= 16

    def __init__(self, on_select_folder, recent_folders: List[str], parent=None):
        super().__init__(parent)
        self.on_select_folder = on_select_folder

        accent_primary = theme_color('accent.primary')
        accent_hover = theme_color('accent.hover')
        accent_active = theme_color('accent.active')
        text_on_accent = theme_color('text.on_accent')
        text_tertiary = theme_color('text.tertiary')
        border_default = theme_color('border.default')

        root_h = QHBoxLayout(self)
        root_h.setContentsMargins(0, 0, 0, 0); root_h.setSpacing(0); root_h.setAlignment(Qt.AlignCenter)

        col = QWidget(self); col.setFixedWidth(self.CONTAINER_W)
        root_h.addWidget(col, 0, Qt.AlignCenter)

        v = QVBoxLayout(col); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        self._column_layout = v

        self._sp_top = QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Fixed)
        v.addItem(self._sp_top)

        self.center_group = QWidget(col)
        cg = QVBoxLayout(self.center_group); cg.setContentsMargins(0, 0, 0, 0); cg.setSpacing(0); cg.setAlignment(Qt.AlignHCenter)
        app = QApplication.instance()
        font_family = getattr(app, "_custom_font_family", "Arial")

        self.title = QLabel(self.center_group)
        self.title.setText(
            f"simple <span style=\"color:{accent_primary}\">raw</span> picker"
        )
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
                background-color: {accent_primary};
                border: 1px solid {accent_hover};
                color: {text_on_accent};
                font-size: 12pt;
                font-weight: bold;
                padding: 12px 18px;
                border-radius: 10px;
            }}
            QPushButton#WelcomeButton:hover {{ background-color: {accent_hover}; }}
            QPushButton#WelcomeButton:pressed {{ background-color: {accent_active}; }}
        """)
        shadow = QGraphicsDropShadowEffect(self); shadow.setBlurRadius(24); shadow.setOffset(0, 4)
        shadow_color = theme_qcolor('accent.primary')
        shadow_color.setAlpha(110)
        shadow.setColor(shadow_color)
        self.select_btn.setGraphicsEffect(shadow)
        cg.addWidget(self.select_btn, 0, Qt.AlignHCenter)

        v.addWidget(self.center_group, 0, Qt.AlignHCenter)

        self._sp_middle = QSpacerItem(
            0, self.GAP_CENTER_BOTTOM, QSizePolicy.Minimum, QSizePolicy.Expanding
        )
        v.addItem(self._sp_middle)

        self.bottom_section = QWidget(col)
        bl = QVBoxLayout(self.bottom_section); bl.setContentsMargins(0, 0, 0, 20); bl.setSpacing(0)

        divider = QFrame(self.bottom_section)
        divider.setFrameShape(QFrame.HLine); divider.setFrameShadow(QFrame.Sunken)
        divider.setStyleSheet(
            f"background-color:{border_default}; color:{border_default};"
        ); divider.setFixedHeight(2)
        bl.addWidget(divider)

        bl.addSpacing(10)

        self.recent_header = QLabel("Recent Folders", self.bottom_section)
        self.recent_header.setAlignment(Qt.AlignLeft)
        hf = QFont(font_family, 10); hf.setLetterSpacing(QFont.PercentageSpacing, 102)
        self.recent_header.setFont(hf); self.recent_header.setStyleSheet(
            f"color:{text_tertiary};"
        )
        bl.addWidget(self.recent_header)
        bl.addSpacing(10)

        self.recent_container = QWidget(self.bottom_section)
        self.recent_layout = QVBoxLayout(self.recent_container); self.recent_layout.setContentsMargins(0, 0, 0, 0); self.recent_layout.setSpacing(6)
        bl.addWidget(self.recent_container, 0, Qt.AlignTop)

        support = QLabel('<a href="http://donate.recu3125.com">Support the developer</a>', self.bottom_section)
        support.setOpenExternalLinks(True); support.setAlignment(Qt.AlignCenter)
        support.setStyleSheet(
            f"QLabel {{ color:{accent_primary}; font-size:10pt; margin-top:{self.FOOTER_MARGIN_TOP}px; }}"
        )

        bl.addStretch(1)
        bl.addWidget(support, 0, Qt.AlignBottom)

        v.addWidget(self.bottom_section, 0)

        self.update_recent_folders(recent_folders)
        QTimer.singleShot(0, self._reflow)

    def _reflow(self):
        if self._column_layout is None or not self.isVisible():
            return

        total_h = max(0, self.height())
        if total_h == 0:
            return

        def _block_height(widget: QWidget) -> int:
            hint = widget.sizeHint().height()
            layout = widget.layout()
            if layout is not None:
                hint = max(hint, layout.sizeHint().height())
            return max(hint, widget.height(), widget.minimumSizeHint().height())

        gh = _block_height(self.center_group)
        bh = _block_height(self.bottom_section)
        gap_min = self.GAP_CENTER_BOTTOM if self.bottom_section.isVisible() else 0

        available = max(0, total_h - (gh + bh))

        if available == 0:
            top = middle = 0.0
        else:
            # Ideal centered placement before enforcing the minimum gap.
            top_center = max(0.0, (total_h - gh) / 2.0)
            gap_center = max(0.0, (total_h - gh) / 2.0 - bh)

            if gap_center >= gap_min:
                top = min(top_center, float(available))
                middle = max(0.0, available - top)
            else:
                if available <= gap_min:
                    top = 0.0
                    middle = float(available)
                else:
                    middle = float(gap_min)
                    top = float(available) - middle

        top_h = int(round(top))
        middle_h = int(round(middle))

        if top_h < 0:
            top_h = 0
        if middle_h < 0:
            middle_h = 0

        self._sp_top.changeSize(0, top_h, QSizePolicy.Minimum, QSizePolicy.Fixed if top_h else QSizePolicy.Minimum)
        self._sp_middle.changeSize(0, middle_h, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self._column_layout.invalidate()
        self._column_layout.activate()

    def resizeEvent(self, e):
        super().resizeEvent(e); self._reflow()

    def _make_recent_row(self, display_text: str, full_path: str) -> QWidget:
        row = QWidget(self.bottom_section)
        hl = QHBoxLayout(row); hl.setContentsMargins(0, 0, 0, 0); hl.setSpacing(8)

        dot = QLabel(row); dot.setFixedSize(8, 8)
        dot.setStyleSheet(
            f"background:{theme_color('text.tertiary')}; border-radius:4px;"
        )
        hl.addWidget(dot, 0, Qt.AlignVCenter)

        btn = QPushButton(display_text, row)
        btn.setObjectName("RecentButton"); btn.setFlat(True)
        btn.setMinimumHeight(30); btn.setCursor(Qt.PointingHandCursor)
        btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn.setToolTip(full_path); btn.setFocusPolicy(Qt.NoFocus)
        btn.clicked.connect(lambda _=False, p=full_path: self.folder_selected.emit(p))
        btn.setStyleSheet(f"""
            QPushButton#RecentButton {{
                background: {theme_color('bg.surface')};
                color: {theme_color('text.secondary')};
                border: 1px solid {theme_color('border.default')};
                padding: 6px 10px;
                border-radius: 8px;
                text-align: left;
            }}
            QPushButton#RecentButton:hover {{ background: {theme_color('bg.elevated')}; border-color: {theme_color('border.default')}; }}
            QPushButton#RecentButton:pressed {{ background: {theme_color('bg.base')}; }}
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
        self.has_seen_tutorial = False
        self.culling_widget: Optional[CullingWidget] = None
        self._export_thread: Optional[QThread] = None
        self._export_worker: Optional[ExportWorker] = None
        self._export_progress_dialog: Optional[QProgressDialog] = None
        self._export_in_progress = False
        self._export_finalizer: Optional[Callable[[ExportResult], Optional[ExportResult]]] = None
        self._export_finalizer_message: Optional[str] = None
        self._export_finalizing = False
        self._export_success_callback: Optional[Callable[[ExportResult], None]] = None
        self._export_error_callback: Optional[Callable[[str], None]] = None
        self._export_cancel_callback: Optional[Callable[[], None]] = None

        self.app_data_path = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        os.makedirs(self.app_data_path, exist_ok=True)
        self.app_state_file = os.path.join(self.app_data_path, "app_state.json")
        self.recent_folders = []
        self._load_app_state()

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.stack.currentChanged.connect(self._update_help_shortcut_enabled_state)
        self._help_shortcuts: List[QShortcut] = []

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
        self._refresh_help_shortcuts()
        QTimer.singleShot(200, self._show_help_if_first_time)

    def _load_app_state(self):
        try:
            with open(self.app_state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
                self.recent_folders = state.get("recent_folders", [])
                self.has_seen_tutorial = bool(state.get("has_seen_tutorial", False))
        except (FileNotFoundError, json.JSONDecodeError):
            self.recent_folders = []
            self.has_seen_tutorial = False

    def _save_app_state(self):
        try:
            state = {
                "recent_folders": self.recent_folders,
                "has_seen_tutorial": self.has_seen_tutorial,
            }
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

        self.act_about = QAction("About", self)
        self.act_about.triggered.connect(self.open_about)
        self.toolbar.addAction(self.act_about)

        self.act_complete = QAction("Complete", self)
        self.act_complete.triggered.connect(self.complete_culling)
        self.toolbar.addAction(self.act_complete)

    def _refresh_help_shortcuts(self):
        for shortcut in getattr(self, "_help_shortcuts", []):
            try:
                shortcut.activated.disconnect(self._handle_help_shortcut)
            except (TypeError, RuntimeError):
                pass
            shortcut.setEnabled(False)
            shortcut.deleteLater()
        self._help_shortcuts = []

        sequence_str = (self.settings.hotkeys.get('help') if self.settings else "") or ""
        sequences = [s.strip() for s in sequence_str.split(',') if s.strip()]
        for seq in sequences:
            shortcut = QShortcut(QKeySequence(seq), self)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.activated.connect(self._handle_help_shortcut)
            self._help_shortcuts.append(shortcut)

        self._update_help_shortcut_enabled_state()

    def _handle_help_shortcut(self):
        if self.stack.currentWidget() is self.welcome_screen:
            self.open_help()

    def _update_help_shortcut_enabled_state(self):
        enabled = self.stack.currentWidget() is self.welcome_screen
        for shortcut in getattr(self, "_help_shortcuts", []):
            shortcut.setEnabled(enabled)

    def update_toolbar_state(self, is_culling: bool):
        self.act_settings.setEnabled(True)
        self.act_complete.setEnabled(is_culling)

    def _set_export_ui_enabled(self, enabled: bool):
        if self.culling_widget:
            self.culling_widget.set_export_enabled(enabled)
        self.act_complete.setEnabled(enabled and bool(self.culling_widget))

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
        if not self.has_seen_tutorial:
            QTimer.singleShot(200, self._show_first_time_tutorial)
        self.update_toolbar_state(is_culling=True)

    def open_settings(self):
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec():
            if self.culling_widget:
                self.culling_widget.update_settings()
            self._refresh_help_shortcuts()
            self.status.showMessage("Settings updated.", 2000)
            
    def show_help_dialog(self, parent: Optional[QWidget] = None):
        dialog = HotkeyDialog(parent or self, self.settings.hotkeys)
        dialog.exec()

    def open_help(self):
        self.show_help_dialog()

    def open_about(self):
        dialog = AboutDialog(self, self.app_data_path)
        dialog.exec()

    def _show_first_time_tutorial(self):
        if not self.culling_widget:
            return
        self._show_help_if_first_time()

    def _show_help_if_first_time(self):
        if self.has_seen_tutorial:
            return
        self.show_help_dialog()
        self.has_seen_tutorial = True
        self._save_app_state()

    def _perform_export(
        self,
        on_success: Callable[[ExportResult], None],
        *,
        on_error: Optional[Callable[[str], None]] = None,
        on_cancel: Optional[Callable[[], None]] = None,
        finalize: Optional[Callable[[ExportResult], Optional[ExportResult]]] = None,
        finalize_message: Optional[str] = None,
    ) -> bool:
        if self._export_in_progress:
            self._show_toast("Export already in progress.", 2000)
            return False

        if not self.culling_widget:
            if on_error:
                on_error("No active culling session.")
            else:
                self._show_toast("No active culling session.", 2000)
            return False

        cw = self.culling_widget
        cw.save_all_dirty_files(wait=True)

        selected_raw_paths = [p.path for p in cw.catalog.photos if p.selected]

        worker = ExportWorker(
            root=cw.catalog.root,
            selected_raw_paths=selected_raw_paths,
            raw_output_folder_name=self.settings.raw_output_folder_name,
            jpeg_output_folder_name=self.settings.jpeg_output_folder_name,
        )

        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        dialog = QProgressDialog("Preparing export...", "Cancel", 0, 0, self)
        dialog.setWindowTitle("Exporting Selected Files")
        dialog.setWindowModality(Qt.ApplicationModal)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setMinimumDuration(0)
        dialog.setValue(0)
        dialog.setCancelButtonText("Cancel")

        dialog.canceled.connect(worker.request_cancel)

        self._export_in_progress = True
        self._export_worker = worker
        self._export_thread = thread
        self._export_progress_dialog = dialog
        self._export_success_callback = on_success
        self._export_error_callback = on_error
        self._export_cancel_callback = on_cancel
        self._export_finalizer = finalize
        self._export_finalizer_message = finalize_message
        self._export_finalizing = False

        worker.progress.connect(self._on_export_progress)
        worker.finished.connect(self._on_export_finished)
        worker.error.connect(self._on_export_error)
        worker.canceled.connect(self._on_export_canceled)

        self._set_export_ui_enabled(False)
        dialog.show()
        thread.start()
        return True

    def _on_export_progress(self, value: int, maximum: int, message: str):
        dialog = self._export_progress_dialog
        if not dialog or self._export_finalizing:
            return
        if maximum <= 0:
            dialog.setRange(0, 0)
        else:
            if dialog.maximum() != maximum or dialog.minimum() != 0:
                dialog.setRange(0, maximum)
            dialog.setValue(min(value, maximum))
        if message:
            dialog.setLabelText(message)

    def _finalize_export_state(self):
        dialog = self._export_progress_dialog
        if dialog:
            dialog.reset()
            dialog.close()
            dialog.deleteLater()
        self._export_progress_dialog = None

        worker = self._export_worker
        if worker:
            worker.deleteLater()
        self._export_worker = None

        thread = self._export_thread
        if thread:
            thread.quit()
            thread.wait()
            thread.deleteLater()
        self._export_thread = None

        self._set_export_ui_enabled(True)
        self._export_in_progress = False
        self._export_finalizer = None
        self._export_finalizer_message = None
        self._export_finalizing = False

    def _execute_export_finalizer(self, result: ExportResult) -> Optional[ExportResult]:
        finalizer = self._export_finalizer
        if not finalizer:
            return result

        dialog = self._export_progress_dialog
        self._export_finalizing = True
        if dialog:
            dialog.setRange(0, 0)
            if self._export_finalizer_message:
                dialog.setLabelText(self._export_finalizer_message)
            try:
                dialog.setCancelButton(None)
            except AttributeError:
                cancel_btn = dialog.findChild(QPushButton)
                if cancel_btn:
                    cancel_btn.setEnabled(False)
                    cancel_btn.hide()
            dialog.repaint()
            QApplication.processEvents()

        try:
            finalized = finalizer(result)
            return finalized if finalized is not None else result
        except Exception as exc:
            details = traceback.format_exc()
            message = f"Finalizing export failed: {exc}"
            self._on_export_error(message, details)
            return None
        finally:
            self._export_finalizing = False

    def _on_export_finished(self, result: ExportResult):
        finalized_result = self._execute_export_finalizer(result)
        if finalized_result is None:
            return

        callback = self._export_success_callback
        self._finalize_export_state()
        self._export_success_callback = None
        self._export_error_callback = None
        self._export_cancel_callback = None
        if callback:
            callback(finalized_result)

    def _on_export_error(self, message: str, details: str):
        print(f"[Export] Error: {message}\n{details}", file=sys.stderr)
        callback = self._export_error_callback
        self._finalize_export_state()
        self._export_success_callback = None
        self._export_error_callback = None
        self._export_cancel_callback = None
        if callback:
            callback(message)
        else:
            QMessageBox.critical(self, "Export Failed", message)

    def _on_export_canceled(self):
        callback = self._export_cancel_callback
        self._finalize_export_state()
        self._export_success_callback = None
        self._export_error_callback = None
        self._export_cancel_callback = None
        if callback:
            callback()
        else:
            self._show_toast("Export canceled.", 2000)

    def handle_export(self):
        def on_success(result: ExportResult):
            msg = (
                f"Sync complete · {result.selected_count} selected → "
                f"{result.dest_raw_count} RAW, {result.dest_jpeg_count} JPEG"
            )
            self._show_toast(msg, 1500)

        def on_error(message: str):
            QMessageBox.critical(self, "Export Failed", message)

        def on_cancel():
            self._show_toast("Export canceled.", 1500)

        self._perform_export(on_success, on_error=on_error, on_cancel=on_cancel)

    def complete_culling(self):
        def on_success(result: ExportResult):
            dlg = CompletionDialog(
                "Culling Complete",
                result.selected_count,
                result.raw_out_dir,
                result.dest_raw_count,
                result.jpeg_out_dir,
                result.dest_jpeg_count,
                self,
            )
            dlg.exec()

        def on_error(message: str):
            QMessageBox.critical(self, "Export Failed", message)

        def on_cancel():
            self._show_toast("Export canceled.", 1500)

        def finalize(result: ExportResult) -> ExportResult:
            cw = self.culling_widget
            if cw:
                cw.cleanup()
                self.stack.removeWidget(cw)
                cw.deleteLater()
                self.culling_widget = None

            self._load_app_state()
            self.welcome_screen.update_recent_folders(self.recent_folders)
            self.stack.setCurrentWidget(self.welcome_screen)
            self.update_toolbar_state(is_culling=False)
            self.status.clearMessage()
            return result

        self._perform_export(
            on_success,
            on_error=on_error,
            on_cancel=on_cancel,
            finalize=finalize,
            finalize_message="Cleaning up session...",
        )
    def _show_toast(self, text: str, ms: int = 1500):
        if not hasattr(self, "_toast_label"):
            self._toast_label = QLabel(self)
            self._toast_label.setObjectName("toastLabel")
            self._toast_label.setStyleSheet("""
                QLabel#toastLabel {
                    background: rgba(0,0,0,180);
                    color: %s;
                    border-radius: 8px;
                    padding: 10px 14px;
                    font-weight: 600;
                }
            """ % theme_color('text.primary'))
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

    resource_root = Path(__file__).resolve().parent
    if hasattr(sys, '_MEIPASS'):
        resource_root = Path(sys._MEIPASS)

    font_path = resource_root / "Gantari-Regular.ttf"

    if not font_path.exists():
        print(
            "Warning: Font file not found at '{}'. Ensure packaged builds include this resource.".format(
                font_path
            )
        )
        font_id = -1
    else:
        font_id = QFontDatabase.addApplicationFont(str(font_path))

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

    app.setStyleSheet(f"""
        QWidget {{
            background-color: {theme_color('bg.base')};
            color: {theme_color('text.primary')};
        }}
        QMainWindow, QDialog {{
            background-color: {theme_color('bg.base')};
        }}
        QToolBar {{
            background: {theme_color('bg.surface')};
            border: none;
            padding: 2px;
        }}
        QToolBar QWidget {{
            background: transparent;
        }}
        QToolBar QToolButton {{
            color: {theme_color('text.primary')};
            background: transparent;
            padding: 6px;
            margin: 2px;
        }}
        QToolBar QToolButton:hover {{
            background: {theme_color('bg.elevated')};
            border-radius: 3px;
        }}
        QToolBar QToolButton:pressed {{
            background: {theme_color('accent.muted')};
        }}
        QStatusBar {{
            background: {theme_color('bg.surface')};
            color: {theme_color('text.secondary')};
        }}
        QGroupBox {{
            border: 1px solid {theme_color('border.default')};
            border-radius: 5px;
            margin-top: 1ex;
            font-weight: bold;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 3px;
        }}
        QPushButton {{
            background-color: {theme_color('accent.primary')};
            color: {theme_color('text.on_accent')};
            border: 1px solid {theme_color('accent.hover')};
            padding: 8px 16px;
            border-radius: 4px;
        }}
        QPushButton:hover {{
            background-color: {theme_color('accent.hover')};
        }}
        QPushButton:pressed {{
            background-color: {theme_color('accent.active')};
        }}
        QPushButton:disabled {{
            background-color: {theme_color('bg.surface')};
            color: {theme_color('text.tertiary')};
            border: 1px solid {theme_color('border.default')};
        }}
        QPushButton:focus {{
            outline: none;
        }}
        QLineEdit, QSpinBox {{
            background-color: {theme_color('bg.surface')};
            border: 1px solid {theme_color('border.default')};
            border-radius: 3px;
            padding: 5px;
            color: {theme_color('text.primary')};
        }}
        QLabel {{
            background-color: transparent;
        }}
        QLabel a {{
            color: {theme_color('accent.primary')};
            text-decoration: none;
        }}
        QMessageBox {{
            background-color: {theme_color('bg.surface')};
        }}
    """)


    win = AppWindow(args)
    win.show()

    if args.root:
        win.select_folder_from_arg(args.root)
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()