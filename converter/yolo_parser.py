"""自动识别两种YOLO目录布局，产出标准化数据"""
from pathlib import Path
from typing import List, Dict
from PIL import Image


LAYOUT_A_PATTERNS = [
    ("images/train", "labels/train"),
    ("images/val", "labels/val"),
    ("images/test", "labels/test"),
]
LAYOUT_B_SUBDIRS = ["train", "valid", "test"]


def detect_layout(dataset_root: Path) -> str:
    """检测YOLO数据集布局类型"""
    items = list(dataset_root.iterdir())
    names = {i.name for i in items}

    if "images" in names and "labels" in names:
        return "A"
    if all(s in names for s in LAYOUT_B_SUBDIRS):
        sub = list((dataset_root / "train").iterdir())
        sub_names = {s.name for s in sub}
        if "images" in sub_names and "labels" in sub_names:
            return "B"
    raise ValueError(f"无法识别的YOLO目录布局: {dataset_root}")


def _find_label(lbl_dir: Path, stem: str) -> Path | None:
    """根据图片stem查找对应label文件"""
    if not lbl_dir.exists():
        return None
    candidates = [
        lbl_dir / f"{stem}.txt",
        lbl_dir / f"{stem}.jpg.txt",
    ]
    for c in candidates:
        if c.exists():
            return c
    for p in lbl_dir.iterdir():
        if p.stem.startswith(stem.split(".")[0]):
            return p
    return None


def _make_record(img_path: Path, lbl_path: Path | None) -> Dict:
    """构建标准化记录"""
    with Image.open(img_path) as img:
        width, height = img.size

    labels = []
    if lbl_path and lbl_path.exists():
        with open(lbl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls, cx, cy, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                labels.append({"category_id": cls, "cx": cx, "cy": cy, "w": w, "h": h})

    return {
        "image_path": str(img_path),
        "label_path": str(lbl_path) if lbl_path else None,
        "width": width,
        "height": height,
        "labels": labels,
    }


def iter_layout_a(dataset_root: Path):
    """布局A: images/ + labels/ 并列"""
    for img_subdir, lbl_subdir in LAYOUT_A_PATTERNS:
        img_dir = dataset_root / img_subdir
        lbl_dir = dataset_root / lbl_subdir
        if not img_dir.exists():
            continue
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            stem = img_path.stem
            lbl_path = _find_label(lbl_dir, stem)
            yield _make_record(img_path, lbl_path)


def iter_layout_b(dataset_root: Path):
    """布局B: train/images + train/labels"""
    for split in LAYOUT_B_SUBDIRS:
        split_dir = dataset_root / split
        if not split_dir.exists():
            continue
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"
        if not img_dir.exists():
            continue
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            stem = img_path.stem
            lbl_path = _find_label(lbl_dir, stem)
            yield _make_record(img_path, lbl_path)


def iter_dataset(dataset_root: Path):
    """遍历YOLO数据集，自动识别布局"""
    layout = detect_layout(dataset_root)
    if layout == "A":
        yield from iter_layout_a(dataset_root)
    else:
        yield from iter_layout_b(dataset_root)
