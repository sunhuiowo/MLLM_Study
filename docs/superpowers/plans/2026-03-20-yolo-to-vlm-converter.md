# YOLO → VLM 数据转换系统实现计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 YOLO 检测数据集自动转换为 VLM 指令微调数据，支持两种 YOLO 目录排版，输出 CV 标注 JSONL 与对话式训练 JSONL。

**Architecture:** 流水线架构：扫描 → 解析 → 转换 → 写出。yolo_parser.py 自动识别两种目录布局并标准化输出；transform.py 负责坐标转换；label_mapper.py 管理类别映射；instruction_generator.py 提供多模板随机 instruction；cv_writer.py 和 vlm_writer.py 分别输出两种格式。

**Tech Stack:** Python 3.8+, Pillow, PyYAML, pathlib, json, random, argparse

---

## Chunk 1: 项目结构搭建与 label_mapper.py

**Files:**
- Create: `converter/main.py`
- Create: `converter/label_mapper.py`
- Create: `converter/__init__.py`
- Create: `converter/label_map.yaml`
- Test: `converter/tests/test_label_mapper.py`

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p /Users/frontis/Desktop/SH/Train_L_VLM/converter/tests
touch /Users/frontis/Desktop/SH/Train_L_VLM/converter/__init__.py
```

- [ ] **Step 2: 编写 label_map.yaml**

```yaml
0:
  name: safety_vest
  cn: 反光衣
1:
  name: no_safety_vest
  cn: 未穿反光衣
```

- [ ] **Step 3: 编写 label_mapper.py**

```python
"""category_id -> {name, cn} 映射管理"""
import yaml
from pathlib import Path
from typing import Dict

class LabelMapper:
    def __init__(self, map_path: str | Path):
        with open(map_path, "r", encoding="utf-8") as f:
            self._map: Dict[int, Dict[str, str]] = yaml.safe_load(f)

    def get_name(self, category_id: int) -> str:
        return self._map[category_id]["name"]

    def get_cn(self, category_id: int) -> str:
        return self._map[category_id]["cn"]

    def get_all(self, category_id: int) -> Dict[str, any]:
        return {"category_id": category_id, **self._map[category_id]}
```

- [ ] **Step 4: 编写 test_label_mapper.py**

```python
import pytest
from converter.label_mapper import LabelMapper
from pathlib import Path

def test_get_name():
    mapper = LabelMapper(Path(__file__).parent.parent / "label_map.yaml")
    assert mapper.get_name(0) == "safety_vest"
    assert mapper.get_name(1) == "no_safety_vest"

def test_get_cn():
    mapper = LabelMapper(Path(__file__).parent.parent / "label_map.yaml")
    assert mapper.get_cn(0) == "反光衣"

def test_get_all():
    mapper = LabelMapper(Path(__file__).parent.parent / "label_map.yaml")
    result = mapper.get_all(0)
    assert result == {"category_id": 0, "name": "safety_vest", "cn": "反光衣"}
```

- [ ] **Step 5: 运行测试验证**

```bash
cd /Users/frontis/Desktop/SH/Train_L_VLM
pytest converter/tests/test_label_mapper.py -v
```

---

## Chunk 2: transform.py（坐标转换）

**Files:**
- Create: `converter/transform.py`
- Test: `converter/tests/test_transform.py`

- [ ] **Step 1: 编写 transform.py**

```python
"""YOLO 归一化坐标 -> 像素 [x1, y1, x2, y2]"""
from typing import List, Tuple

def yolo_to_pixel(cx: float, cy: float, w: float, h: float, width: int, height: int) -> Tuple[int, int, int, int]:
    """YOLO归一化坐标转换为像素[x1,y1,x2,y2]

    Args:
        cx, cy, w, h: YOLO归一化值 (0-1)
        width, height: 图片宽高（像素）

    Returns:
        (x1, y1, x2, y2) 像素整数坐标
    """
    cx_pixel = float(cx) * width
    cy_pixel = float(cy) * height
    w_pixel  = float(w)  * width
    h_pixel  = float(h)  * height

    x1 = int(cx_pixel - w_pixel / 2)
    y1 = int(cy_pixel - h_pixel / 2)
    x2 = int(cx_pixel + w_pixel / 2)
    y2 = int(cy_pixel + h_pixel / 2)
    return (x1, y1, x2, y2)


def parse_yolo_line(line: str) -> Tuple[int, float, float, float, float]:
    """解析YOLO标签行: class_id cx cy w h"""
    parts = line.strip().split()
    cls = int(parts[0])
    cx, cy, w, h = map(float, parts[1:5])
    return cls, cx, cy, w, h


def transform_yolo_to_pixel(line: str, width: int, height: int) -> dict:
    """解析并转换单行YOLO标签为像素坐标字典"""
    cls, cx, cy, w, h = parse_yolo_line(line)
    x1, y1, x2, y2 = yolo_to_pixel(cx, cy, w, h, width, height)
    return {
        "category_id": cls,
        "cx_norm": cx, "cy_norm": cy, "w_norm": w, "h_norm": h,
        "bbox": [x1, y1, x2, y2]
    }
```

- [ ] **Step 2: 编写 test_transform.py**

```python
import pytest
from converter.transform import yolo_to_pixel, parse_yolo_line, transform_yolo_to_pixel

def test_yolo_to_pixel():
    # 归一化中心点(0.5, 0.5), 宽高(0.2, 0.3) 在1920x1080图片上
    x1, y1, x2, y2 = yolo_to_pixel(0.5, 0.5, 0.2, 0.3, 1920, 1080)
    assert x1 == 576   # 960 - 192
    assert y1 == 378   # 540 - 162
    assert x2 == 1344  # 960 + 192
    assert y2 == 702   # 540 + 162

def test_parse_yolo_line():
    cls, cx, cy, w, h = parse_yolo_line("0 0.569712 0.340144 0.067308 0.161058")
    assert cls == 0
    assert abs(cx - 0.569712) < 1e-6

def test_transform_yolo_to_pixel():
    result = transform_yolo_to_pixel("0 0.5 0.5 0.2 0.3", 1920, 1080)
    assert result["category_id"] == 0
    assert result["bbox"] == [576, 378, 1344, 702]
```

- [ ] **Step 3: 运行测试验证**

```bash
pytest converter/tests/test_transform.py -v
```

---

## Chunk 3: yolo_parser.py（目录布局解析）

**Files:**
- Create: `converter/yolo_parser.py`
- Test: `converter/tests/test_yolo_parser.py`

- [ ] **Step 1: 编写 yolo_parser.py**

```python
"""自动识别两种YOLO目录布局，产出标准化数据"""
from pathlib import Path
from typing import List, Dict
import re

# 布局A: images/train + labels/train 并列
LAYOUT_A_PATTERNS = [
    ("images/train", "labels/train"),
    ("images/val", "labels/val"),
    ("images/test", "labels/test"),
]

# 布局B: train/images + train/labels 在同一父目录
LAYOUT_B_SUBDIRS = ["train", "val", "test"]


def detect_layout(dataset_root: Path) -> str:
    """检测YOLO数据集布局类型"""
    items = list(dataset_root.iterdir())
    names = {i.name for i in items}

    # 布局A特征: 有 images/ 和 labels/ 直接子目录
    if "images" in names and "labels" in names:
        return "A"

    # 布局B特征: 有 train/val/test 子目录，子目录内含 images/ 和 labels/
    if all(s in names for s in LAYOUT_B_SUBDIRS):
        sub = list((dataset_root / "train").iterdir())
        sub_names = {s.name for s in sub}
        if "images" in sub_names and "labels" in sub_names:
            return "B"

    raise ValueError(f"无法识别的YOLO目录布局: {dataset_root}")


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
            # YOLO标签文件可能在子目录中，尝试匹配 stem
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
    # 模糊匹配：stem包含hash部分
    for p in lbl_dir.iterdir():
        if p.stem.startswith(stem.split(".")[0]):
            return p
    return None


def _make_record(img_path: Path, lbl_path: Path | None) -> Dict:
    """构建标准化记录"""
    from PIL import Image
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


def iter_dataset(dataset_root: Path):
    """遍历YOLO数据集，自动识别布局"""
    layout = detect_layout(dataset_root)
    if layout == "A":
        yield from iter_layout_a(dataset_root)
    else:
        yield from iter_layout_b(dataset_root)
```

- [ ] **Step 2: 编写 test_yolo_parser.py**

```python
import pytest
from pathlib import Path
from converter.yolo_parser import detect_layout, iter_dataset

def test_detect_layout_a():
    path = Path("/Users/frontis/Desktop/SH/Train_L_VLM/datasets/YOLO/Safety_Vests_Detection_Dataset_YOLO")
    assert detect_layout(path) == "A"

def test_detect_layout_b():
    path = Path("/Users/frontis/Desktop/SH/Train_L_VLM/datasets/YOLO/vest_v1i_yolov11")
    assert detect_layout(path) == "B"
```

- [ ] **Step 3: 运行测试验证**

```bash
pytest converter/tests/test_yolo_parser.py -v
```

---

## Chunk 4: instruction_generator.py + cv_writer.py + vlm_writer.py

**Files:**
- Create: `converter/instruction_generator.py`
- Create: `converter/cv_writer.py`
- Create: `converter/vlm_writer.py`
- Test: `converter/tests/test_instruction_generator.py`

- [ ] **Step 1: 编写 instruction_generator.py**

```python
"""多模板 instruction 生成"""
import random
from typing import List

TEMPLATES = [
    "请检测图像中的所有目标，并输出bbox_2d（[x1,y1,x2,y2]）和label。",
    "识别图中的所有目标，给出它们的边界框和类别。",
    "请找出图像中所有目标，输出bbox_2d和label。",
]

class InstructionGenerator:
    def __init__(self, templates: List[str] | None = None):
        self.templates = templates or TEMPLATES

    def generate(self) -> str:
        return random.choice(self.templates)
```

- [ ] **Step 2: 编写 test_instruction_generator.py**

```python
import pytest
from converter.instruction_generator import InstructionGenerator, TEMPLATES

def test_generate_returns_template():
    gen = InstructionGenerator()
    result = gen.generate()
    assert result in TEMPLATES

def test_custom_templates():
    custom = ["custom instruction"]
    gen = InstructionGenerator(templates=custom)
    assert gen.generate() == "custom instruction"
```

- [ ] **Step 3: 编写 cv_writer.py**

```python
"""输出 annotations.jsonl（CV用结构化标注）"""
import json
from pathlib import Path
from typing import List, Dict, Callable

class CVWriter:
    def __init__(self, output_path: str | Path):
        self.fh = open(output_path, "w", encoding="utf-8")

    def write(self, record: Dict):
        """写入单条CV标注记录"""
        self.fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self):
        self.fh.close()
```

- [ ] **Step 4: 编写 vlm_writer.py**

```python
"""输出 vlm_train.jsonl（VLM指令微调用）"""
import json
from pathlib import Path
from typing import List, Dict

class VLMWriter:
    def __init__(self, output_path: str | Path):
        self.fh = open(output_path, "w", encoding="utf-8")

    def write(self, image_rel_path: str, annotations: List[Dict], instruction: str):
        """写入单条VLM对话记录

        Args:
            image_rel_path: 图片相对路径（相对于VLM/images/）
            annotations: [{"bbox": [x1,y1,x2,y2], "category_name": "..."}, ...]
            instruction: 人类问题文本
        """
        gpt_items = [
            {"bbox_2d": ann["bbox"], "label": ann["category_name"]}
            for ann in annotations
        ]
        record = {
            "image": image_rel_path,
            "conversations": [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": json.dumps(gpt_items, ensure_ascii=False)},
            ]
        }
        self.fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self):
        self.fh.close()
```

- [ ] **Step 5: 运行测试验证**

```bash
pytest converter/tests/test_instruction_generator.py -v
```

---

## Chunk 5: stats.py + main.py（统计 + 入口）

**Files:**
- Create: `converter/stats.py`
- Create: `converter/main.py`
- Test: `converter/tests/test_stats.py`

- [ ] **Step 1: 编写 stats.py**

```python
"""数据集统计收集"""
from typing import Dict, List
from collections import defaultdict
import json

class Stats:
    def __init__(self):
        self.total_images = 0
        self.total_annotations = 0
        self.category_counts: Dict[int, int] = defaultdict(int)
        self.bbox_sizes: Dict[str, int] = {"small": 0, "medium": 0, "large": 0}
        self.annotations_per_image: List[int] = []
        self.empty_images = 0

    def update(self, annotations: List[Dict], width: int, height: int):
        self.total_images += 1
        count = len(annotations)
        self.annotations_per_image.append(count)
        self.total_annotations += count

        if count == 0:
            self.empty_images += 1
            return

        for ann in annotations:
            cid = ann["category_id"]
            self.category_counts[cid] += 1

            x1, y1, x2, y2 = ann["bbox"]
            area = (x2 - x1) * (y2 - y1)
            img_area = width * height
            ratio = area / img_area

            if ratio < 0.01:
                self.bbox_sizes["small"] += 1
            elif ratio < 0.1:
                self.bbox_sizes["medium"] += 1
            else:
                self.bbox_sizes["large"] += 1

    def to_dict(self) -> Dict:
        ann_counts = self.annotations_per_image
        return {
            "total_images": self.total_images,
            "total_annotations": self.total_annotations,
            "category_stats": [
                {"category_id": cid, "count": cnt}
                for cid, cnt in sorted(self.category_counts.items())
            ],
            "empty_images": self.empty_images,
            "bbox_size_distribution": self.bbox_sizes,
            "annotations_per_image": {
                "min": min(ann_counts) if ann_counts else 0,
                "max": max(ann_counts) if ann_counts else 0,
                "avg": round(sum(ann_counts) / len(ann_counts), 2) if ann_counts else 0,
            }
        }

    def save(self, path: str | Path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
```

- [ ] **Step 2: 编写 test_stats.py**

```python
import pytest
from converter.stats import Stats

def test_stats_empty():
    s = Stats()
    s.update([], 1920, 1080)
    assert s.total_images == 1
    assert s.empty_images == 1

def test_stats_with_annotations():
    s = Stats()
    s.update([{"category_id": 0, "bbox": [100, 100, 200, 200]}], 1920, 1080)
    assert s.total_images == 1
    assert s.total_annotations == 1
    assert s.category_counts[0] == 1
```

- [ ] **Step 3: 编写 main.py**

```python
"""命令行入口"""
import argparse
import shutil
from pathlib import Path
from converter.yolo_parser import iter_dataset, detect_layout
from converter.transform import transform_yolo_to_pixel
from converter.label_mapper import LabelMapper
from converter.instruction_generator import InstructionGenerator
from converter.cv_writer import CVWriter
from converter.vlm_writer import VLMWriter
from converter.stats import Stats


def build_relative_path(abs_image_path: Path, vlm_root: Path) -> str:
    """从VLM根目录算起的相对路径"""
    # 保持原有子目录结构
    return str(abs_image_path.name)


def main():
    parser = argparse.ArgumentParser(description="YOLO -> VLM 数据转换")
    parser.add_argument("--input", type=str, required=True, help="YOLO数据集根目录")
    parser.add_argument("--output", type=str, required=True, help="VLM输出根目录")
    parser.add_argument("--label-map", type=str, default=None, help="label_map.yaml路径")
    parser.add_argument("--copy-images", action="store_true", default=True, help="是否拷贝图片")
    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)
    images_dir = output_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # label_map
    if args.label_map:
        label_mapper = LabelMapper(args.label_map)
    else:
        # 默认使用内置map（需要label_mapper.py支持默认路径）
        label_mapper = LabelMapper(Path(__file__).parent / "label_map.yaml")

    instruction_gen = InstructionGenerator()
    stats = Stats()

    cv_writer = CVWriter(output_root / "annotations.jsonl")
    vlm_writer = VLMWriter(output_root / "vlm_train.jsonl")

    # 遍历每个子数据集
    for sub_dataset in input_root.iterdir():
        if not sub_dataset.is_dir():
            continue
        print(f"处理数据集: {sub_dataset.name}")
        for record in iter_dataset(sub_dataset):
            # 拷贝图片
            img_src = Path(record["image_path"])
            img_dst = images_dir / f"{sub_dataset.name}_{img_src.name}"
            if args.copy_images and not img_dst.exists():
                shutil.copy2(img_src, img_dst)

            img_rel = f"images/{sub_dataset.name}_{img_src.name}"

            # 转换坐标 + 映射类别
            cv_annotations = []
            vlm_annotations = []
            for lbl in record["labels"]:
                bbox_pixel = transform_yolo_to_pixel(
                    f"{lbl['category_id']} {lbl['cx']} {lbl['cy']} {lbl['w']} {lbl['h']}",
                    record["width"],
                    record["height"],
                )
                label_info = label_mapper.get_all(bbox_pixel["category_id"])
                cv_ann = {
                    "category_id": label_info["category_id"],
                    "category_name": label_info["name"],
                    "category_cn": label_info["cn"],
                    "bbox": bbox_pixel["bbox"],
                }
                cv_annotations.append(cv_ann)
                vlm_annotations.append(cv_ann)

            # 写入CV标注
            cv_writer.write({
                "image_path": img_rel,
                "width": record["width"],
                "height": record["height"],
                "annotations": cv_annotations,
            })

            # 写入VLM数据
            instruction = instruction_gen.generate()
            vlm_writer.write(img_rel, vlm_annotations, instruction)

            # 统计
            stats.update(vlm_annotations, record["width"], record["height"])

    cv_writer.close()
    vlm_writer.close()
    stats.save(output_root / "stats.json")
    print(f"完成！输出目录: {output_root}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行完整测试验证**

```bash
cd /Users/frontis/Desktop/SH/Train_L_VLM
python converter/main.py \
    --input datasets/YOLO \
    --output datasets/VLM \
    --label-map converter/label_map.yaml
```

预期：datasets/VLM/ 下生成 images/, annotations.jsonl, vlm_train.jsonl, stats.json

---

## Chunk 6: 最终验证

**Files:**
- Verify: `datasets/VLM/annotations.jsonl`
- Verify: `datasets/VLM/vlm_train.jsonl`
- Verify: `datasets/VLM/stats.json`

- [ ] **Step 1: 检查 vlm_train.jsonl 格式**

```bash
head -2 datasets/VLM/vlm_train.jsonl | python -m json.tool
```

预期输出包含 `"image"`, `"conversations"` 两个key

- [ ] **Step 2: 检查空标注处理**

```bash
grep '"bbox_2d": \[\]' datasets/VLM/vlm_train.jsonl | head -1
```

预期：有或无空标注结果都合法

- [ ] **Step 3: 检查 stats.json**

```bash
cat datasets/VLM/stats.json | python -m json.tool
```

预期：包含 total_images, category_stats, bbox_size_distribution
