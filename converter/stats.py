"""数据集统计收集"""
import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


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
            ratio = area / img_area if img_area > 0 else 0

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
