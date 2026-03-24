"""YOLO 归一化坐标 -> 像素 [x1, y1, x2, y2]"""
from typing import Tuple


def yolo_to_pixel(cx: float, cy: float, w: float, h: float, width: int, height: int) -> Tuple[int, int, int, int]:
    """YOLO归一化坐标转换为像素[x1,y1,x2,y2]"""
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
        "cx_norm": cx,
        "cy_norm": cy,
        "w_norm": w,
        "h_norm": h,
        "bbox": [x1, y1, x2, y2],
    }
