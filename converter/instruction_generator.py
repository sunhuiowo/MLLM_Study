"""多模板 instruction 生成，支持动态注入类别名称"""
import random
import yaml
from pathlib import Path
from typing import List, Dict, Optional

TEMPLATES = [
    "请检测图像中的所有目标，类别包括：{categories}。输出bbox_2d（[x1,y1,x2,y2]）和label。",
    "识别图中的所有目标，类别包括：{categories}。给出它们的边界框和类别。",
    "请找出图像中所有目标，类别包括：{categories}。输出bbox_2d和label。",
]


def load_categories_from_data_yaml(dataset_path: Path) -> List[str]:
    """从 data.yaml 读取类别名称列表，自动识别编码"""
    yaml_path = dataset_path / "data.yaml"
    if not yaml_path.exists():
        return []

    # 尝试多种编码，防止服务器上 data.yaml 编码不一致
    for encoding in ["utf-8", "gbk", "gb2312", "latin1"]:
        try:
            with open(yaml_path, "r", encoding=encoding) as f:
                data = yaml.safe_load(f)
            break
        except UnicodeDecodeError:
            continue

    names = data.get("names", []) if data else []
    names = data.get("names", []) if data else []
    # names 可能是 list 或 dict
    if isinstance(names, list):
        return names
    if isinstance(names, dict):
        return list(names.values())
    return []


def make_instruction(categories: List[str], template: Optional[str] = None) -> str:
    """生成包含类别信息的 instruction"""
    if not categories:
        categories_str = "未知类别"
    else:
        categories_str = ", ".join(categories)

    if template is None:
        template = random.choice(TEMPLATES)

    return template.format(categories=categories_str)


class InstructionGenerator:
    def __init__(self, categories: List[str] | None = None, templates: List[str] | None = None):
        self.templates = templates or TEMPLATES
        self.categories = categories or []

    def generate(self, categories: List[str] | None = None) -> str:
        """生成 instruction，可临时指定类别覆盖默认类别"""
        cats = categories if categories is not None else self.categories
        return make_instruction(cats, random.choice(self.templates))
