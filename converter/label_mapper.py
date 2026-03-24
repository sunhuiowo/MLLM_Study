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

    def get_all(self, category_id: int) -> Dict:
        return {"category_id": category_id, **self._map[category_id]}
