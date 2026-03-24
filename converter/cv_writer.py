"""输出 annotations.jsonl（CV用结构化标注）"""
import json
from pathlib import Path
from typing import Dict


class CVWriter:
    def __init__(self, output_path: str | Path):
        self.fh = open(output_path, "w", encoding="utf-8")

    def write(self, record: Dict):
        self.fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self):
        self.fh.close()
