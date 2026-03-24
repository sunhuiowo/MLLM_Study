"""输出 ms-SWIFT grounding 格式 (VLM 微调)"""
import json
from pathlib import Path


class QwenVLWriter:
    """输出 ms-SWIFT grounding 格式

    格式:
    {
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "<image>请找出图像中所有目标..."},
        {"role": "assistant", "content": "<ref-object>label<bbox>[x1,y1,x2,y2]..."}
      ],
      "images": ["images/xxx.jpg"],
      "objects": {
        "ref": ["label1", "label2"],
        "bbox": [[x1,y1,x2,y2], [x1,y1,x2,y2]]
      }
    }
    """

    def __init__(self, output_path: str | Path, system_prompt: str = "你是一个目标识别专家，你可以识别到图片中存在的物体，并可以给出相应的位置。"):
        self.fh = open(output_path, "w", encoding="utf-8")
        self.system_prompt = system_prompt
        self._count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fh.close()
        return False

    @property
    def count(self):
        return self._count

    def write(self, image_path: str, instruction: str, labels: list, bboxes: list):
        """写入单条 ms-SWIFT grounding 记录

        Args:
            image_path: 图片路径（相对路径）
            instruction: 用户指令文本
            labels: 类别名称列表，如 ["safety_vest", "no_safety_vest"]
            bboxes: 边界框列表，如 [[x1,y1,x2,y2], [x1,y1,x2,y2]]
        """
        # 构建 assistant 回复内容: <ref-object>label<bbox>[x1,y1,x2,y2]...
        parts = []
        for label, bbox in zip(labels, bboxes):
            # bbox 转为浮点数
            bbox_float = [float(v) for v in bbox]
            bbox_str = f"[{bbox_float[0]},{bbox_float[1]},{bbox_float[2]},{bbox_float[3]}]"
            parts.append(f"<ref-object>{label}<bbox>{bbox_str}")
        response = "".join(parts)

        record = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"<image>{instruction}"},
                {"role": "assistant", "content": response}
            ],
            "images": [image_path],
            "objects": {
                "ref": labels,
                "bbox": [[float(v) for v in bbox] for bbox in bboxes]
            }
        }
        self.fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._count += 1

    def close(self):
        self.fh.close()
