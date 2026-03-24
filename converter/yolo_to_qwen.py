"""将已转换的 annotations.jsonl 转为 Qwen VL messages 格式"""
import argparse
import json
from pathlib import Path

from converter.instruction_generator import InstructionGenerator
from converter.vlm_writer import QwenVLWriter


def read_annotations(annotations_path: Path):
    """逐行读取 annotations.jsonl"""
    with open(annotations_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def convert_dataset(
    dataset_root: Path,
    output_path: Path,
    system_prompt: str = "你是一个目标识别专家，你可以识别到图片中存在的物体，并可以给出相应的位置。",
):
    """将数据集的 annotations.jsonl 转为 Qwen VL 格式

    Args:
        dataset_root: 数据集根目录（包含 annotations.jsonl）
        output_path: 输出 jsonl 路径
        system_prompt: system prompt
    """
    annotations_path = dataset_root / "annotations.jsonl"

    if not annotations_path.exists():
        raise FileNotFoundError(f"annotations.jsonl not found: {annotations_path}")

    # 先读取所有记录，收集类别
    all_records = list(read_annotations(annotations_path))

    # 从 annotations 直接推断所有类别
    categories = set()
    for record in all_records:
        for ann in record.get("annotations", []):
            categories.add(ann["category_name"])
    categories = sorted(categories)

    instruction_gen = InstructionGenerator(categories=categories)
    writer = QwenVLWriter(output_path, system_prompt=system_prompt)

    for record in all_records:
        image_rel = record["image_path"]  # e.g. "images/xxx.jpg"
        instruction = instruction_gen.generate()

        annotations = record.get("annotations", [])
        labels = [ann["category_name"] for ann in annotations]
        bboxes = [ann["bbox"] for ann in annotations]

        writer.write(image_rel, instruction, labels, bboxes)

    writer.close()
    print(f"  转换完成: {len(all_records)} 条记录 -> {output_path}")
    return len(all_records)


def main():
    parser = argparse.ArgumentParser(description="annotations.jsonl -> Qwen VL messages 格式")
    parser.add_argument("--input", type=str, required=True, help="数据集根目录（包含 data.yaml 和 annotations.jsonl）")
    parser.add_argument("--output", type=str, required=True, help="输出 jsonl 路径")
    parser.add_argument("--system-prompt", type=str, default="你是一个目标识别专家，你可以识别到图片中存在的物体，并可以给出相应的位置。", help="system prompt")
    args = parser.parse_args()

    count = convert_dataset(
        dataset_root=Path(args.input),
        output_path=Path(args.output),
        system_prompt=args.system_prompt,
    )
    print(f"总计转换: {count} 条")


if __name__ == "__main__":
    main()
