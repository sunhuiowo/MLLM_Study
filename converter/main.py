"""命令行入口"""
import argparse
import json
import shutil
from pathlib import Path
from converter.yolo_parser import iter_dataset
from converter.transform import transform_yolo_to_pixel
from converter.label_mapper import LabelMapper
from converter.instruction_generator import InstructionGenerator, load_categories_from_data_yaml
from converter.cv_writer import CVWriter
from converter.vlm_writer import QwenVLWriter
from converter.stats import Stats


def process_dataset(
    dataset_path: Path,
    output_root: Path,
    label_mapper: LabelMapper,
    copy_images: bool,
):
    """处理单个 YOLO 数据集，输出到独立目录"""
    dataset_name = dataset_path.name
    out_dir = output_root / dataset_name
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # 从 data.yaml 读取类别
    categories = load_categories_from_data_yaml(dataset_path)
    instruction_gen = InstructionGenerator(categories=categories)
    stats = Stats()

    cv_writer = CVWriter(out_dir / "annotations.jsonl")
    vlm_writer = QwenVLWriter(out_dir / "vlm_train.jsonl")

    print(f"  处理数据集: {dataset_name}, 类别: {categories}")
    for record in iter_dataset(dataset_path):
        img_src = Path(record["image_path"])
        img_dst = images_dir / img_src.name

        if copy_images and not img_dst.exists():
            shutil.copy2(img_src, img_dst)

        img_rel = f"images/{img_src.name}"

        cv_annotations = []
        for lbl in record["labels"]:
            yolo_line = f"{lbl['category_id']} {lbl['cx']} {lbl['cy']} {lbl['w']} {lbl['h']}"
            bbox_pixel = transform_yolo_to_pixel(yolo_line, record["width"], record["height"])
            label_info = label_mapper.get_all(bbox_pixel["category_id"])
            ann = {
                "category_id": label_info["category_id"],
                "category_name": label_info["name"],
                "category_cn": label_info["cn"],
                "bbox": bbox_pixel["bbox"],
            }
            cv_annotations.append(ann)

        cv_writer.write({
            "image_path": img_rel,
            "width": record["width"],
            "height": record["height"],
            "annotations": cv_annotations,
        })

        instruction = instruction_gen.generate()
        labels = [ann["category_name"] for ann in cv_annotations]
        bboxes = [ann["bbox"] for ann in cv_annotations]
        vlm_writer.write(img_rel, instruction, labels, bboxes)

        stats.update(cv_annotations, record["width"], record["height"])

    cv_writer.close()
    vlm_writer.close()
    stats.save(out_dir / "stats.json")

    print(f"    图片数: {stats.total_images}, 标注数: {stats.total_annotations}, 空图: {stats.empty_images}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="YOLO -> VLM 数据转换")
    parser.add_argument("--input", type=str, required=True, help="YOLO数据集根目录")
    parser.add_argument("--output", type=str, required=True, help="VLM输出根目录")
    parser.add_argument("--label-map", type=str, default=None, help="label_map.yaml路径")
    parser.add_argument("--no-copy-images", action="store_true", help="不拷贝图片")
    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    label_mapper = LabelMapper(
        args.label_map if args.label_map else Path(__file__).parent / "label_map.yaml"
    )
    copy_images = not args.no_copy_images

    total_images = 0
    total_annotations = 0

    for sub_dataset in sorted(input_root.iterdir()):
        if not sub_dataset.is_dir():
            continue
        stats = process_dataset(sub_dataset, output_root, label_mapper, copy_images)
        total_images += stats.total_images
        total_annotations += stats.total_annotations

    print(f"\n完成！输出目录: {output_root}")
    print(f"  累计图片数: {total_images}")
    print(f"  累计标注数: {total_annotations}")


if __name__ == "__main__":
    main()
