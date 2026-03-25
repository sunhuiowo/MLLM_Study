# YOLO → VLM 数据转换系统设计

## 一、目标

将 YOLO 检测数据集自动转换为 VLM（视觉-语言模型）指令微调数据，支持两种 YOLO 目录排版输入，输出结构化标注 JSONL 与对话式训练数据 JSONL。

## 二、输入数据

### 2.1 两种 YOLO 目录排版

**布局A** — `Safety_Vests_Detection_Dataset_YOLO`:
```
images/train/, images/val/, images/test/
labels/train/, labels/val/, labels/test/
```

**布局B** — `vest_v1i_yolov11`:
```
train/images/, train/labels/
val/images/, val/labels/
test/images/, test/labels/
```

### 2.2 YOLO 标签格式（归一化）

每行: `class_id center_x center_y width height`（全部 0-1 归一化）

### 2.3 类别映射（示例）

```python
LABEL_MAP = {
    0: {"name": "safety_vest", "cn": "反光衣"},
    1: {"name": "no_safety_vest", "cn": "未穿反光衣"},
}
```

## 三、输出结构

```
datasets/VLM/
├── images/                  # 原图拷贝
├── annotations.jsonl        # CV用结构化标注（一行一样本）
├── vlm_train.jsonl          # VLM指令微调数据（核心输出）
└── stats.json               # 数据集统计
```

## 四、输出格式详解

### 4.1 annotations.jsonl（CV用）

```json
{"image_path": "Safety_Vests_Detection_Dataset_YOLO/train/xxx.jpg", "width": 1920, "height": 1080, "annotations": [{"category_id": 0, "category_name": "safety_vest", "category_cn": "反光衣", "bbox": [100, 200, 300, 400]}]}
```

### 4.2 vlm_train.jsonl（VLM微调用）

```json
{
  "image": "relative/path/to/image.jpg",
  "conversations": [
    {"from": "human", "value": "请检测图像中的所有目标，并输出bbox_2d（[x1,y1,x2,y2]）和label。"},
    {"from": "gpt", "value": "[{\"bbox_2d\": [100, 200, 300, 400], \"label\": \"safety_vest\"}]"}
  ]
}
```

- bbox_2d 为像素坐标整数 [x1, y1, x2, y2]
- label 使用 category_name（英文）

### 4.3 stats.json

```json
{
  "total_images": 3897,
  "total_annotations": 8472,
  "category_stats": [{"category_id": 0, "category_name": "safety_vest", "count": 6448}, ...],
  "empty_images": 23,
  "bbox_size_distribution": {"small": 1200, "medium": 5000, "large": 2272},
  "annotations_per_image_distribution": {"min": 0, "max": 15, "avg": 2.17}
}
```

## 五、坐标转换公式

归一化 YOLO → 像素 [x1, y1, x2, y2]:

```
cls, cx, cy, w, h = line.split()           # class_id 不参与坐标运算

cx_pixel = float(cx) * width
cy_pixel = float(cy) * height
w_pixel  = float(w)  * width
h_pixel  = float(h)  * height

x1 = int(cx_pixel - w_pixel / 2)
y1 = int(cy_pixel - h_pixel / 2)
x2 = int(cx_pixel + w_pixel / 2)
y2 = int(cy_pixel + h_pixel / 2)
```

## 六、模块架构

```
converter/
├── main.py                  # 命令行入口，参数解析
├── yolo_parser.py           # 自动识别两种YOLO目录布局，产出标准化的 (image_path, label_path, width, height, labels[])
├── transform.py             # YOLO归一化坐标 → 像素 [x1,y1,x2,y2]
├── label_mapper.py           # category_id → {name, cn}
├── instruction_generator.py  # instruction 模板管理，支持多模板随机
├── cv_writer.py             # 写入 annotations.jsonl
├── vlm_writer.py            # 写入 vlm_train.jsonl
└── stats.py                 # 收集统计信息，写入 stats.json
```

### 6.1 yolo_parser.py 标准化输出格式

```python
{
    "image_path": str,
    "label_path": str,
    "width": int,
    "height": int,
    "labels": [{"category_id": int, "cx": float, "cy": float, "w": float, "h": float}]
}
```

### 6.2 instruction_generator.py

支持多模板随机抽取：
```python
TEMPLATES = [
    "请检测图像中的所有目标，并输出bbox_2d（[x1,y1,x2,y2]）和label。",
    "识别图中的所有目标，给出它们的边界框和类别。",
    "请找出图像中所有违规目标，输出bbox_2d和label。",
]
```

### 6.3 vlm_writer.py GPT回复格式

```python
[{"bbox_2d": [x1, y1, x2, y2], "label": "safety_vest"}, ...]
```

- 无目标时: `[]`
- 多目标时: 列表拼接

## 七、空标注样本处理

当某图片对应的 label 文件为空或不存在时：
- `annotations.jsonl`: `"annotations": []`
- `vlm_train.jsonl`: GPT回复为 `"value": "[]"`，图片保留但无检测目标

## 八、处理流程

1. **扫描输入目录** — 遍历 `datasets/YOLO/` 下所有子数据集
2. **自动识别布局** — 根据目录结构判断是布局A还是布局B
3. **遍历图片** — 匹配图片与对应 label 文件
4. **读取宽高** — 用 PIL/cv2 读取原图获取 width, height
5. **解析YOLO标签** — 读取 .txt，每行解析为 (class_id, cx, cy, w, h)
6. **坐标转换** — 归一化 → 像素 [x1, y1, x2, y2]
7. **类别映射** — category_id → name/cn
8. **生成instruction** — 随机抽取一个模板
9. **写入annotations.jsonl** — CV用
10. **写入vlm_train.jsonl** — VLM用
11. **更新统计** — 计数、分布计算
12. **写入stats.json** — 最终统计

## 九、命令行接口

```bash
python converter/main.py \
    --input datasets/YOLO \
    --output datasets/VLM \
    --label-map converter/label_map.yaml
```

## 十、依赖

- Python 3.8+
- Pillow（读取图片尺寸）
- PyYAML（读取 label_map 配置）
- 标准库: pathlib, json, random, argparse
