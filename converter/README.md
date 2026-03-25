# YOLO → ms-SWIFT Groundding 数据转换器

将 YOLO 检测数据集转换为 ms-SWIFT grounding 格式，用于微调 Qwen VL / Qwen3 VL / Qwen3.5 VL 等主流 VLM 模型。

---

## 完整转换流程

```
YOLO 原始数据 (datasets/YOLO/)
     │
     │  Step 1: main.py
     ▼
┌──────────────────────────────────────────────────────┐
│  • 自动识别 YOLO 目录布局 (A / B)                      │
│  • 解析 YOLO .txt 标签                                │
│  • 坐标归一化 → 像素 [x1,y1,x2,y2]                     │
│  • 写入 annotations.jsonl (万能中间格式)                 │
│  • 写入 vlm_train.jsonl (ms-SWIFT grounding 格式)      │
│  • 拷贝图片到输出目录                                   │
└──────────────────────────────────────────────────────┘
     │
     ▼
datasets/VLM/<dataset_name>/
├── images/              ← 原图拷贝
├── annotations.jsonl    ← 结构化中间格式
├── vlm_train.jsonl      ← ms-SWIFT grounding 格式
└── stats.json           ← 数据集统计

     │
     │  Step 2: yolo_to_qwen.py (可选)
     ▼
qwen_train.jsonl         ← 重新生成的 ms-SWIFT grounding 格式
```

---

## 输出文件说明

| 文件 | 说明 | 必要性 |
|------|------|--------|
| `annotations.jsonl` | 结构化中间格式，包含 image_path、bbox、category 等所有信息 | ✅ 扩展用 |
| `vlm_train.jsonl` | ms-SWIFT grounding 格式，Step 1 自动生成 | ✅ 微调用 |
| `qwen_train.jsonl` | 同 grounding 格式，通过 yolo_to_qwen.py 生成 | ✅ 微调用 |
| `images/` | 原图拷贝目录 | ✅ 微调用（需要图片） |
| `stats.json` | 数据集统计（图片数、标注数、bbox 分布等） | 可选 |

---

## 数据格式

### annotations.jsonl（中间格式）

所有转换的起点，包含完整的结构化标注信息：

```json
{
  "image_path": "images/xxx.jpg",
  "width": 850,
  "height": 850,
  "annotations": [
    {
      "category_id": 0,
      "category_name": "safety_vest",
      "category_cn": "反光衣",
      "bbox": [126, 57, 698, 767]
    }
  ]
}
```

### vlm_train.jsonl / qwen_train.jsonl（ms-SWIFT grounding 格式）

可用于 ms-SWIFT 微调的最终格式：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "你是一个目标识别专家，你可以识别到图片中存在的物体，并可以给出相应的位置。"
    },
    {
      "role": "user",
      "content": "<image>请检测图像中的所有目标，类别包括：safety_vest, no_safety_vest。输出bbox和label。"
    },
    {
      "role": "assistant",
      "content": "<ref-object>safety_vest<bbox>[126.0,57.0,698.0,767.0]<ref-object>no_safety_vest<bbox>[200.0,100.0,400.0,300.0]"
    }
  ],
  "images": ["images/xxx.jpg"],
  "objects": {
    "ref": ["safety_vest", "no_safety_vest"],
    "bbox": [[126.0, 57.0, 698.0, 767.0], [200.0, 100.0, 400.0, 300.0]]
  }
}
```

**字段说明：**

| 字段 | 说明 |
|------|------|
| `messages[].role=system` | 系统提示词 |
| `messages[].role=user` | 用户消息，`<image>` 占位符 + 指令文本 |
| `messages[].role=assistant` | 回复内容，`<ref-object>label<bbox>[x1,y1,x2,y2]` 格式 |
| `images` | 图片路径数组（相对路径，ms-SWIFT 根目录拼接） |
| `objects.ref` | 类别名称数组，与 bbox 一一对应 |
| `objects.bbox` | 边界框坐标数组（浮点数） |

---

## 使用方法

### Step 1：一键转换（推荐）

运行 `main.py`，同时输出 `annotations.jsonl` + `vlm_train.jsonl` + 图片拷贝：

```bash
# 基本用法（会拷贝图片，处理所有数据集）
PYTHONPATH=. python converter/main.py \
    --input datasets/YOLO \
    --output datasets/VLM

# 只处理指定的数据集（当 YOLO 文件夹包含多个子数据集时）
PYTHONPATH=. python converter/main.py \
    --input datasets/YOLO \
    --output datasets/VLM \
    --select Safety_Vests_Detection_Dataset_YOLO

# 只处理另一个数据集
PYTHONPATH=. python converter/main.py \
    --input datasets/YOLO \
    --output datasets/VLM \
    --select vest_v1i_yolov11

# 不拷贝图片（仅生成标注文件）
PYTHONPATH=. python converter/main.py \
    --input datasets/YOLO \
    --output datasets/VLM \
    --no-copy-images

# 指定自定义类别映射
PYTHONPATH=. python converter/main.py \
    --input datasets/YOLO \
    --output datasets/VLM \
    --label-map converter/label_map.yaml
```

### Step 2：重新生成 ms-SWIFT 格式（可选）

已有 `annotations.jsonl`，想重新生成 grounding 格式或批量转换：

```bash
# 单个数据集
PYTHONPATH=. python converter/yolo_to_qwen.py \
    --input datasets/VLM/Safety_Vests_Detection_Dataset_YOLO \
    --output datasets/VLM/Safety_Vests_Detection_Dataset_YOLO/qwen_train.jsonl

# 自定义 system prompt
PYTHONPATH=. python converter/yolo_to_qwen.py \
    --input datasets/VLM/Safety_Vests_Detection_Dataset_YOLO \
    --output datasets/VLM/Safety_Vests_Detection_Dataset_YOLO/qwen_train.jsonl \
    --system-prompt "你是一个安全检测助手"

# 批量转换所有子数据集
for dir in datasets/VLM/*/; do
    name=$(basename "$dir")
    PYTHONPATH=. python converter/yolo_to_qwen.py \
        --input "$dir" \
        --output "$dir/qwen_train.jsonl"
done
```

---

## 支持的 YOLO 目录布局

自动识别两种布局，无需手动指定：

**布局A** — `images/` + `labels/` 并列
```
dataset/
├── data.yaml
├── images/
│   ├── train/, val/, test/
└── labels/
    ├── train/, val/, test/
```

**布局B** — `train/val/test` 内含 `images/` + `labels/`
```
dataset/
├── data.yaml
├── train/images/, train/labels/
├── val/images/, val/labels/
└── test/images/, test/labels/
```

---

## 类别映射

通过 `label_map.yaml` 配置 category_id → 类别名称/中文的映射：

```yaml
0:
  name: safety_vest
  cn: 反光衣
1:
  name: no_safety_vest
  cn: 未穿反光衣
```

> **注意**：类别名称应与 data.yaml 中的 names 一致。如果 data.yaml 中类别名不规范（如 `vest_v1i_yolov11` 的 `-` 和 `vest`），建议先修改 data.yaml 后再进行转换。

---

## 模块说明

| 文件 | 职责 |
|------|------|
| `main.py` | 命令行入口，YOLO → annotations.jsonl + vlm_train.jsonl + 图片 |
| `yolo_to_qwen.py` | annotations.jsonl → qwen_train.jsonl（重新生成 grounding 格式） |
| `yolo_parser.py` | 自动识别两种 YOLO 目录布局，解析 .txt 标签文件 |
| `transform.py` | YOLO 归一化坐标 (cx,cy,w,h) → 像素 [x1,y1,x2,y2] |
| `label_mapper.py` | category_id → name/cn 映射 |
| `instruction_generator.py` | 多模板 instruction，自动注入类别名称 |
| `cv_writer.py` | 输出 annotations.jsonl |
| `vlm_writer.py` | 输出 ms-SWIFT grounding 格式 |
| `stats.py` | 收集并输出数据集统计信息 |

---

## ms-SWIFT 训练示例

转换完成后，可通过以下方式启动训练（以 Qwen3 VL 为例）：

```bash
# 单卡训练
swift sft \
    --model qwen3-vl-4b \
    --dataset datasets/VLM/Safety_Vests_Detection_Dataset_YOLO/vlm_train.jsonl \
    --output_dir output

# 多卡训练
NPROC_PER_NODE=8 swift sft \
    --model qwen3-vl-4b \
    --dataset datasets/VLM/Safety_Vests_Detection_Dataset_YOLO/vlm_train.jsonl \
    --output_dir output
```

---

## 依赖

```
Pillow
PyYAML
Python 3.8+
```

---

## 扩展到其他 VLM 格式

`annotations.jsonl` 是万能中间格式。如需适配其他 VLM 格式（如 LLaVA、InternVL 等）：

1. 在 `vlm_writer.py` 中新增对应的 Writer 类
2. 新建对应的转换脚本，读取 `annotations.jsonl` 输出目标格式

一切扩展都围绕 `annotations.jsonl` 展开。
