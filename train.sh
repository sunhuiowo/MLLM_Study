#!/bin/bash
#===============================================================================
# Qwen3.5 LoRA 微调训练脚本
# 支持：Mac MPS / Linux CUDA / CPU 三平台
# 新增：gradient_checkpointing, flash_attn, ROOT_IMAGE_DIR 自动推导
#===============================================================================

set -e

echo "==============================================================================="
echo "🚀 Qwen3.5 LoRA 微调训练"
echo "==============================================================================="

#-------------------------------------------------------------------------------
# 平台检测与配置
#-------------------------------------------------------------------------------
PLATFORM="unknown"

if command -v nvidia-smi &> /dev/null && nvidia-smi >/dev/null 2>&1; then
    PLATFORM="cuda"
    echo "🐧 检测到 Linux/CUDA 平台"
elif system_profiler SPDisplaysDataType 2>/dev/null | grep -q "Apple Silicon"; then
    PLATFORM="mac"
    echo "🍎 检测到 Mac/Apple Silicon 平台"
else
    PLATFORM="cpu"
    echo "⚠️ 仅 CPU 模式，训练会很慢"
fi

#-------------------------------------------------------------------------------
# 平台特定配置
#-------------------------------------------------------------------------------
if [ "$PLATFORM" = "cuda" ]; then
    # Linux/CUDA 配置
    export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
    export NPROC_PER_NODE=1
    export CUDA_VISIBLE_DEVICES=0
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_DISABLE=1
    
    # 训练参数
    TORCH_DTYPE="bfloat16"
    BATCH_SIZE=4
    DATASET_NUM_PROC=4
    DATALOADER_NUM_WORKERS=4
    DEEPSPEED="zero2"
    MAX_PIXELS=1003520
    GRADIENT_CHECKPOINTING="true"
    
elif [ "$PLATFORM" = "mac" ]; then
    # Mac 配置
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
    export TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    
    # 训练参数（Mac 优化）
    TORCH_DTYPE="float16"
    BATCH_SIZE=1
    DATASET_NUM_PROC=1
    DATALOADER_NUM_WORKERS=0
    DEEPSPEED=""
    MAX_PIXELS=451584
    GRADIENT_CHECKPOINTING="true"
    
else
    # CPU 配置
    TORCH_DTYPE="float32"
    BATCH_SIZE=1
    DATASET_NUM_PROC=1
    DATALOADER_NUM_WORKERS=0
    DEEPSPEED=""
    MAX_PIXELS=200704
    GRADIENT_CHECKPOINTING="false"  # CPU 不需要
fi

#-------------------------------------------------------------------------------
# WandB 配置
#-------------------------------------------------------------------------------
export WANDB_PROJECT="qwen3.5-sft"
export WANDB_WATCH="false"
export WANDB_LOG_MODEL="false"

#-------------------------------------------------------------------------------
# 🎯 训练参数配置区（可修改）
#-------------------------------------------------------------------------------

# 模型配置
MODEL="./models/Qwen/Qwen3___5-0___8B"
# MODEL="./models/Qwen/Qwen3___5-27B"  # 👈 27B 模型取消注释

OUTPUT_DIR="./output/Qwen3.5-0.8B"
# OUTPUT_DIR="./output/Qwen3.5-27B"  # 👈 27B 输出目录

# LoRA 配置
LORA_RANK=8
# LORA_RANK=16  # 👈 27B 建议增大

LORA_ALPHA=32
TARGET_MODULES="all-linear"

# 训练配置
LEARNING_RATE=1e-4
NUM_EPOCHS=1
WARMUP_RATIO=0.05
MAX_LENGTH=2048

# Batch 配置
GRADIENT_ACCUMULATION_STEPS=1
# GRADIENT_ACCUMULATION_STEPS=8  # 👈 27B 建议增大等效 batch

# 保存配置
SAVE_STEPS=50
EVAL_STEPS=50
SAVE_TOTAL_LIMIT=2
LOGGING_STEPS=5

# 数据处理
SPLIT_RATIO=0.01
LOAD_FROM_CACHE="true"
GROUP_BY_LENGTH="true"
TORCH_COMPILE="false"

# Qwen3.5 专用
ADD_NON_THINKING_PREFIX="true"
LOSS_SCALE="ignore_empty_think"

# 日志配置
REPORT_TO="wandb"
RUN_NAME="qwen3.5-sft-$(date +%Y%m%d-%H%M%S)"

# 数据集（支持多个）
# DATASETS=(
#     "AI-ModelScope/alpaca-gpt4-data-zh#500"
#     "AI-ModelScope/alpaca-gpt4-data-en#500"
#     "swift/self-cognition#500"
#     "AI-ModelScope/LaTeX_OCR:human_handwrite#2000"
# )
DATASETS=(
    "./datasets/VLM/Safety_Vests_Detection_Dataset_YOLO"
)

#-------------------------------------------------------------------------------
# 🔧 ROOT_IMAGE_DIR 自动推导（修复图片路径问题）
#-------------------------------------------------------------------------------
auto_set_root_image_dir() {
    for ds in "${DATASETS[@]}"; do
        # 跳过远程数据集
        if [[ ! "$ds" =~ ^(\./|/|~) ]]; then
            continue
        fi
        
        # 解析路径
        local ds_path
        if [[ "$ds" =~ ^~ ]]; then
            ds_path="${HOME}${ds:1}"
        else
            ds_path="$ds"
        fi
        
        local root_image_dir=""
        if [ -d "$ds_path" ]; then
            # 情况1: 数据集是目录
            root_image_dir="$(cd "$ds_path" && pwd)"
        elif [ -f "$ds_path" ]; then
            # 情况2: 数据集是文件，取父目录
            root_image_dir="$(cd "$(dirname "$ds_path")" && pwd)"
        else
            continue
        fi
        
        # 设置环境变量
        export ROOT_IMAGE_DIR="$root_image_dir"
        echo "🔧 设置 ROOT_IMAGE_DIR=$root_image_dir"
        
        # 验证 images 目录
        if [ -d "$root_image_dir/images" ]; then
            local img_count=$(find "$root_image_dir/images" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) | wc -l)
            echo "✅ 找到 images 目录，共 $img_count 张图片"
        else
            echo "⚠️ 未找到 images 目录: $root_image_dir/images"
        fi
        break
    done
}

#-------------------------------------------------------------------------------
# 创建输出目录
#-------------------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR"
echo "✅ 输出目录：$OUTPUT_DIR"

#-------------------------------------------------------------------------------
# 自动设置 ROOT_IMAGE_DIR
#-------------------------------------------------------------------------------
auto_set_root_image_dir

#-------------------------------------------------------------------------------
# 构建数据集参数
#-------------------------------------------------------------------------------
DATASET_ARGS=""
for ds in "${DATASETS[@]}"; do
    DATASET_ARGS="$DATASET_ARGS --dataset $ds"
done

#-------------------------------------------------------------------------------
# DeepSpeed 参数
#-------------------------------------------------------------------------------
DEEPSPEED_ARG=""
if [ -n "$DEEPSPEED" ]; then
    DEEPSPEED_ARG="--deepspeed $DEEPSPEED"
fi

#-------------------------------------------------------------------------------
# 训练命令
#-------------------------------------------------------------------------------
echo ""
echo "📋 训练命令:"
echo "-------------------------------------------------------------------------------"

swift sft \
    --model "$MODEL" \
    --tuner_type lora \
    $DATASET_ARGS \
    --torch_dtype "$TORCH_DTYPE" \
    --max_pixels "$MAX_PIXELS" \
    --max_length "$MAX_LENGTH" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --target_modules "$TARGET_MODULES" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --per_device_eval_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --num_train_epochs "$NUM_EPOCHS" \
    --warmup_ratio "$WARMUP_RATIO" \
    --output_dir "$OUTPUT_DIR" \
    --save_steps "$SAVE_STEPS" \
    --eval_steps "$EVAL_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --logging_steps "$LOGGING_STEPS" \
    --dataset_num_proc "$DATASET_NUM_PROC" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --load_from_cache_file "$LOAD_FROM_CACHE" \
    --group_by_length "$GROUP_BY_LENGTH" \
    --torch_compile "$TORCH_COMPILE" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --add_non_thinking_prefix "$ADD_NON_THINKING_PREFIX" \
    --loss_scale "$LOSS_SCALE" \
    --split_dataset_ratio "$SPLIT_RATIO" \
    $DEEPSPEED_ARG \
    --report_to "$REPORT_TO" \
    --run_name "$RUN_NAME"

echo "-------------------------------------------------------------------------------"
echo ""

#-------------------------------------------------------------------------------
# 完成
#-------------------------------------------------------------------------------
echo "==============================================================================="
echo "✅ 训练完成!"
echo "📁 模型保存在：$OUTPUT_DIR"
echo "📊 WandB 面板：https://wandb.ai/$WANDB_PROJECT"
echo "==============================================================================="