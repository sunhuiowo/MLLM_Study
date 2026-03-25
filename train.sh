#!/bin/bash
#===============================================================================
# Qwen3.5 LoRA 微调训练脚本
# 支持：Mac MPS / Linux CUDA 双平台
#===============================================================================

set -e

echo "==============================================================================="
echo "🚀 Qwen3.5 LoRA 微调训练"
echo "==============================================================================="

#-------------------------------------------------------------------------------
# 平台检测与配置
#-------------------------------------------------------------------------------
PLATFORM="unknown"

if command -v nvidia-smi &> /dev/null; then
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
    
    # 训练参数
    TORCH_DTYPE="bfloat16"
    BATCH_SIZE=4
    DATASET_NUM_PROC=4
    DATALOADER_NUM_WORKERS=4
    DEEPSPEED="zero2"
    MAX_PIXELS=1003520
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
else
    # CPU 配置
    TORCH_DTYPE="float32"
    BATCH_SIZE=1
    DATASET_NUM_PROC=1
    DATALOADER_NUM_WORKERS=0
    DEEPSPEED=""
    MAX_PIXELS=200704
fi

#-------------------------------------------------------------------------------
# WandB 配置
#-------------------------------------------------------------------------------
export WANDB_PROJECT="qwen3.5-sft"
export WANDB_WATCH="false"
export WANDB_LOG_MODEL="false"

#-------------------------------------------------------------------------------
# 训练参数
#-------------------------------------------------------------------------------
MODEL="./models/Qwen/Qwen3___5-0___8B"
OUTPUT_DIR="./output/Qwen3.5-0.8B"
LEARNING_RATE=1e-4
NUM_EPOCHS=1
LORA_RANK=8
LORA_ALPHA=32
MAX_LENGTH=2048
WARMUP_RATIO=0.05
SAVE_STEPS=50
EVAL_STEPS=50
LOGGING_STEPS=5
SPLIT_RATIO=0.01
RUN_NAME="qwen3.5-sft-$(date +%Y%m%d-%H%M%S)"

# 数据集
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
# 创建输出目录
#-------------------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR"
echo "✅ 输出目录：$OUTPUT_DIR"

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
    --target_modules all-linear \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --per_device_eval_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps 1 \
    --learning_rate "$LEARNING_RATE" \
    --num_train_epochs "$NUM_EPOCHS" \
    --warmup_ratio "$WARMUP_RATIO" \
    --output_dir "$OUTPUT_DIR" \
    --save_steps "$SAVE_STEPS" \
    --eval_steps "$EVAL_STEPS" \
    --save_total_limit 2 \
    --logging_steps "$LOGGING_STEPS" \
    --dataset_num_proc "$DATASET_NUM_PROC" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --load_from_cache_file true \
    --group_by_length true \
    --torch_compile false \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --split_dataset_ratio "$SPLIT_RATIO" \
    $DEEPSPEED_ARG \
    --report_to wandb \
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