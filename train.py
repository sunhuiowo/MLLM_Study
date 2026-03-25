#!/usr/bin/env python3
"""
Qwen3.5 LoRA 微调训练脚本
支持：Mac MPS / Linux CUDA 双平台
"""

import os
import sys
import argparse
import subprocess
import shutil
import datetime
from pathlib import Path

# ========== 配置区域 ==========
DEFAULT_CONFIG = {
    # 模型配置
    "model": "./models/Qwen/Qwen3___5-0___8B",
    "model_author": None,
    "model_name": None,
    
    # 数据配置
    "dataset": ["./datasets/VLM/Safety_Vests_Detection_Dataset_YOLO"],
    "load_from_cache_file": True,
    "group_by_length": True,

    # 训练配置
    "tuner_type": "lora",
    "torch_dtype": "bfloat16",  # Mac 改为 float16
    "max_pixels": 1003520,
    "max_length": 2048,
    
    # LoRA 配置
    "lora_rank": 8,
    "lora_alpha": 32,
    "target_modules": "all-linear",
    
    # Batch 配置
    "per_device_train_batch_size": 4,  # Mac 改为 1
    "per_device_eval_batch_size": 4,   # Mac 改为 1
    "gradient_accumulation_steps": 1,
    
    # 优化器配置
    "learning_rate": 1e-4,
    "num_train_epochs": 1,
    "warmup_ratio": 0.05,
    
    # 保存配置
    "output_dir": "./output/Qwen3.5-0.8B",
    "save_steps": 50,
    "eval_steps": 50,
    "save_total_limit": 2,
    
    # 日志配置
    "logging_steps": 5,
    
    # WandB 配置
    "report_to": "wandb",
    "run_name": None,  # 👈 改为 None，由代码动态生成
    
    # 数据处理
    "dataset_num_proc": 4,      # Mac 改为 1
    "dataloader_num_workers": 4, # Mac 改为 0
    "load_from_cache_file": True,
    "group_by_length": True,
    
    # 高级配置
    "deepspeed": None,  # Mac 改为 None, CUDA 默认 zero2
    "torch_compile": False,
    "gradient_checkpointing": True,  # 👈 新增：梯度检查点（节省显存）
    
    # Qwen3.5 专用
    "add_non_thinking_prefix": True,
    "loss_scale": "ignore_empty_think",
    "split_dataset_ratio": 0.01,
}


# ========== 辅助函数 ==========
def detect_platform():
    """检测运行平台"""
    import torch
    if torch.backends.mps.is_available():
        return "mac"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def apply_platform_defaults(args):
    """根据平台应用默认配置"""
    platform = detect_platform()
    
    if platform == "mac":
        print("🍎 检测到 Mac 平台，应用 MPS 优化配置")
        if args.torch_dtype == "bfloat16":
            args.torch_dtype = "float16"
        if args.per_device_train_batch_size > 1:
            args.per_device_train_batch_size = 1
        if args.per_device_eval_batch_size > 1:
            args.per_device_eval_batch_size = 1
        if args.dataloader_num_workers > 0:
            args.dataloader_num_workers = 0
        if args.dataset_num_proc > 1:
            args.dataset_num_proc = 1
        if args.deepspeed:
            args.deepspeed = None
        if args.max_pixels > 451584:
            args.max_pixels = 451584
            
    elif platform == "cpu":
        print("🖥️ 检测到 CPU 平台，应用低资源配置")
        args.torch_dtype = "float32"
        args.per_device_train_batch_size = 1
        args.per_device_eval_batch_size = 1
        args.dataloader_num_workers = 0
        args.dataset_num_proc = 1
        args.deepspeed = None
        args.max_pixels = 200704
        args.gradient_checkpointing = False  # CPU 不需要
        
    else:
        print("🐧 检测到 Linux/CUDA 平台，使用完整配置")
        # CUDA 平台默认启用 deepspeed zero2（如果未指定）
        if args.deepspeed is None:
            args.deepspeed = "zero2"
    
    return args


def build_command(args):
    """构建 swift sft 命令"""
    swift_cmd = shutil.which("swift") or "swift"
    print(f"🔧 使用 swift 命令：{swift_cmd}")
    
    cmd = [
        swift_cmd, "sft",
        "--model", args.model,
        "--tuner_type", args.tuner_type,
    ]
    
    # 数据集（支持多个）
    for ds in args.dataset:
        cmd.extend(["--dataset", ds])
    
    # 训练配置
    cmd.extend([
        "--torch_dtype", args.torch_dtype,
        "--max_pixels", str(args.max_pixels),
        "--max_length", str(args.max_length),
        "--lora_rank", str(args.lora_rank),
        "--lora_alpha", str(args.lora_alpha),
        "--target_modules", args.target_modules,
        "--per_device_train_batch_size", str(args.per_device_train_batch_size),
        "--per_device_eval_batch_size", str(args.per_device_eval_batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--learning_rate", str(args.learning_rate),
        "--num_train_epochs", str(args.num_train_epochs),
        "--warmup_ratio", str(args.warmup_ratio),
        "--output_dir", args.output_dir,
        "--save_steps", str(args.save_steps),
        "--eval_steps", str(args.eval_steps),
        "--save_total_limit", str(args.save_total_limit),
        "--logging_steps", str(args.logging_steps),
        "--dataset_num_proc", str(args.dataset_num_proc),
        "--dataloader_num_workers", str(args.dataloader_num_workers),
        "--load_from_cache_file", str(args.load_from_cache_file).lower(),
        "--group_by_length", str(args.group_by_length).lower(),
        "--torch_compile", str(args.torch_compile).lower(),
    ])
    
    # 👇 新增：梯度检查点配置
    if hasattr(args, 'gradient_checkpointing'):
        cmd.extend(["--gradient_checkpointing", str(args.gradient_checkpointing).lower()])
  
    
    # Qwen3.5 专用参数
    if args.add_non_thinking_prefix:
        cmd.extend(["--add_non_thinking_prefix", "true"])
    if args.loss_scale:
        cmd.extend(["--loss_scale", args.loss_scale])
    if args.split_dataset_ratio > 0:
        cmd.extend(["--split_dataset_ratio", str(args.split_dataset_ratio)])
    
    # 高级配置
    if args.deepspeed:
        cmd.extend(["--deepspeed", args.deepspeed])
    
    # WandB 配置
    if args.report_to:
        cmd.extend(["--report_to", args.report_to])
    if args.run_name:
        cmd.extend(["--run_name", args.run_name])
    
    # 模型元数据
    if args.model_author:
        cmd.extend(["--model_author", args.model_author])
    if args.model_name:
        cmd.extend(["--model_name", args.model_name])
    
    # 可选参数
    if args.resume_from_checkpoint:
        cmd.extend(["--resume_from_checkpoint", args.resume_from_checkpoint])
    
    return cmd


def set_environment(args):
    """设置环境变量"""
    platform = detect_platform()
    
    if platform == "cuda":
        # Linux/CUDA 环境
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["NCCL_IB_DISABLE"] = "1"
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NPROC_PER_NODE"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        # Mac / CPU 环境
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
    
    # WandB 配置
    os.environ["WANDB_PROJECT"] = "qwen3.5-sft"
    os.environ["WANDB_WATCH"] = "false"
    os.environ["WANDB_LOG_MODEL"] = "false"
    
    # 🔧 图片路径自动推导（修复版）
    if hasattr(args, 'dataset') and args.dataset:
        for ds in args.dataset:
            # 跳过远程数据集
            if not ds.startswith(("./", "/", "~")):
                continue
                
            ds_path = Path(ds).expanduser().resolve()
            
            if ds_path.is_dir():
                # 数据集是目录：ROOT_IMAGE_DIR = 目录本身
                root_image_dir = ds_path
            elif ds_path.is_file():
                # 数据集是文件：ROOT_IMAGE_DIR = 文件所在目录
                root_image_dir = ds_path.parent
            else:
                continue  # 路径不存在，跳过
            
            os.environ["ROOT_IMAGE_DIR"] = str(root_image_dir)
            print(f"🔧 设置 ROOT_IMAGE_DIR={root_image_dir}")
            
            # 调试信息
            images_dir = root_image_dir / "images"
            if images_dir.exists():
                img_count = len(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
                print(f"✅ 找到 images 目录，共 {img_count} 张图片")
            else:
                print(f"⚠️ 未找到 images 目录: {images_dir}")
            break


def check_environment():
    """检查训练环境"""
    print("🔍 检查训练环境...")
    print("=" * 60)
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        platform = detect_platform()
        if platform == "cuda":
            print(f"✅ CUDA: {torch.version.cuda}")
            print(f"✅ GPU 数量：{torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
        elif platform == "mac":
            print("✅ MPS: 可用 (Apple Silicon)")
        else:
            print("⚠️ 仅 CPU 模式，训练会很慢")
            
    except ImportError as e:
        print(f"❌ PyTorch 导入失败：{e}")
        return False
    
    required_packages = ["transformers", "peft", "accelerate", "swift"]
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"✅ {pkg}: 已安装")
        except ImportError:
            print(f"❌ {pkg}: 未安装")
            return False
    
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5 LoRA 微调训练脚本")
    
    # ========== 模型配置 ==========
    parser.add_argument("--model", type=str, default=DEFAULT_CONFIG["model"], help="模型名称或路径")
    parser.add_argument("--model_author", type=str, default=None, help="模型作者")
    parser.add_argument("--model_name", type=str, default=None, help="模型名称")
    
    # ========== 数据配置 ==========
    parser.add_argument("--dataset", type=str, nargs="+", default=DEFAULT_CONFIG["dataset"], help="数据集列表")
    parser.add_argument("--load_from_cache_file", type=bool, default=DEFAULT_CONFIG["load_from_cache_file"])
    parser.add_argument("--group_by_length", type=bool, default=DEFAULT_CONFIG["group_by_length"])
    
    # ========== 训练配置 ==========
    parser.add_argument("--tuner_type", type=str, default=DEFAULT_CONFIG["tuner_type"], choices=["lora", "full"])
    parser.add_argument("--torch_dtype", type=str, default=DEFAULT_CONFIG["torch_dtype"], choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max_pixels", type=int, default=DEFAULT_CONFIG["max_pixels"])
    parser.add_argument("--max_length", type=int, default=DEFAULT_CONFIG["max_length"])
    
    # ========== LoRA 配置 ==========
    parser.add_argument("--lora_rank", type=int, default=DEFAULT_CONFIG["lora_rank"])
    parser.add_argument("--lora_alpha", type=int, default=DEFAULT_CONFIG["lora_alpha"])
    parser.add_argument("--target_modules", type=str, default=DEFAULT_CONFIG["target_modules"])
    
    # ========== Batch 配置 ==========
    parser.add_argument("--per_device_train_batch_size", type=int, default=DEFAULT_CONFIG["per_device_train_batch_size"])
    parser.add_argument("--per_device_eval_batch_size", type=int, default=DEFAULT_CONFIG["per_device_eval_batch_size"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=DEFAULT_CONFIG["gradient_accumulation_steps"])
    
    # ========== 优化器配置 ==========
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--num_train_epochs", type=int, default=DEFAULT_CONFIG["num_train_epochs"])
    parser.add_argument("--warmup_ratio", type=float, default=DEFAULT_CONFIG["warmup_ratio"])
    
    # ========== 保存配置 ==========
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--save_steps", type=int, default=DEFAULT_CONFIG["save_steps"])
    parser.add_argument("--eval_steps", type=int, default=DEFAULT_CONFIG["eval_steps"])
    parser.add_argument("--save_total_limit", type=int, default=DEFAULT_CONFIG["save_total_limit"])
    
    # ========== 日志配置 ==========
    parser.add_argument("--logging_steps", type=int, default=DEFAULT_CONFIG["logging_steps"])
    
    # ========== WandB 配置 ==========
    parser.add_argument("--report_to", type=str, default=DEFAULT_CONFIG["report_to"], choices=["wandb", "tensorboard", "all", "none"])
    parser.add_argument("--run_name", type=str, default=DEFAULT_CONFIG["run_name"])
    
    # ========== 数据处理 ==========
    parser.add_argument("--dataset_num_proc", type=int, default=DEFAULT_CONFIG["dataset_num_proc"])
    parser.add_argument("--dataloader_num_workers", type=int, default=DEFAULT_CONFIG["dataloader_num_workers"])
    
    # ========== 高级配置 ==========
    parser.add_argument("--deepspeed", type=str, default=DEFAULT_CONFIG["deepspeed"], choices=["zero2", "zero3", None])
    parser.add_argument("--torch_compile", type=bool, default=DEFAULT_CONFIG["torch_compile"])
    parser.add_argument("--gradient_checkpointing", type=bool, default=DEFAULT_CONFIG["gradient_checkpointing"])  # 👈 新增
    
    # ========== Qwen3.5 专用 ==========
    parser.add_argument("--add_non_thinking_prefix", type=bool, default=DEFAULT_CONFIG["add_non_thinking_prefix"])
    parser.add_argument("--loss_scale", type=str, default=DEFAULT_CONFIG["loss_scale"])
    parser.add_argument("--split_dataset_ratio", type=float, default=DEFAULT_CONFIG["split_dataset_ratio"])
    
    # ========== 可选参数 ==========
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true", help="只打印命令，不执行")
    parser.add_argument("--auto_platform", action="store_true", help="自动根据平台调整配置")
    
    args = parser.parse_args()
    
    # ========== 开始执行 ==========
    print("\n" + "=" * 60)
    print("🚀 Qwen3.5 LoRA 微调训练")
    print("=" * 60 + "\n")
    
    # 👈 动态生成 run_name（如果未手动指定）
    if args.run_name is None or args.run_name == DEFAULT_CONFIG["run_name"]:
        args.run_name = f"qwen3.5-sft-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        print(f"📊 自动生成 run_name: {args.run_name}")
    
    # 1. 自动平台适配
    if args.auto_platform:
        args = apply_platform_defaults(args)
    
    # 2. 检查环境
    if not check_environment():
        print("\n❌ 环境检查失败，请先安装依赖")
        sys.exit(1)
    
    # 3. 设置环境变量
    set_environment(args)
    
    # 4. 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"✅ 输出目录：{args.output_dir}")
    
    # 5. 构建命令
    cmd = build_command(args)
    
    print("\n📋 训练命令:")
    print(" " + " \\\n ".join([""] + cmd))
    print()
    
    # 6. 显示 WandB 信息
    if args.report_to in ["wandb", "all"]:
        print(f"📊 WandB 项目：{os.environ.get('WANDB_PROJECT', 'qwen3.5-sft')}")
        print(f"📊 WandB 运行名：{args.run_name}")
        print(f"📊 查看面板：https://wandb.ai/  ")
        print()
    
    # 7. Dry Run 模式
    if args.dry_run:
        print("\n⚠️ Dry Run 模式，未执行训练")
        sys.exit(0)
    
    # 8. 执行训练
    print("🔥 开始训练...")
    print("=" * 60 + "\n")
    
    try:
        process = subprocess.run(cmd, check=True)
        
        if process.returncode == 0:
            print("\n" + "=" * 60)
            print("✅ 训练完成!")
            print(f"📁 模型保存在：{args.output_dir}")
            print("=" * 60)
        else:
            print("\n❌ 训练异常退出")
            sys.exit(1)
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练失败：{e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断训练")
        sys.exit(0)


if __name__ == "__main__":
    main()