#!/bin/bash
#SBATCH --job-name=uva_train
#SBATCH --nodes=1
#SBATCH --ntasks=8                  # 启动 8 个任务（等于进程数）
#SBATCH --gpus=8                    # 请求 8 块 GPU
#SBATCH --cpus-per-task=4          # 每个进程分配 4 个 CPU（根据模型大小和数据处理需求调整）
#SBATCH --mem=128G                  # 总内存（也可使用 --mem-per-cpu）
#SBATCH --time=48:00:00            # 最长运行时间
#SBATCH --output=logs/%x_%j.out    # 保存日志文件
#SBATCH --error=logs/%x_%j.err

export WANDB_API_KEY=c06481c89b99d2873815e4c33858063b22d185e9

# (可选) 载入你使用的环境模块或 conda 环境
# module load cuda/11.7
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate your_env_name

# 设置 NCCL 参数（推荐用于多卡通信优化）
export NCCL_DEBUG=info
# export NCCL_IB_DISABLE=0
# export NCCL_P2P_LEVEL=NVL
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# 启动任务
accelerate launch --num_processes=8 train.py \
    --config-dir=. \
    --config-name=uva_libero10.yaml \
    model.policy.action_model_params.predict_action=False \
    model.policy.selected_training_mode=video_model \
    model.policy.optimizer.learning_rate=1e-4 \
    logging.project=uva_libero \
    hydra.run.dir="checkpoints/uva_libero_video_model"
