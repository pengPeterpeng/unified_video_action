#!/bin/bash
#SBATCH --job-name=uva_umi_multi_act
#SBATCH --nodes=1
#SBATCH --time=96:00:00            # 最长运行时间
#SBATCH --output=logs/%x_%j.out    # 保存日志文件
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

export WANDB_API_KEY=c06481c89b99d2873815e4c33858063b22d185e9
# export WANDB_MODE=dryrun
# export MUJOCO_GL=osmesa
# export OMP_NUM_THREADS=4
export HYDRA_FULL_ERROR=1

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
    --config-name=uva_umi_multi.yaml \
    model.policy.autoregressive_model_params.pretrained_model_path=checkpoints/uva_umi_multitask_video_action/checkpoints/epoch_0021-val_action_l2_distances_0.075.ckpt \
    model.policy.action_model_params.predict_action=True \
    model.policy.use_proprioception=True \
    model.policy.predict_proprioception=True \
    model.policy.shift_action=False \
    model.policy.different_history_freq=True \
    model.policy.optimizer.learning_rate=1e-4 \
    task.dataset.dataset_root_dir=./data/zarr \
    logging.project=uva_umi_multi \
    hydra.run.dir="checkpoints/uva_umi_multitask_video_action_ablation"