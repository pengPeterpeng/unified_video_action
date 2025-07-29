import os
import torch
import dill
import hydra
import numpy as np
from tqdm import tqdm
from omegaconf import open_dict
from omegaconf import OmegaConf
import click
from pathlib import Path

from unified_video_action.policy.base_image_policy import BaseImagePolicy
from unified_video_action.workspace.base_workspace import BaseWorkspace
from unified_video_action.common.pytorch_util import dict_apply
from unified_video_action.dataset.umi_multi_dataset import UmiMultiDataset
from unified_video_action.utils.data_utils import resize_image, unnormalize_future_action
from unified_video_action.eval.eval import test_video_fvd, test_action_l2, test_action_l2_with_meta


class StaticEvaluator:
    def __init__(self, ckpt_path: str, device: str, output_dir: str):
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        
        self.ckpt_path = ckpt_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.ckpt_path.endswith(".ckpt"):
            self.ckpt_path = os.path.join(self.ckpt_path, "checkpoints", "latest.ckpt")
        
        # Load checkpoint
        payload = torch.load(
            open(self.ckpt_path, "rb"), map_location="cpu", pickle_module=dill
        )
        self.cfg = payload["cfg"]  # training configuration and dataset configurations included
        # import ipdb; ipdb.set_trace()
        # Modify config for evaluation
        with open_dict(self.cfg):
            self.cfg.training.n_gpus = 1
            self.cfg.training.mixed_precision = None 

        # Setup workspace and model
        cls = hydra.utils.get_class(self.cfg.model._target_)
        self.workspace = cls(self.cfg, output_dir=output_dir)
        self.workspace: BaseWorkspace
        self.workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
        # Get policy
        self.policy: BaseImagePolicy = self.workspace.model
        if self.cfg.training.use_ema:
            self.policy = self.workspace.ema_model
            print("Using EMA model")

        
        self.policy.eval().to(self.device)

        # Initialize test dataset
        self.test_dataset = self._init_test_dataset()
        self.test_dataloader = self.test_dataset.get_dataloader()

    def _init_test_dataset(self):
        """Initialize test dataset using the same split as training
        for umi dataset
        """
        dataset: UmiMultiDataset
        dataset = hydra.utils.instantiate(self.cfg.task.dataset)
        return dataset.split_unused_episodes()

    def evaluate(self):
        """Run evaluation on test dataset"""
        print(f"\nStarting evaluation on {len(self.test_dataloader)} batches...")

        # results = test_action_l2(
        #     cfg=self.cfg,
        #     model=self.policy,
        #     loader=self.test_dataloader,
        #     it=0,  # static evaluation iteration
        #     output_dir=str(self.output_dir),
        #     device=self.device,
        #     name_label="static_eval_"
        # )

        results = test_action_l2_with_meta(
            cfg=self.cfg,
            model=self.policy,
            loader=self.test_dataloader,
            it=0,  # static evaluation iteration
            output_dir=str(self.output_dir),
            device=self.device,
            name_label="static_eval_"
        )

        # Save results
        output_path = Path(self.workspace.output_dir) / 'eval_results.pt'
        torch.save(results, output_path)

        # # Print results
        # print("\nEvaluation Results:")
        # for key, value in results.items():
        #     if isinstance(value, (int, float)):
        #         print(f"{key}: {value:.4f}")
        #     else:
        #         print(f"{key}: {value}")
        
        return results

@click.command()
@click.option("--input", "-i", required=True, help="Path to checkpoint")
@click.option("--device", default="cuda", help="Device to run on")
@click.option("--output_dir", required=True, help="Directory to save results")
def main(input, device, output_dir):
    
    evaluator = StaticEvaluator(input, device, output_dir)
    evaluator.evaluate()

if __name__ == "__main__":
    main()