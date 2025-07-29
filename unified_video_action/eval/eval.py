import sys

sys.path.extend([".", "src"])
import torch
import os
from einops import rearrange
import torch.nn.functional as F
import wandb

from unified_video_action.fvd.fvd import get_fvd_logits, frechet_distance
from unified_video_action.fvd.download import load_i3d_pretrained
from unified_video_action.common.pytorch_util import dict_apply
from unified_video_action.utils.utils import AverageMeter
from unified_video_action.utils.data_utils import resize_image
from unified_video_action.utils.data_utils import (
    normalize_action,
    normalize_obs,
    unnormalize_future_action,
)
from unified_video_action.utils.data_utils import (
    process_data,
    save_image_grid,
    get_vae_latent,
    get_trajectory,
    decode_from_sample_autoregressive,
)
from unified_video_action.utils.language_model import extract_text_features
import h5py
import numpy as np


def prepare_data_predict_action(
    cfg, x, actions, model, T, device, language_goal=None, eval=False
):
    ## normalize actions and observations
    nactions = normalize_action(
        normalizer=model.normalizer,
        normalizer_type=model.normalizer_type,
        actions=actions,
    )
    x = normalize_obs(
        normalizer=model.normalizer, normalizer_type=model.normalizer_type, batch=x
    )

    ## process data
    x, proprioception_input, _ = process_data(
        x,
        task_name=cfg.task.name,
        eval=eval,
        use_proprioception=cfg.model.policy.use_proprioception,
        different_history_freq=cfg.model.policy.different_history_freq,
    )

    real, _, c, latent_size, proprioception_input = get_vae_latent(
        x, model.vae_model, eval=True, proprioception_input=proprioception_input
    )
    history_trajectory, trajectory = get_trajectory(
        nactions,
        T,
        cfg.model.policy.shift_action,
        use_history_action=cfg.model.policy.use_history_action,
    )

    text_latents = None
    if cfg.task.dataset.language_emb_model is not None:
        if "umi" in cfg.task.name:
            text_latents = language_goal
        elif "libero" in cfg.task.name:
            if cfg.task.dataset.language_emb_model == "clip":
                text_tokens = {
                    "input_ids": language_goal[:, 0].long()[:, 0],
                    "attention_mask": language_goal[:, 0].long()[:, 1],
                }
                text_latents = extract_text_features(
                    model.text_model,
                    text_tokens,
                    language_emb_model=cfg.task.dataset.language_emb_model,
                )
            elif cfg.task.dataset.language_emb_model == "flant5":
                text_tokens = language_goal[:, 0].long()
                text_latents = extract_text_features(
                    model.text_model,
                    text_tokens,
                    language_emb_model=cfg.task.dataset.language_emb_model,
                ).float()
            else:
                raise NotImplementedError
    return (
        x,
        real,
        latent_size,
        c,
        text_latents,
        history_trajectory,
        trajectory,
        proprioception_input,
    )


def test_video_fvd(
    cfg, model, loader, it, output_dir, device, name_label="", plot_actions=False
):
    losses = dict()
    losses["fvd"] = AverageMeter()

    i3d = load_i3d_pretrained(device)
    real_embeddings = []
    pred_embeddings = []

    reals = []
    predictions = []

    n_examples = 4

    with torch.no_grad():
        for n, batch in enumerate(loader):
            if n % 10 == 0:
                print("test_video_fvd", n, len(loader))

            x = batch
            if n >= n_examples:
                break

            x = dict_apply(x, lambda x: x.to(device, non_blocking=True))
            actions = x["action"]

            if cfg.model.policy.use_history_action:
                x = dict_apply(x, lambda x: x[:, 1:])

            x = resize_image(cfg, x)

            B, T, C, H, W = x["obs"]["image"].size()
            k = min(n_examples, B)

            actions = actions[:k]
            x = dict_apply(x, lambda x: x[:k])

            if cfg.task.dataset.language_emb_model is not None:
                if "language" in x["obs"]:
                    language_goal = x["obs"]["language"]
                    del x["obs"]["language"]
                elif "language_latents" in x:
                    language_goal = x["language_latents"]
                    del x["language_latents"]
                else:
                    raise NotImplementedError
            else:
                language_goal = None

            (
                x,
                real,
                _,
                c,
                text_latents,
                history_trajectory,
                trajectory,
                proprioception_input,
            ) = prepare_data_predict_action(
                cfg, x, actions, model, T, device, language_goal=language_goal
            )

            z, act_out = model.model.sample_tokens(
                bsz=k,
                cond=c,
                text_latents=text_latents,
                num_iter=cfg.model.policy.autoregressive_model_params.num_iter,
                cfg=cfg.model.policy.autoregressive_model_params.cfg,
                cfg_schedule=cfg.model.policy.autoregressive_model_params.cfg_schedule,
                temperature=cfg.model.policy.autoregressive_model_params.temperature,
                history_nactions=history_trajectory,
                nactions=trajectory,
                proprioception_input=proprioception_input,
                task_mode="full_dynamic_model",
            )
            pred = decode_from_sample_autoregressive(model.vae_model, z / 0.2325)
            pred = pred.clamp(-1, 1).cpu()

            pred = 1 + rearrange(pred, "(b t) c h w -> b t h w c", b=k)
            real = (1 + rearrange(real, "b c t h w -> b t h w c")).cpu()

            pred = pred * 127.5
            pred = pred.type(torch.uint8)

            real = real * 127.5
            real = real.type(torch.uint8)

            x = (1 + x) * 127.5  # b c t h w
            x = x.type(torch.uint8).cpu()

            if len(predictions) < n_examples:
                reals.append(
                    torch.cat(
                        [
                            x[:, :, : x.size(2) // 2],
                            rearrange(real, "b t h w c -> b c t h w"),
                        ],
                        dim=2,
                    )
                )
                predictions.append(
                    torch.cat(
                        [
                            x[:, :, : x.size(2) // 2],
                            rearrange(pred, "b t h w c -> b c t h w"),
                        ],
                        dim=2,
                    )
                )

            if real.shape[1] < 16:
                pred = pred.repeat_interleave(repeats=4, dim=1)
                real = real.repeat_interleave(repeats=4, dim=1)

            pred_embeddings.append(get_fvd_logits(pred.numpy(), i3d=i3d, device=device))
            real_embeddings.append(get_fvd_logits(real.numpy(), i3d=i3d, device=device))

    log_data = dict()
    reals = torch.cat(reals)
    predictions = torch.cat(predictions)

    real_embeddings = torch.cat(real_embeddings)
    pred_embeddings = torch.cat(pred_embeddings)
    fvd = frechet_distance(
        pred_embeddings.clone().detach(), real_embeddings.clone().detach()
    )
    fvd = fvd.item()

    os.makedirs(output_dir + "/vis", exist_ok=True)
    real_vid = save_image_grid(
        reals.cpu().numpy(),
        os.path.join(output_dir, f"vis/{name_label}real_{it}.gif"),
        drange=[0, 255],
        grid_size=(reals.size(0) // 4, 4),
    )  # [4, 3, 8, 128, 128]
    pred_vid = save_image_grid(
        predictions.cpu().numpy(),
        os.path.join(output_dir, f"vis/{name_label}predicted_{it}.gif"),
        drange=[0, 255],
        grid_size=(predictions.size(0) // 4, 4),
    )  # [4, 3, 8, 128, 128]

    real_video = wandb.Video(os.path.join(output_dir, f"vis/{name_label}real_{it}.gif"))
    pred_video = wandb.Video(
        os.path.join(output_dir, f"vis/{name_label}predicted_{it}.mp4")
    )

    log_data[f"{name_label}video_fvd"] = fvd
    log_data[f"{name_label}real_img"] = real_video
    log_data[f"{name_label}predicted_img"] = pred_video

    return log_data


def test_action_l2(
    cfg,
    model,
    loader,
    it,
    output_dir,
    device,
    text_model=None,
    name_label="",
    plot_actions=False,
):
    action_l2_distances = []
    # import ipdb; ipdb.set_trace()
    with torch.no_grad():
        for n, batch in enumerate(loader):
            if n % 10 == 0:
                print("test_action_l2", n, len(loader))

            x = batch
            x = dict_apply(x, lambda x: x.to(device, non_blocking=True))
            actions = x["action"]

            if cfg.model.policy.use_history_action:
                x = dict_apply(x, lambda x: x[:, 1:])

            x = resize_image(cfg, x)

            B, T, C, H, W = x["obs"]["image"].size()

            if cfg.task.dataset.language_emb_model is not None:
                if "language" in x["obs"]:
                    language_goal = x["obs"]["language"]
                    del x["obs"]["language"]
                elif "language_latents" in x:
                    language_goal = x["language_latents"]
                    del x["language_latents"]
                else:
                    raise NotImplementedError
            else:
                language_goal = None

            (
                x,
                real,
                _,
                c,
                text_latents,
                history_trajectory,
                trajectory,
                proprioception_input,
            ) = prepare_data_predict_action(
                cfg, x, actions, model, T, device, language_goal=language_goal
            )

            z, act_out = model.model.sample_tokens(
                bsz=B,
                cond=c,
                text_latents=text_latents,
                num_iter=cfg.model.policy.autoregressive_model_params.num_iter,
                cfg=cfg.model.policy.autoregressive_model_params.cfg,
                cfg_schedule=cfg.model.policy.autoregressive_model_params.cfg_schedule,
                temperature=cfg.model.policy.autoregressive_model_params.temperature,
                history_nactions=history_trajectory,
                nactions=trajectory,
                proprioception_input=proprioception_input,
                task_mode="policy_model",
            )

            if cfg.model.policy.action_model_params.predict_action:
                act_out = unnormalize_future_action(
                    normalizer=model.normalizer,
                    normalizer_type=model.normalizer_type,
                    actions=act_out,
                )
                trajectory = unnormalize_future_action(
                    normalizer=model.normalizer,
                    normalizer_type=model.normalizer_type,
                    actions=trajectory,
                )

                ## calculate l2 distance between the predicted action and ground truth action
                l2_distance = torch.sqrt(
                    torch.sum((trajectory[:, :, :9] - act_out[:, :, :9]) ** 2, dim=-1)
                )
                action_l2_distances.append(l2_distance.mean())

            if cfg.training.debug:
                break

    log_data = dict()
    if cfg.model.policy.action_model_params.predict_action:
        log_data[f"{name_label}val_action_l2_distances"] = (
            torch.stack(action_l2_distances).mean().item()
        )

    return log_data


def test_action_l2_with_meta(
    cfg,
    model,
    loader,
    it,
    output_dir,
    device,
    text_model=None,
    name_label="",
    plot_actions=False,
):
    """Enhanced version of test_action_l2 with episode metadata"""
    action_l2_distances = []
    episode_info = []  # 存储每个episode的详细信息

    with torch.no_grad():
        for n, batch in enumerate(loader):
            if n % 10 == 0:
                print("test_action_l2", n, len(loader))
            # import ipdb; ipdb.set_trace()
            # 收集episode信息
            episode_meta = {
                'episode_idx': batch['ids'][0].item() if 'ids' in batch else n,
                'img_indices': batch['img_indices'].cpu().numpy() if 'img_indices' in batch else None,
                'dataset_name': batch['dataset_name'] if 'dataset_name' in batch else 'unknown'
            }

            x = batch
            x = dict_apply(x, lambda x: x.to(device, non_blocking=True))
            actions = x["action"]

            if cfg.model.policy.use_history_action:
                x = dict_apply(x, lambda x: x[:, 1:])

            x = resize_image(cfg, x)
            B, T, C, H, W = x["obs"]["image"].size()

            # 处理语言目标
            if cfg.task.dataset.language_emb_model is not None:
                if "language" in x["obs"]:
                    language_goal = x["obs"]["language"]
                    del x["obs"]["language"]
                elif "language_latents" in x:
                    language_goal = x["language_latents"]
                    del x["language_latents"]
                else:
                    raise NotImplementedError
            else:
                language_goal = None

            # 准备数据
            (x, real, _, c, text_latents, history_trajectory, trajectory, 
             proprioception_input,) = prepare_data_predict_action(
                cfg, x, actions, model, T, device, language_goal=language_goal
            )
            
            # 模型预测
            z, act_out = model.model.sample_tokens(
                bsz=B,
                cond=c,
                text_latents=text_latents,
                num_iter=cfg.model.policy.autoregressive_model_params.num_iter,
                cfg=cfg.model.policy.autoregressive_model_params.cfg,
                cfg_schedule=cfg.model.policy.autoregressive_model_params.cfg_schedule,
                temperature=cfg.model.policy.autoregressive_model_params.temperature,
                history_nactions=history_trajectory,
                nactions=trajectory,
                proprioception_input=proprioception_input,
                task_mode="policy_model",
            )

            if cfg.model.policy.action_model_params.predict_action:
                # 反归一化动作
                act_out = unnormalize_future_action(
                    normalizer=model.normalizer,
                    normalizer_type=model.normalizer_type,
                    actions=act_out,
                )
                trajectory = unnormalize_future_action(
                    normalizer=model.normalizer,
                    normalizer_type=model.normalizer_type,
                    actions=trajectory,
                )

                # 计算L2距离
                l2_distance = torch.sqrt(
                    torch.sum((trajectory[:, :, :9] - act_out[:, :, :9]) ** 2, dim=-1)
                )
                mean_l2 = l2_distance.mean().item()
                action_l2_distances.append(mean_l2)
                
                # vis img and predict results
                import ipdb; ipdb.set_trace()
                visualize_results_combined(x, act_out, trajectory, output_dir)
                
                # 添加性能指标到episode信息
                episode_meta.update({
                    'l2_distance': mean_l2,
                    'trajectory_length': T,
                    'predicted_actions': act_out.cpu().numpy(),
                    'ground_truth_actions': trajectory.cpu().numpy()
                })

            episode_info.append(episode_meta)

    # # 保存详细结果
    # results_path = os.path.join(output_dir, f'{name_label}eval_results.h5')
    # with h5py.File(results_path, 'w') as f:
    #     # 保存总体指标
    #     metrics = f.create_group('metrics')
    #     if cfg.model.policy.action_model_params.predict_action:
    #         metrics.create_dataset('mean_l2_distance', 
    #                              data=torch.stack(action_l2_distances).mean().item())
        
    #     # 保存每个episode的详细信息
    #     episodes = f.create_group('episodes')
    #     for i, ep_info in enumerate(episode_info):
    #         ep_group = episodes.create_group(f'episode_{i}')
    #         for key, value in ep_info.items():
    #             if value is not None:
    #                 if isinstance(value, (np.ndarray, list)):
    #                     ep_group.create_dataset(key, data=value, compression='gzip')
    #                 else:
    #                     ep_group.create_dataset(key, data=value)

    log_data = {
        f"{name_label}val_action_l2_distances": torch.stack(action_l2_distances).mean().item(),
        'episode_info': episode_info
    }

    return log_data


def visualize_results_combined(normalized_img, act_out, trajectory, output_dir: str="vis_results", idx : int=0):
    """
    Visualize RGB images and end-effector trajectory with rotation visualization
    Args:
        idx: index of the sample to visualize 
        output_dir: directory to save visualization results
    """
    import matplotlib.pyplot as plt
    import os
    from PIL import Image
    import io
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    images = []
    import ipdb; ipdb.set_trace()
    if normalized_img.shape[0] > 1:
        rgb_frames = normalized_img[idx].permute(1,0,2,3)  # (8, C, H, W)
        rgb_frames = ((rgb_frames + 1) * 127.5).type(torch.uint8).cpu().numpy()
        # import ipdb; ipdb.set_trace()
        # Get full trajectory and rotation data
        preds = act_out[idx][:, :3].cpu().numpy()  # (16, 3)
        gts = trajectory[idx][:, :3].cpu().numpy()  # (16, 3)
        # rot = sample['obs']['robot0_eef_rot_axis_angle'].cpu().numpy()  # (32, 3)

        # set dpi
        plt.rcParams['figure.dpi'] = 100
        
        # Process each timestep (16 frames total), half for pred results
        for t in range(len(preds)):
            # Only update image every 4 timesteps
            img_idx = t // 4 + 4 if t // 4 < len(rgb_frames) else -1  # show last 4 frames
            
            fig = plt.figure(figsize=(15, 15))
            
            # Left subplot: RGB image
            ax1 = fig.add_subplot(221)
            ax1.imshow(rgb_frames[img_idx].transpose(1,2,0))
            ax1.axis('off')
            ax1.set_title(f'Frame {img_idx+1}/{len(rgb_frames)}')
            
            # Right subplot: 3D trajectory with rotation
            ax2 = fig.add_subplot(222, projection='3d')
            
            # Plot historical trajectory
            if t > 0:
                ax2.plot(preds[:t, 0], preds[:t, 1], preds[:t, 2], 
                        color='gray', linestyle='-', alpha=0.5,
                        label='Historical Path')
            
            # Plot current position and orientation
            ax2.scatter(preds[t, 0], preds[t, 1], preds[t, 2], 
                    color='red', marker='o', s=100,
                    label='Current Position')
            
            # # Visualize rotation using arrows
            # arrow_length = 0.05  # Adjust this based on your scale
            # rot_mat = axis_angle_to_matrix(rot[t])  # You'll need to implement this
            
            # # Draw coordinate axes at current position
            # colors = ['r', 'g', 'b']  # x, y, z axes
            # for i in range(3):
            #     direction = rot_mat[:, i]
            #     ax2.quiver(pos[t, 0], pos[t, 1], pos[t, 2],
            #             direction[0] * arrow_length,
            #             direction[1] * arrow_length,
            #             direction[2] * arrow_length,
            #             color=colors[i], alpha=0.6)
            
            # Plot future trajectory
            if t < len(preds)-1:
                ax2.plot(preds[t:, 0], preds[t:, 1], preds[t:, 2], 
                        color='blue', linestyle='--', alpha=0.2,
                        label='Future Path')
            
            ax2.plot(gts[:t, 0], gts[:t, 1], gts[:t, 2], 
                        color='green', linestyle='-', alpha=0.7,
                        label='GT Path')
            
            # Set consistent view limits
            ax2.set_xlim([preds[:, 0].min()-0.1, preds[:, 0].max()+0.1])
            ax2.set_ylim([preds[:, 1].min()-0.1, preds[:, 1].max()+0.1])
            ax2.set_zlim([preds[:, 2].min()-0.1, preds[:, 2].max()+0.1])
            
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            ax2.set_title('End-effector Pose')
            ax2.legend()
            
            # xy plane
            ax3 = fig.add_subplot(223)
            ax3.plot(preds[:t+1, 0], preds[:t+1, 1], color='gray', linestyle='-', alpha=0.5)
            ax3.scatter(preds[t, 0], preds[t, 1], color='red', marker='o', s=100)
            if t < len(preds)-1:
                ax3.plot(preds[t:, 0], preds[t:, 1], 
                        color='blue', linestyle='--', alpha=0.2)
            ax3.plot(gts[:t, 0], gts[:t, 1], color='green', linestyle='-', alpha=0.7)
            ax3.set_xlim([preds[:, 0].min()-0.1, preds[:, 0].max()+0.1])
            ax3.set_ylim([preds[:, 1].min()-0.1, preds[:, 1].max()+0.1])

            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_title('XY Plane Projection')
            ax3.grid(True)
            # zx plane
            ax4 = fig.add_subplot(224)
            ax4.plot(preds[:t+1, 0], preds[:t+1, 2], color='gray', linestyle='-', alpha=0.5)
            ax4.scatter(preds[t, 0], preds[t, 2], color='red', marker='o', s=100)
            if t < len(preds)-1:
                ax4.plot(preds[t:, 0], preds[t:, 2], 
                        color='blue', linestyle='--', alpha=0.2)
            ax4.plot(gts[:t, 0], gts[:t, 2], color='green', linestyle='-', alpha=0.7)
            ax4.set_xlim([preds[:, 0].min()-0.1, preds[:, 0].max()+0.1])
            ax4.set_ylim([preds[:, 2].min()-0.1, preds[:, 2].max()+0.1])

            ax4.set_xlabel('X')
            ax4.set_ylabel('Z')
            ax4.set_title('XZ Plane Projection')
            ax4.grid(True)

            plt.tight_layout()
            
            # Save the combined figure
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            images.append(Image.open(buf))
            plt.close()
        
        # Save as GIF
        images[0].save(
            os.path.join(output_dir, f'sample_{idx}_with_proj.gif'),
            save_all=True,
            append_images=images[1:],
            duration=100,  # 0.1秒每帧
            loop=0
        )
        print(f"Saved combined visualization to {output_dir}/sample_{idx}_with_proj.gif")