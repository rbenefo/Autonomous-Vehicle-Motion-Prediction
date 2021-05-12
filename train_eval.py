import tqdm
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.geometry import transform_points

import torch
from torch.utils.data import DataLoader
import shutil
from pathlib import Path
from tempfile import gettempdir
import numpy as np


def get_train_dataloaders(cfg, dm):
    """Modified from L5Kit"""
    train_cfg = cfg["train_data_loader"]
    rasterizer = build_rasterizer(cfg, dm)
    train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
    train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
    train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 
                                num_workers=train_cfg["num_workers"])

    train_dataset_ego = EgoDataset(cfg, train_zarr, rasterizer)
    train_dataloader_ego = DataLoader(train_dataset_ego, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 
                                num_workers=train_cfg["num_workers"])

    return train_dataset, train_dataset_ego, train_dataloader, train_dataloader_ego



def train(model, train_dataloader, train_dataset_ego, train_dataset, criterion, device, epochs, optimizer):
    tr_it = iter(train_dataloader)
    losses_train = []
    for i in range(epochs):
        print("Epoch {}".format(i))
        progress_bar = tqdm(range(400))
        for _ in progress_bar:
            try:
                data, indices = next(tr_it)
            except StopIteration:
                tr_it = iter(train_dataloader)
                data, indices = next(tr_it)
            model.train()
            torch.set_grad_enabled(True)
            loss, predictions, confidences = model.run(data, indices, model, train_dataset_ego, train_dataset, criterion, device = device)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_train.append(loss.item())
            progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")
    return losses_train


########################################
#EVAL
########################################
def generate_eval_dataset(cfg, dm, rasterizer):
    eval_cfg = cfg["test_data_loader"]
    eval_dir = shutil.copytree(dm.require(eval_cfg["key"]), '/tmp/lyft/test.zarr')
    eval_cfg = cfg["test_data_loader"]
    num_frames_to_chop = 50
    eval_base_path = create_chopped_dataset(eval_dir, cfg["raster_params"]["filter_agents_threshold"], 
                                num_frames_to_chop, cfg["model_params"]["future_num_frames"], MIN_FUTURE_STEPS)

    eval_zarr_path = str(Path(eval_base_path) / "test.zarr")
    eval_mask_path = str(Path(eval_base_path) / "mask.npz")
    eval_gt_path = str(Path(eval_base_path) / "gt.csv")

    eval_zarr = ChunkedDataset(eval_zarr_path).open()
    eval_mask = np.load(eval_mask_path)["arr_0"]
    # ===== INIT DATASET AND LOAD MASK
    eval_dataset = AgentDataset(cfg, eval_zarr, rasterizer, agents_mask=eval_mask)
    eval_dataloader = DataLoader(eval_dataset, shuffle=eval_cfg["shuffle"], batch_size=eval_cfg["batch_size"], 
                                num_workers=eval_cfg["num_workers"])
    eval_dataset_ego = EgoDataset(cfg, eval_zarr, rasterizer)

    return eval_dataset, eval_dataloader, eval_dataset_ego, eval_gt_path


def eval(model, eval_dataloader, eval_dataset_ego, eval_dataset, criterion, device, eval_gt_path, cfg):
    # ===== EVAL Loop
    model.eval()
    torch.set_grad_enabled(False)
    # store information for evaluation
    future_coords_offsets_pd = []
    timestamps = []
    confidences_list = []
    agent_ids = []
    progress_bar = tqdm(eval_dataloader)
    for data, indices in progress_bar:
        loss, preds, confidences = mode.run(data, indices, model, eval_dataset_ego, eval_dataset, criterion, device = device)
        #fix for the new environment
        preds = preds.cpu().numpy()
        world_from_agents = data["world_from_agent"].numpy()
        centroids = data["centroid"].numpy()
        coords_offset = []

        # convert into world coordinates and compute offsets
        for idx in range(len(preds)):
            for mode in range(cfg["model_params"]["num_trajectories"]):
                preds[idx,mode,:, :] = transform_points(preds[idx,mode,:, :], world_from_agents[idx]) - centroids[idx][:2]

        future_coords_offsets_pd.append(preds.copy())
        confidences_list.append(confidences.cpu().numpy().copy())
        timestamps.append(data["timestamp"].numpy().copy())
        agent_ids.append(data["track_id"].numpy().copy()) 
    pred_path = f"{gettempdir()}/pred.csv"

    write_pred_csv(pred_path,
                timestamps=np.concatenate(timestamps),
                track_ids=np.concatenate(agent_ids),
                coords=np.concatenate(future_coords_offsets_pd),
                confs = np.concatenate(confidences_list)
                )
    metrics = compute_metrics_csv(eval_gt_path, pred_path, [neg_multi_log_likelihood, time_displace])
    for metric_name, metric_mean in metrics.items():
        print(metric_name, metric_mean)
