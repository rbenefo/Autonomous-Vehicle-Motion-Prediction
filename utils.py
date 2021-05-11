
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from torch.utils.data import DataLoader
from l5kit.data import LocalDataManager, ChunkedDataset
from torch import Tensor
from typing import List, Optional, Tuple
import l5kit
import bisect
import torch
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
import numpy as np
from l5kit.geometry import transform_points

def visualize_output(cfg, model, eval_gt_path, eval_dataset,rasterizer, eval_zarr):
    model.eval()
    torch.set_grad_enabled(False)

    # build a dict to retrieve future trajectories from GT
    gt_rows = {}
    for row in read_gt_csv(eval_gt_path):
        gt_rows[row["track_id"] + row["timestamp"]] = row["coord"]

    eval_ego_dataset = EgoDataset(cfg, eval_dataset.dataset, rasterizer)
    for frame_number in range(99, len(eval_zarr.frames), 100):  # start from last frame of scene_0 and increase by 100
        plt.figure(figsize=(10,10))
        agent_indices = eval_dataset.get_frame_indices(frame_number) 
        if not len(agent_indices):
            continue

        # get AV point-of-view frame
        data_ego, index = eval_ego_dataset[frame_number]
        im_ego = rasterizer.to_rgb(data_ego["image"].transpose(1, 2, 0))
        center = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]
        
        predicted_positions = []
        target_positions = []

        for v_index in agent_indices:
            data_agent, index = eval_dataset[v_index]

            predictions, confidences= display_CSPforward(data_agent, index, model, eval_ego_dataset, eval_dataset, device = device)
    #         out_net = model(torch.from_numpy(data_agent["image"]).unsqueeze(0).to(device))
    #         _, preds, confidences = forward(data, model, device, criterion)    
            best_prediction_idx = torch.argmax(confidences)
            out_pos = predictions[:,best_prediction_idx,:,:]
            out_pos = out_pos.reshape(-1, 2).detach().cpu().numpy()
            
            # store absolute world coordinates
            predicted_positions.append(transform_points(out_pos, data_agent["world_from_agent"]))
            # retrieve target positions from the GT and store as absolute coordinates
            track_id, timestamp = data_agent["track_id"], data_agent["timestamp"]
            target_positions.append(gt_rows[str(track_id) + str(timestamp)] + data_agent["centroid"][:2])


        # convert coordinates to AV point-of-view so we can draw them
        predicted_positions = transform_points(np.concatenate(predicted_positions), data_ego["raster_from_world"])
        target_positions = transform_points(np.concatenate(target_positions), data_ego["raster_from_world"])

        draw_trajectory(im_ego, predicted_positions, PREDICTED_POINTS_COLOR)
        draw_trajectory(im_ego, target_positions, TARGET_POINTS_COLOR)
        plt.imshow(im_ego)
        plt.show()

def generate_mask_coords(agent_centroid, ego_centroid, world_from_agent, mask):
    """agent_centroid: (1,2) arr
       ego_centroid: (,2 arr)"""
    assert mask.shape[1]%2 == 1 #masks must be odd number shaped
    assert mask.shape[2]%2 == 1 #masks must be odd number shaped
    ego_centroid = torch.from_numpy(np.expand_dims(ego_centroid, axis = 0))
    ego_centroid = transform_points(ego_centroid, world_from_agent)
    rel_centroid = (agent_centroid.squeeze()-ego_centroid.numpy().squeeze())
    #for now, let's generate a 13x13 mask. Could choose long mask (e.x. 13x3) if we
    #input ego heading to determine whether x or y becomes the long end.
    #1 square == ~15 feet
    map_x_uncentered = rel_centroid[0]//15
    map_y_uncentered = rel_centroid[1]//15
    
    #Adjust coords to put ego vehicle in center, and flip x and y for python array indexing
    map_x_center = mask.shape[1]//2+1
    map_y_center = mask.shape[2]//2+1
    
    map_x_centered = map_y_center-map_y_uncentered
    map_y_centered = map_x_center+map_x_uncentered

    if map_x_centered < 0 or map_x_centered >= mask.shape[1]:
        return None
    if map_y_centered < 0 or map_y_centered >= mask.shape[2]:
        return None
    else:
        return [int(map_x_centered), int(map_y_centered)]



def get_agent_data(index, ego_dataset, agent_dataset):
    if index < 0:
        if -index > len(agent_dataset):
            raise ValueError("absolute value of index should not exceed dataset length")
        index = len(agent_dataset) + index
        index = agent_dataset.agents_indices[index]

    frame_index = bisect.bisect_right(agent_dataset.cumulative_sizes_agents, index)
    scene_index = bisect.bisect_right(agent_dataset.cumulative_sizes, frame_index)
    if scene_index == 0:
        state_index = frame_index
    else:
        state_index = frame_index - agent_dataset.cumulative_sizes[scene_index - 1]
    frames = ego_dataset.dataset.frames[l5kit.data.filter.get_frames_slice_from_scenes(ego_dataset.dataset.scenes[scene_index])]

    agents = ego_dataset.dataset.agents
    x = get_agent_context(state_index, frames, agents, history_num_frames= 10, future_num_frames = 50)
    return x[2]
def get_agent_context(
        state_index: int,
        frames: np.ndarray,
        agents: np.ndarray,
        history_num_frames: int,
        future_num_frames: int,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    MODIFIED FROM L5KIT
    Slice zarr or numpy arrays to get the context around the agent onf interest (both in space and time)
    Args:
        state_index (int): frame index inside the scene
        frames (np.ndarray): frames from the scene
        agents (np.ndarray): agents from the scene
        history_num_frames (int): how many frames in the past to slice
        future_num_frames (int): how many frames in the future to slice
    Returns:
        Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]
    """

    #  the history slice is ordered starting from the latest frame and goes backward in time., ex. slice(100, 91, -2)
    history_slice = l5kit.sampling.slicing.get_history_slice(state_index, history_num_frames, 1, include_current_state=True)
    future_slice = l5kit.sampling.slicing.get_future_slice(state_index, future_num_frames, 1)
    history_frames = frames[history_slice].copy()  # copy() required if the object is a np.ndarray
    future_frames = frames[future_slice].copy()
    sorted_frames = np.concatenate((history_frames[::-1], future_frames))  # from past to future

    # get agents (past and future)
    agent_slice = l5kit.data.filter.get_agents_slice_from_frames(sorted_frames[0], sorted_frames[-1])
    agents = agents[agent_slice].copy()
    # sync interval with the agents array
    history_frames["agent_index_interval"] -= agent_slice.start
    future_frames["agent_index_interval"] -= agent_slice.start
    history_agents = l5kit.data.filter.filter_agents_by_frames(history_frames, agents)
    future_agents = l5kit.data.filter.filter_agents_by_frames(future_frames, agents)
    return history_frames, future_frames, history_agents, future_agents

def pytorch_neg_multi_log_likelihood_batch(
    gt: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor
) -> Tensor:
    """
    COPIED FROM L5KIT

    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    assert len(pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    batch_size, num_modes, future_len, num_coords = pred.shape

    assert gt.shape == (batch_size, future_len, num_coords), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"
    assert confidences.shape == (batch_size, num_modes), f"expected 1D (Modes) array for gt, got {confidences.shape}"
    assert torch.allclose(torch.sum(confidences, dim=1), confidences.new_ones((batch_size,))), "confidences should sum to 1"
    assert avails.shape == (batch_size, future_len), f"expected 1D (Time) array for gt, got {avails.shape}"
    # assert all data are valid
    assert torch.isfinite(pred).all(), "invalid value found in pred"
    assert torch.isfinite(gt).all(), "invalid value found in gt"
    assert torch.isfinite(confidences).all(), "invalid value found in confidences"
    assert torch.isfinite(avails).all(), "invalid value found in avails"

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # reduce coords and use availability

    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = torch.log(confidences + 1e-10) - 0.5 * torch.sum(error, dim=-1)  # reduce time

    # use max aggregator on modes for numerical stability
    # error (batch_size, num_modes)
    max_value, _ = error.max(dim=1, keepdim=True)  # error are negative at this point, so max() gives the minimum one
    error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value  # reduce modes
    # print("error", error)
    return torch.mean(error)


def visualize_trajectory(dataset, index, title="target_positions movement with draw_trajectory"):
    """Copied from L5Kit"""
    data, indice = dataset[index]
    im = data["image"].transpose(1, 2, 0)
    print(im.shape)
    im = dataset.rasterizer.to_rgb(im)
    target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])
    draw_trajectory(im, target_positions_pixels, TARGET_POINTS_COLOR, radius=1, yaws=data["target_yaws"])

    plt.title(title)
    plt.imshow(im[::-1])
    plt.show()


def get_train_dataloaders(cfg):
    """Modified from L5Kit"""
    DIR_INPUT = cfg["data_path"]
    os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
    dm = LocalDataManager(None)
    train_cfg = cfg["train_data_loader"]
    rasterizer = build_rasterizer(cfg, dm)
    train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()

    #create mask 


    train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
    train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 
                                num_workers=train_cfg["num_workers"])

    train_dataset_ego = EgoDataset(cfg, train_zarr, rasterizer)
    train_dataloader_ego = DataLoader(train_dataset_ego, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 
                                num_workers=train_cfg["num_workers"])

    return train_dataset, train_dataset_ego, train_dataloader, train_dataloader_ego