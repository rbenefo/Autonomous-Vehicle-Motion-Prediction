
from models import *
from utils import *
import tqdm
import torch
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace

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
        torch.save(model, "models/SuperNetV4_epoch{}.pt".format(i))

def eval(model, eval_dataloader, eval_dataset_ego, eval_dataset, criterion, device):
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

        loss, preds, confidences = CSPforward(data, indices, model, eval_dataset_ego, eval_dataset, criterion, device = device)
    #     loss, preds, confidences = forward(data, model, device, criterion)
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


def main():
    # set env variable for data
    DIR_INPUT = cfg["data_path"]
    os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
    dm = LocalDataManager(None)