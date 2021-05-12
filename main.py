
from models import *
from utils import *
import tqdm
import torch
from torch import optim
import os

from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.rasterization import build_rasterizer

from cfg import cfg
from loss_functions import pytorch_neg_multi_log_likelihood_batch
from train_eval import *
import matplotlib.pyplot as plt



def main(model_name, plot_loss = True):
    DIR_INPUT = cfg["data_path"]
    os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
    dm = LocalDataManager(None)
    rasterizer = build_rasterizer(cfg, dm)
    train_dataset, train_dataset_ego, train_dataloader, train_dataloader_ego = get_train_dataloaders(cfg, dm)
    criterion = pytorch_neg_multi_log_likelihood_batch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CSP(cfg, device).to(device)
    epochs = 10
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    train_losses = train(model, train_dataloader, train_dataset_ego, train_dataset, criterion, device, epochs, optimizer)
    torch.save(model, "models/{}.pt".format(model_name))
    np.save("models/training_loss_{}.npy".format(model_name), train_losses)

    eval_dataset, eval_dataloader, eval_dataset_ego, eval_gt_path = generate_eval_dataset(cfg, dm, rasterizer)

    eval(model, eval_dataloader, eval_dataset_ego, eval_dataset, criterion, device, eval_gt_path, cfg)
    
    if plot_loss:
        plt.plot(np.arange(len(train_losses)), train_losses, label="train loss")
        plt.legend()
        plt.show()

    

if __name__ == "__main__":
    model_name = "SuperNetV4"
    main(model_name)