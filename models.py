import torch
from torch import nn, optim
from torch.nn import functional as F
import cfg
import timm
class SimpleNet(nn.Module):
    def __init__(self, cfg):
        super(SimpleNet, self).__init__()
        num_targets = 2 * cfg["model_params"]["future_num_frames"]
        self.net = nn.Sequential(nn.Conv2d(25, 5, kernel_size = 3, padding=1),
                                 nn.ReLU(),
                                 nn.Conv2d(5, 2, kernel_size = 3, padding=1),
                                 nn.Flatten(),
                                 nn.Linear(224*224*2, num_targets)
                                 )
    def forward(self, x):
        x = self.net(x)
        return x
    def run(self, data, device, criterion):
        inputs = data["image"].to(device)
        target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
        targets = data["target_positions"].to(device)
        # Forward pass
        outputs = model(inputs).reshape(targets.shape)

        loss = criterion(outputs, targets)
        # not all the output steps are valid, but we can filter them out from the loss using availabilities
        loss = loss * target_availabilities
        loss = loss.mean()
        return loss, outputs

class ConvNet1(nn.Module):
    def __init__(self, cfg, backbone):
        super().__init__()
        
        num_targets = 2 * cfg["model_params"]["future_num_frames"]
        num_predictions = cfg["model_params"]["num_trajectories"]
        self.backbone = timm.create_model(backbone, pretrained=True, in_chans=25)
        if backbone == "xception41":
            num_out_backbone = 2048
        elif backbone == "xception71":
            num_out_backbone = 2048
#         self.fc1 = nn.Linear(num_out_backbone, num_targets)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(num_out_backbone, num_targets*num_predictions)
        self.confidences = nn.Linear(num_out_backbone, 3)
        self.predictions_shape = (cfg["model_params"]["num_trajectories"], cfg["model_params"]["future_num_frames"], 2)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim = 1)
        x = F.relu(x)
        predictions = self.fc1(x)
        confidences = F.softmax(self.confidences(x), dim = 1)
        predictions = predictions.view(-1, self.predictions_shape[0], self.predictions_shape[1], self.predictions_shape[2])

#         if not torch.allclose(torch.sum(confidences, dim=1),torch.ones((16)).to("cuda")):
#             breakpoint()
        return predictions, confidences

    def run(self, data, model, device, criterion):
        inputs = data["image"].to(device)
        target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
        targets = data["target_positions"].to(device)
        # Forward pass
    #     outputs = model(inputs).reshape(targets.shape)
        outputs, confidences = self.forward(inputs)

    #     loss = criterion(outputs, targets)
        loss = criterion(targets, outputs, confidences, target_availabilities.squeeze())
        # not all the output steps are valid, but we can filter them out from the loss using availabilities
    #     loss = loss * target_availabilities
    #     loss = loss.mean()
        return loss, outputs, confidences


##########
#CSP+ConvNet
##########
class CSP(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.CSPEncoder = CSPEncoder(cfg).to(device)
        self.SocialPooling = SocialPooling(cfg, device).to(device)
        self.CSPDecoder = CSPDecoder(cfg).to(device)

    def forward(self,nbrs_hist, ego_hist, masks, ego_curr, ims):
        nbrs_encoded, ego_encoded = self.CSPEncoder(nbrs_hist, ego_hist)
        trajectory_concated = self.SocialPooling(nbrs_encoded, ego_encoded, masks)
        predictions, confidences = self.CSPDecoder(trajectory_concated, ego_curr, ims)
        return predictions, confidences
    def CSPforward(self, data,indices, ego_dataset, agent_dataset, criterion=pytorch_neg_multi_log_likelihood_batch, device = "cuda"):
        masks = torch.zeros((len(indices), 13, 13))
        tot_nbrs = []
        for i, scene_id in enumerate(indices):
            nbrs = torch.zeros(0,10,2)
            agent_data = get_agent_data(scene_id, ego_dataset, agent_dataset)
            cur_agents = agent_data[0]
            filtered_agents = l5kit.data.filter.filter_agents_by_labels(cur_agents, 0.5)
            curr_valid_agents = filtered_agents["track_id"]
            for agent_id in curr_valid_agents:
                agent_history = []
                for frame in agent_data:
                    agent_id_index = np.where(frame["track_id"] == agent_id)
                    if len(agent_id_index[0]) > 0:
                        agent_history.append(frame["centroid"][agent_id_index])
                    else:
                        #Agent was not seen for this frame...
                        agent_history.append(np.array([[0,0]]))
                    agent_history_arr = np.array(agent_history)[:-1].squeeze(1)
                agent_history_arr=np.flip(agent_history_arr, axis=0).copy()
                agent_curr_coords = agent_history_arr[-1]
                agent_mask_coords = generate_mask_coords(agent_curr_coords, data["centroid"][i].numpy(), data["world_from_agent"][i].numpy(), masks)
                if agent_mask_coords is not None:
                    masks[i, agent_mask_coords[0], agent_mask_coords[1]] = 1
                new_hist = torch.from_numpy(agent_history_arr).unsqueeze(0)
                nbrs = torch.cat((nbrs, new_hist), axis = 0)
            tot_nbrs.append(nbrs)
        
        nbrs_batch = torch.nn.utils.rnn.pad_sequence(tuple(tot_nbrs), batch_first=True)
        ego_hist = data["history_positions"][:,:-1]
        ego_hist = ego_hist[torch.arange(ego_hist.shape[0]).unsqueeze(-1), data["history_availabilities"][:,:-1].long()]
        #transform ego_hist to 0 frame: 
        target_availabilities = data["target_availabilities"].to(device)
        targets = data["target_positions"].to(device)
        
        nbrs_current_mask = data["image"][:,cfg["model_params"]["history_num_frames"]+1,:,:]
        ims = data["image"]
        
        predictions, confidences = self.forward(nbrs_batch.to(device), ego_hist.to(device), masks.to(device), data["centroid"].to(device), ims.to(device))
        loss = criterion(targets, predictions, confidences, target_availabilities)
        return loss, predictions, confidences
    def display_CSPforward(self, data,index, ego_dataset, agent_dataset, device = "cuda"):
#     nbrs_batch = torch.zeros(len(indices),0,0,2) #Want BxNxHx2
        masks = torch.zeros((1, 13, 13))
        tot_nbrs = []

        nbrs = torch.zeros(0,10,2)
        agent_data = get_agent_data(index, ego_dataset, agent_dataset)
        cur_agents = agent_data[0]
        filtered_agents = l5kit.data.filter.filter_agents_by_labels(cur_agents, 0.5)
        curr_valid_agents = filtered_agents["track_id"]
        for agent_id in curr_valid_agents:
            agent_history = []
            for frame in agent_data:
                agent_id_index = np.where(frame["track_id"] == agent_id)
                if len(agent_id_index[0]) > 0:
                    agent_history.append(frame["centroid"][agent_id_index])
                else:
                    #Agent was not seen for this frame...
                    agent_history.append(np.array([[0,0]]))
                agent_history_arr = np.array(agent_history)[:-1].squeeze(1)
            agent_history_arr=np.flip(agent_history_arr, axis=0).copy()
            agent_curr_coords = agent_history_arr[-1]
            
            agent_mask_coords = generate_mask_coords(agent_curr_coords, data["centroid"], data["world_from_agent"], masks)
            if agent_mask_coords is not None:
                masks[0, agent_mask_coords[0], agent_mask_coords[1]] = 1
            new_hist = torch.from_numpy(agent_history_arr).unsqueeze(0)
            nbrs = torch.cat((nbrs, new_hist), axis = 0)
        tot_nbrs.append(nbrs)

        nbrs_batch = torch.nn.utils.rnn.pad_sequence(tuple(tot_nbrs), batch_first=True)
    #     nbrs_batch_packed = torch.nn.utils.rnn.pack_sequence(tuple(tot_nbrs),enforce_sorted=False)
        ego_hist = torch.from_numpy(data["history_positions"][:-1])
        ego_hist = ego_hist[data["history_availabilities"][:-1]]
        
        target_availabilities = torch.from_numpy(data["target_availabilities"]).to(device)
        targets =torch.from_numpy(data["target_positions"]).to(device)

        
        nbrs_current_mask = data["image"][cfg["model_params"]["history_num_frames"]+1,:,:]
        ims = data["image"]
        
        nbrs_current_mask = torch.from_numpy(nbrs_current_mask)
        ims = torch.from_numpy(ims)
        
        ego_centroid = torch.from_numpy(data["centroid"])
        ims, nbrs_current_mask,ego_hist, ego_centroid=  ims.unsqueeze(0), nbrs_current_mask.unsqueeze(0), ego_hist.unsqueeze(0).float(), ego_centroid.unsqueeze(0)
        
        predictions, confidences = self.forward(nbrs_batch.to(device), ego_hist.to(device), masks.to(device), ego_centroid.to(device), ims.to(device))
        return predictions, confidences

class CSPEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        #need linear layer??
        self.num_history = cfg["model_params"]["history_num_frames"]
        self.alpha = cfg["model_params"]["alpha"]
        self.LSTM_hidden_size = cfg["model_params"]["LSTM_hidden_size"]
        self.encoderLSTM = nn.LSTM(2, self.LSTM_hidden_size,1)
  
    def forward(self,nbrs_hist, ego_hist):
        #nbrs_hist: BxNxHx2 --> (BxN)xHx2
        nbrs_hist_encoded =  nbrs_hist.view(nbrs_hist.shape[0]*nbrs_hist.shape[1],nbrs_hist.shape[2], nbrs_hist.shape[3])
        _, (neighbors_encoded, _) = self.encoderLSTM(nbrs_hist_encoded.permute(1,0,2).float())
        neighbors_encoded = neighbors_encoded.view(nbrs_hist.shape[0], nbrs_hist.shape[1], self.LSTM_hidden_size)
        neighbors_encoded = neighbors_encoded.squeeze(0)
        neighbors_encoded = F.leaky_relu(neighbors_encoded, self.alpha)
        _, (ego_encoded, _) = self.encoderLSTM(ego_hist.permute(1,0,2))
        ego_encoded = ego_encoded.squeeze(0)
        ego_encoded = F.leaky_relu(ego_encoded, self.alpha)
        return neighbors_encoded, ego_encoded

class SocialPooling(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.encoder_LSTM_hidden_state_shape = cfg["model_params"]["LSTM_hidden_size"]

        self.conv1 = nn.Conv2d(self.encoder_LSTM_hidden_state_shape, 64, kernel_size = 3)
        self.conv2 = nn.Conv2d(64, 16, kernel_size = (3,1))
        self.max_pool = nn.MaxPool2d(kernel_size = (2,1))
        self.social_reduction = nn.Linear(704, 32)
        self.vehicle_dynamics = nn.Linear(self.encoder_LSTM_hidden_state_shape, 32)
        self.alpha = cfg["model_params"]["alpha"]
        self.device = device
  
    def forward(self, nbrs_encoding, ego_encoding, masks):
        social_encoding = torch.zeros((masks.shape[0],masks.shape[1],masks.shape[2], self.encoder_LSTM_hidden_state_shape)).to(self.device)
        nbrs_curr_stacked = torch.repeat_interleave(masks.byte().unsqueeze(3), self.encoder_LSTM_hidden_state_shape, dim = 3)
        
        social_encoding = social_encoding.masked_scatter_(nbrs_curr_stacked, nbrs_encoding)
        nbrs_conved = self.conv1(social_encoding.permute(0,3,1,2))
        nbrs_conved = F.leaky_relu(nbrs_conved, self.alpha)
        nbrs_conved = self.conv2(nbrs_conved)
        nbrs_conved = F.leaky_relu(nbrs_conved, self.alpha)
        nbrs_conved = self.max_pool(nbrs_conved)
        nbrs_conved = torch.flatten(nbrs_conved, start_dim = 1, end_dim = 3)
        nbrs_conved = self.social_reduction(nbrs_conved)
        x_ego = self.vehicle_dynamics(ego_encoding)

        traj_concat = torch.cat((nbrs_conved, x_ego), dim = 1)
        return traj_concat


class CSPDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.ComplexNet = torch.load("models/ConvNetV3.pt")
        self.ComplexNet.eval()
        self.decoderLSTM = nn.LSTM(64, 30, 1)
        self.output_flat_shape = 2*cfg["model_params"]["future_num_frames"]*cfg["model_params"]["num_trajectories"]
        self.traj_concat_fc = nn.Linear(64, 64)
        self.output_predictions = nn.Linear(30, 2) #50 50x2x3

        self.convScaling = nn.Linear(self.output_flat_shape, self.output_flat_shape)
        self.CSP_Scaling = nn.Linear(self.output_flat_shape, self.output_flat_shape)
        
        self.confidences = nn.Linear(self.output_flat_shape, cfg["model_params"]["num_trajectories"]) #50 50x2x3
        self.alpha = cfg["model_params"]["alpha"]
        self.predictions_shape = (cfg["model_params"]["num_trajectories"], cfg["model_params"]["future_num_frames"],2)
        
        self.convConfidencesScaling = nn.Linear(cfg["model_params"]["num_trajectories"], cfg["model_params"]["num_trajectories"])
        self.cspConfidencesScaling = nn.Linear(cfg["model_params"]["num_trajectories"], cfg["model_params"]["num_trajectories"])
    def forward(self, trajectory_concated, ego_curr, ims):
        trajectory_concated = torch.flatten(trajectory_concated, 1)
        trajectory_concated = self.traj_concat_fc(trajectory_concated)
        trajectory_concated = trajectory_concated.repeat((int(self.output_flat_shape/2), 1, 1))
        
        output, (_, _) = self.decoderLSTM(trajectory_concated)
        
        output = F.leaky_relu(output.permute(1,0,2), self.alpha)
        complexNet_predictions, complexNet_confidences = self.ComplexNet(ims)
        csp_predictions = self.output_predictions(output)
        complexNet_predictions = torch.flatten(complexNet_predictions, 1)
        csp_predictions = torch.flatten(csp_predictions, 1)
        predictions = self.convScaling(complexNet_predictions)+self.CSP_Scaling(csp_predictions)
        csp_confidences = F.softmax(self.confidences(predictions).squeeze(1))
        confidences = self.convConfidencesScaling(complexNet_confidences)+self.cspConfidencesScaling(csp_confidences)
        confidences = F.softmax(confidences)
        predictions = predictions.view(-1, self.predictions_shape[0], self.predictions_shape[1], self.predictions_shape[2])

        return predictions, confidences 

