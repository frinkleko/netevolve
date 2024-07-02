# Standard Library
import gc
import os

# Third Party Libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# First Party Libraries
import config
from agent_policy import AgentPolicy
from env import Env
from init_real_data import init_real_data

# Environment Settings
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
print(torch.__config__.parallel_info())

# Constants
EPISODES = 32
STORY_COUNT = 32
GENERATE_COUNT = 5
LEARNED_TIME = 4
GENERATE_TIME = 5
TOTAL_TIME = 10
DEVICE = config.select_device

# Hyperparameters
LR = 4.414072937107742e-06
P_GAMMA = 0.38100283002040913


def execute_data() -> None:
    # Load model parameters
    np_alpha, np_beta, np_gamma, np_delta = [], [], [], []
    with open("model.param.data.fast", "r") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="load data", ncols=80):
            datus = line.strip().split(",")
            np_alpha.append(np.float32(datus[0]))
            np_beta.append(np.float32(datus[1]))
            np_gamma.append(np.float32(datus[2]))
            np_delta.append(np.float32(datus[3]))

    # Define parameters of policy function
    T = np.ones(len(np_alpha), dtype=np.float32)
    e = np.ones(len(np_beta), dtype=np.float32)
    r = np.ones(len(np_alpha), dtype=np.float32)
    w = np.ones(len(np_alpha), dtype=np.float32)
    m = np.ones(len(np_alpha), dtype=np.float32) * 1e-2

    # Define parameters of reward function
    alpha = torch.tensor(np_alpha, dtype=torch.float32).to(DEVICE)
    beta = torch.tensor(np_beta, dtype=torch.float32).to(DEVICE)
    gamma = torch.tensor(np_gamma, dtype=torch.float32).to(DEVICE)
    delta = torch.tensor(np_delta, dtype=torch.float32).to(DEVICE)

    agent_policy = AgentPolicy(r=r, W=w, T=T, e=e, m=m)
    agent_optimizer = optim.Adadelta(agent_policy.parameters(), lr=LR)

    N = len(np_alpha)
    del np_alpha, np_beta, np_gamma, np_delta

    # Setup data
    load_data = init_real_data()
    field = Env(
        edges=load_data.adj[LEARNED_TIME].clone(),
        feature=load_data.feature[LEARNED_TIME].clone(),
        temper=T,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )

    for episode in tqdm(range(EPISODES), desc="episode", ncols=100):
        if episode == 0:
            field.reset(
                load_data.adj[LEARNED_TIME].clone(),
                load_data.feature[LEARNED_TIME].clone(),
            )

        total_reward = 0
        memory = []
        for _ in tqdm(range(STORY_COUNT), desc="story", ncols=100):
            reward = 0
            neighbor_state, feat = field.state()
            action_probs, predict_feat, _ = agent_policy.predict(
                edges=neighbor_state, attributes=feat, N=N)
            reward = field.future_step(action_probs.detach().clone(),
                                       predict_feat.detach())
            total_reward += reward
            memory.append((reward, action_probs))

        if memory:
            G, loss = 0, 0
            for reward, prob in reversed(memory):
                G = reward + P_GAMMA * G
                loss += -torch.sum(torch.log(prob) * G)
            agent_optimizer.zero_grad()
            loss.backward()
            agent_optimizer.step()

    gc.collect()

    calc_log = np.zeros((10, 5))
    calc_nll_log = np.zeros((10, 5))
    attr_calc_log = np.zeros((10, 5))
    attr_calc_nll_log = np.zeros((10, 5))

    for count in range(10):
        field.reset(
            load_data.adj[LEARNED_TIME].clone(),
            load_data.feature[LEARNED_TIME].clone(),
        )

        for t in range(TOTAL_TIME - GENERATE_TIME):
            gc.collect()
            neighbor_state, feat = field.state()
            action_probs, predict_feat, attr_probs = agent_policy.predict(
                edges=neighbor_state, attributes=feat, N=N)

            reward = field.future_step(action_probs, predict_feat)

            # Attribute AUC and NLL
            target_prob = predict_feat.reshape(-1).cpu().detach().numpy()
            detach_attr = load_data.feature[GENERATE_TIME + t].reshape(
                -1).detach().cpu().numpy()
            detach_attr[detach_attr > 0] = 1.0
            try:
                auc_actv = roc_auc_score(detach_attr, target_prob)
                criterion = nn.CrossEntropyLoss()
                error_attr = criterion(torch.from_numpy(target_prob),
                                       torch.from_numpy(detach_attr))
            except ValueError as ve:
                print(ve)
            finally:
                print(f"attr auc, t={t}: {auc_actv}")
                print(f"attr nll, t={t}: {error_attr.item()}")
                attr_calc_log[count][t] = auc_actv
                attr_calc_nll_log[count][t] = error_attr.item()

            # Edge AUC and NLL
            target_prob = action_probs.reshape(-1).cpu().detach().numpy()
            detach_edge = load_data.adj[GENERATE_TIME +
                                        t].reshape(-1).detach().cpu().numpy()
            try:
                auc_actv = roc_auc_score(detach_edge, target_prob)
                criterion = nn.CrossEntropyLoss()
                error_edge = criterion(torch.from_numpy(target_prob),
                                       torch.from_numpy(detach_edge))
            except ValueError as ve:
                print(ve)
            finally:
                print(
                    f"-------\nedge auc, t={t}: {auc_actv}\nedge nll, t={t}: {error_edge.item()}\n-------"
                )
                calc_log[count][t] = auc_actv
                calc_nll_log[count][t] = error_edge.item()

        print("---")

    # np.save("proposed_edge_dblp_auc", calc_log)
    # np.save("proposed_edge_dblp_nll", calc_nll_log)
    # np.save("proposed_attr_dblp_auc", attr_calc_log)
    # np.save("proposed_attr_dblp_nll", attr_calc_nll_log)


if __name__ == "__main__":
    execute_data()
