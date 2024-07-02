# Standard Library
import gc
import os

# Third Party Libraries
import numpy as np
import optuna
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# First Party Libraries
import config
from agent_policy import AgentPolicy
from env import Env
from init_real_data import init_real_data

# Environment settings
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
print(torch.__config__.parallel_info())

# Constants
EPISODES = 32
STORY_COUNT = 32
GENERATE_COUNT = 5
LEARNED_TIME = 2
GENERATE_TIME = 3
TOTAL_TIME = 5
DEVICE = config.select_device


def execute_data(trial) -> float:
    # Hyperparameters
    lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
    p_gamma = trial.suggest_float("p_gamma", 0.1, 0.98, log=True)

    # Load data
    np_alpha, np_beta, np_gamma, np_delta = [], [], [], []
    with open("model.param.data.fast", "r") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="load data", ncols=80):
            datus = line.strip().split(",")
            np_alpha.append(np.float32(datus[0]))
            np_beta.append(np.float32(datus[1]))
            np_gamma.append(np.float32(datus[2]))
            np_delta.append(np.float32(datus[3]))

    # Convert data to numpy arrays
    np_alpha, np_beta, np_gamma, np_delta = map(
        np.array, (np_alpha, np_beta, np_gamma, np_delta))

    # Define parameters
    T, e, r, w, m = [
        np.ones(np_alpha.shape, dtype=np.float32) for _ in range(5)
    ]
    alpha, beta, gamma, delta = [
        torch.ones(arr.shape, dtype=torch.float32).to(DEVICE)
        for arr in (np_alpha, np_beta, np_gamma, np_delta)
    ]

    # Initialize AgentPolicy and optimizer
    agent_policy = AgentPolicy(r=r, W=w, T=T, e=e, m=m)
    agent_optimizer = optim.Adadelta(agent_policy.parameters(), lr=lr)

    N = len(np_alpha)
    del np_alpha, np_beta, np_gamma, np_delta

    # Load environment data
    load_data = init_real_data()
    field = Env(
        edges=load_data.adj[LEARNED_TIME].detach().clone(),
        feature=load_data.feature[LEARNED_TIME].detach().clone(),
        temper=T,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )

    # Training loop
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

        if not memory:
            continue

        G, loss = 0, 0
        for reward, prob in reversed(memory):
            G = reward + p_gamma * G
            loss += -torch.sum(torch.log(prob) * G)
        agent_optimizer.zero_grad()
        loss.backward()
        agent_optimizer.step()

    gc.collect()

    # Evaluation
    field.reset(
        load_data.adj[LEARNED_TIME].clone(),
        load_data.feature[LEARNED_TIME].clone(),
    )
    edge_auc, attr_auc = 0, 0
    for t in range(TOTAL_TIME - GENERATE_TIME):
        gc.collect()
        neighbor_state, feat = field.state()
        action_probs, predict_feat, attr_probs = agent_policy.predict(
            edges=neighbor_state, attributes=feat, N=N)

        reward = field.future_step(action_probs, predict_feat)

        # Attribute AUC
        target_prob = predict_feat.reshape(-1).cpu().detach().numpy()
        detach_attr = load_data.feature[GENERATE_TIME +
                                        t].reshape(-1).detach().cpu().numpy()
        detach_attr[detach_attr > 0] = 1.0
        try:
            auc_actv = roc_auc_score(detach_attr, target_prob)
            attr_auc += auc_actv / (TOTAL_TIME - GENERATE_TIME)
        except ValueError as ve:
            print(ve)
        finally:
            print(f"attr auc, t={t}: {auc_actv}")

        # Edge AUC
        target_prob = action_probs.reshape(-1).cpu().detach().numpy()
        detach_edge = load_data.adj[GENERATE_TIME +
                                    t].reshape(-1).detach().cpu().numpy()
        try:
            auc_actv = roc_auc_score(detach_edge, target_prob)
            edge_auc += auc_actv / (TOTAL_TIME - GENERATE_TIME)
        except ValueError as ve:
            print(ve)
        finally:
            print(f"-------\nedge auc, t={t}: {auc_actv}\n-------")

    del field

    return (edge_auc + attr_auc) / 2


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(execute_data, n_trials=100)
    print("Best learning rate:", study.best_params["lr"])
