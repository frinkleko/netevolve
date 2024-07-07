import gc
import os
import argparse
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

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


def load_model_params():
    np_alpha, np_beta, np_gamma, np_delta = [], [], [], []
    with open("model.param.data.fast", "r") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="load data", ncols=80):
            datus = line.strip().split(",")
            np_alpha.append(np.float32(datus[0]))
            np_beta.append(np.float32(datus[1]))
            np_gamma.append(np.float32(datus[2]))
            np_delta.append(np.float32(datus[3]))
    return np.array(np_alpha), np.array(np_beta), np.array(np_gamma), np.array(
        np_delta)


def initialize_parameters(np_alpha, np_beta, np_gamma, np_delta):
    T, e, r, w, m = [
        np.ones(np_alpha.shape, dtype=np.float32) for _ in range(5)
    ]
    alpha, beta, gamma, delta = [
        torch.tensor(arr, dtype=torch.float32).to(DEVICE)
        for arr in (np_alpha, np_beta, np_gamma, np_delta)
    ]
    return T, e, r, w, m, alpha, beta, gamma, delta


def train_agent(agent_policy, agent_optimizer, field, N, p_gamma, load_data):
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


def evaluate_agent(agent_policy, field, load_data, N):
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

    return (edge_auc + attr_auc) / 2


def execute_data_optim(trial) -> float:
    # Hyperparameters
    lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
    p_gamma = trial.suggest_float("p_gamma", 0.1, 0.98, log=True)

    # Load data and initialize parameters
    np_alpha, np_beta, np_gamma, np_delta = load_model_params()
    T, e, r, w, m, alpha, beta, gamma, delta = initialize_parameters(
        np_alpha, np_beta, np_gamma, np_delta)

    # Initialize AgentPolicy and optimizer
    agent_policy = AgentPolicy(r=r, W=w, T=T, e=e, m=m, device=DEVICE)
    agent_optimizer = optim.Adadelta(agent_policy.parameters(), lr=lr)

    N = len(np_alpha)
    del np_alpha, np_beta, np_gamma, np_delta

    # Load environment data
    load_data = init_real_data(DATASET)
    field = Env(
        edges=load_data.adj[LEARNED_TIME].detach().clone(),
        feature=load_data.feature[LEARNED_TIME].detach().clone(),
        temper=T,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        device=DEVICE,
    )

    # Train agent
    train_agent(agent_policy, agent_optimizer, field, N, p_gamma, load_data)

    # Evaluate agent
    return evaluate_agent(agent_policy, field, load_data, N)


def execute_data_rl(args) -> None:
    # Load data and initialize parameters
    np_alpha, np_beta, np_gamma, np_delta = load_model_params()
    T, e, r, w, m, alpha, beta, gamma, delta = initialize_parameters(
        np_alpha, np_beta, np_gamma, np_delta)

    # Initialize AgentPolicy and optimizer
    agent_policy = AgentPolicy(r=r, W=w, T=T, e=e, m=m, device=DEVICE)
    agent_optimizer = optim.Adadelta(agent_policy.parameters(), lr=LR)

    N = len(np_alpha)
    del np_alpha, np_beta, np_gamma, np_delta

    # Load environment data
    load_data = init_real_data(DATASET)
    field = Env(
        edges=load_data.adj[LEARNED_TIME].clone(),
        feature=load_data.feature[LEARNED_TIME].clone(),
        temper=T,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        device=DEVICE,
    )

    # Train agent
    train_agent(agent_policy, agent_optimizer, field, N, P_GAMMA, load_data)

    # Evaluate agent
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        type=str,
                        choices=["optimize", "run"],
                        required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset", type=str, default="NIPS")
    args = parser.parse_args()

    # Constants
    EPISODES = 32
    STORY_COUNT = 32
    GENERATE_COUNT = 5
    LEARNED_TIME = 4
    GENERATE_TIME = 5
    TOTAL_TIME = 10
    DEVICE = torch.device(args.device)
    DATASET = args.dataset

    if args.mode == "optimize":
        study = optuna.create_study(direction="maximize")
        study.optimize(execute_data_optim, n_trials=100)
        print("Best learning rate:", study.best_params["lr"])
    elif args.mode == "run":
        # Hyperparameters
        LR = 4.414072937107742e-06
        P_GAMMA = 0.38100283002040913
        execute_data_rl(args)
    else:
        raise ValueError("Invalid mode")
