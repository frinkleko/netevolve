# Third Party Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from data_loader import init_real_data


class Model(nn.Module):

    def __init__(self, alpha, beta, gamma, delta, device):
        super().__init__()
        self.alpha = nn.Parameter(alpha, requires_grad=True).to(device)
        self.beta = nn.Parameter(beta, requires_grad=True).to(device)
        self.gamma = nn.Parameter(gamma, requires_grad=True).to(device)


class Optimizer:

    def __init__(self, edges, feats, model: Model, size: int, device):
        self.edges = edges
        self.feats = feats
        self.model = model
        self.size = size
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001)
        self.device = device

    def optimize(self, t: int):
        feat = self.feats[t].to(self.device)
        edge = self.edges[t].to(self.device)
        self.optimizer.zero_grad()

        dot_product = torch.matmul(feat, feat.t()).to(self.device)
        sim = torch.mul(edge, dot_product)
        sim = torch.mul(sim, self.model.alpha)
        sim = torch.add(sim, 0.001)

        costs = torch.mul(edge, self.model.beta)
        costs = torch.add(costs, 0.001)

        reward = torch.sub(sim, costs)

        if t > 0:
            reward += torch.sum(
                torch.softmax(torch.abs(self.feats[t] - self.feats[t - 1]),
                              dim=1),
                dim=1,
            ) * self.model.gamma

        loss = -reward.sum()
        loss.backward()
        self.optimizer.step()

    def export_param(self):
        with open("model.param.data.fast", "w") as f:
            max_alpha = 1.0
            max_beta = 1.0
            max_gamma = 1.0

            for i in range(self.size):
                f.write("{},{},{},{}\n".format(
                    self.model.alpha[i].item() / max_alpha,
                    self.model.beta[i].item() / max_beta,
                    self.model.gamma[i].item() / max_gamma,
                    1.0,
                ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset", type=str, default="NIPS")
    args = parser.parse_args()
    DEVICE = torch.device(args.device)
    DATASET = args.dataset
    data = init_real_data(DATASET, device=DEVICE)
    data_size = len(data.adj[0])

    alpha = torch.ones(data_size, dtype=torch.float32).to(DEVICE)
    beta = torch.ones(data_size, dtype=torch.float32).to(DEVICE)
    gamma = torch.ones(data_size, dtype=torch.float32).to(DEVICE)
    delta = torch.ones(data_size, dtype=torch.float32).to(DEVICE)

    model = Model(alpha, beta, gamma, delta, device=DEVICE)
    optimizer = Optimizer(data.adj,
                          data.feature,
                          model,
                          data_size,
                          device=DEVICE)

    for t in range(5):
        optimizer.optimize(t)

    optimizer.export_param()
