# Standard Library
from enum import IntEnum

# Third Party Libraries
import torch
import torch.nn as nn
import torch.optim as optim

# First Party Libraries
import config
from init_real_data import init_real_data

device = config.select_device

class Model(nn.Module):

    def __init__(self, alpha, beta, gamma, delta):
        super().__init__()
        self.alpha = nn.Parameter(alpha, requires_grad=True).to(device)
        self.beta = nn.Parameter(beta, requires_grad=True).to(device)
        self.gamma = nn.Parameter(gamma, requires_grad=True).to(device)


class Optimizer:

    def __init__(self, edges, feats, model: Model, size: int):
        self.edges = edges
        self.feats = feats
        self.model = model
        self.size = size
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001)

    def optimize(self, t: int):
        feat = self.feats[t].to(device)
        edge = self.edges[t].to(device)
        self.optimizer.zero_grad()

        dot_product = torch.matmul(feat, feat.t()).to(device)
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
    data = init_real_data()
    data_size = len(data.adj[0])

    alpha = torch.ones(data_size, dtype=torch.float32).to(device)
    beta = torch.ones(data_size, dtype=torch.float32).to(device)
    gamma = torch.ones(data_size, dtype=torch.float32).to(device)
    delta = torch.ones(data_size, dtype=torch.float32).to(device)

    model = Model(alpha, beta, gamma, delta)
    optimizer = Optimizer(data.adj, data.feature, model, data_size)

    for t in range(5):
        optimizer.optimize(t)

    optimizer.export_param()
