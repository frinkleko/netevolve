# Standard Library
from typing import Tuple

# Third Party Library
import torch
import torch.nn as nn


class AgentPolicy(nn.Module):

    def __init__(self, T, e, r, W, m, device) -> None:
        super().__init__()
        self.device = device
        self.T = nn.Parameter(torch.tensor(T).float().to(self.device),
                              requires_grad=True)
        self.e = nn.Parameter(torch.tensor(e).float().to(self.device),
                              requires_grad=True)

        self.r = nn.Parameter(torch.tensor(r).float().view(-1,
                                                           1).to(self.device),
                              requires_grad=True)

        self.W = nn.Parameter(torch.tensor(W).float().view(-1,
                                                           1).to(self.device),
                              requires_grad=True)
        self.m = nn.Parameter(
            torch.tensor(m).float().view(-1, 1).to(self.device), requires_grad=True
        )

    def forward(self, attributes, edges,
                N) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edges = (edges > 0).float().to(self.device)

        tmp_tensor = self.W * torch.matmul(edges, attributes)

        # Computing feat
        feat = self.r * attributes + tmp_tensor * (1 - self.r)
        feat_prob = torch.tanh(feat)
        # Compute similarity
        x = torch.mm(feat, feat.t())
        # print(feat)
        x = torch.tanh(x.div(self.T).exp().mul(self.e))
        # print("prob", x)

        return x, feat, feat_prob

    def forward_neg(self, edges, feat):
        feat = feat.to(self.device)
        x = torch.mm(feat, feat.t())
        x.neg_().add_(1.0)  # Negate and add in place
        x.div_(self.T).exp_().mul_(self.e)  # exp and mul in place
        # x.div_(self.T).exp_()  # exp and mul in place
        x = torch.tanh(x)
        return x

    def predict(self, attributes, edges,
                N) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward(attributes, edges, N)
