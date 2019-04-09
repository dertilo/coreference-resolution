import torch
from torch import nn as nn
from utils import to_cuda


class FFNN(nn.Module):
    def __init__(self, embeds_dim, hidden_dim=50):# paper says 150
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(embeds_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """ Output a scalar score for an input x """
        return self.score(x)


class DistanceEmbedder(nn.Module):
    """ Learned, continuous representations for: span widths, distance
    between spans
    """

    bins = [1,2,3,4,8,16,32,64]

    def __init__(self, distance_dim=20):
        super().__init__()

        self.dim = distance_dim
        self.embeds = nn.Sequential(
            nn.Embedding(len(self.bins)+1, distance_dim),
            nn.Dropout(0.20)
        )

    def forward(self, *args):
        """ Embedding table lookup """
        return self.embeds(self.stoi(*args))

    def num_bins_smaller_thresh(self,thresh):
        return sum([True for i in self.bins if thresh >= i])

    def stoi(self, lengths):
        tensor = torch.tensor([self.num_bins_smaller_thresh(length) for length in lengths], requires_grad=False)
        return to_cuda(tensor)