from typing import Tuple, Union, cast, Optional, Sequence, Any

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np


TRANSFORMS_REQUIRED_UPDATE = ["vanilla_sgd"]
TRANSFORMS_REQUIRED_LOSS = ["vanilla_sgd"]

class Vanilla_VectorQuantizer(nn.Module):
    def __init__(self, number_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99,
                 n_codebooks=17, update_codebook=True, epsilon=1e-5, reduction=True):
        super(Vanilla_VectorQuantizer, self).__init__()
        self._transform_name = "vanilla_sgd"

        self.number_embeddings = number_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.decay = decay
        self.epsilon = epsilon

        self.codebooks = nn.Parameter(torch.Tensor(number_embeddings, embedding_dim))
        self.codebooks.data.normal_()

        self._update_codebook = update_codebook
        self.reduction = reduction

    def enable_update_codebook(self):
        self._update_codebook = True

    def disable_update_codebook(self):
        self._update_codebook = False

    def forward(self, z):
        B, D = z.shape  # BxD

        # z = z.unsqueeze(2)  # BxDx1
        # Flatten input
        flat_input = z.view(-1, self.embedding_dim)     # BxD

        # Z_mat = z.repeat(1, 1, self.number_embeddings)  # BxDx1 -> BxDxK
        # E_mat = self.codebooks.unsqueeze(0).repeat(B, 1, 1)  # DxK   -> BxDxK

        # distances from z to embeddings e_j
        # distances = (Z_mat - E_mat) ** 2  # BxDxK

        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.codebooks ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.codebooks.t()))

        # find closest encodings
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # BxDx1
        encodings = torch.zeros(encoding_indices.shape[0], self.number_embeddings, device=z.device)  # BxDxK
        encodings.scatter_(1, encoding_indices, 1)  # one-hot: BxDxK

        # get quantized latent vectors
        # quantized = torch.sum(encodings * self.codebooks, dim=2, keepdim=True)  # BxDx1
        quantized = torch.matmul(encodings, self.codebooks).view(z.shape)

        if self.training and self._update_codebook:
            q_latent_loss = F.mse_loss(quantized, z.detach())
        else:
            q_latent_loss = F.mse_loss(quantized.detach(), z.detach())

        # compute loss for embedding
        # e_latent_loss = F.mse_loss(quantized.detach(), z)
        # loss = self.commitment_cost * e_latent_loss
        loss = q_latent_loss

        # Straight Through Estimator
        quantized = z + (quantized - z).detach()

        # perplexity
        # avg_probs = torch.mean(encodings, dim=0)
        # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized


class BitDepthReduction(nn.Module):
    def __init__(self, step=0.1):
        super(BitDepthReduction, self).__init__()
        self._transform_name = "bdr"

        self.step = step

    def forward(self, z):
        quantized = torch.div(z, self.step, rounding_mode="floor") * self.step

        # Straight Through Estimator
        quantized = z + (quantized - z).detach()
        loss = torch.tensor(-1.0)

        return loss, quantized


class AEDenoiser(nn.Module):
    def __init__(self, input_dim):
        super(AEDenoiser, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dim, 128), nn.ReLU(),
        #     nn.Linear(128, 64), nn.ReLU(),
        #     nn.Linear(64, 32), nn.ReLU(),
        # )
        # self.decoder = nn.Sequential(
        #     nn.Linear(32, 64), nn.ReLU(),
        #     nn.Linear(64, 128), nn.ReLU(),
        #     nn.Linear(128, input_dim)
        # )

    def forward(self, x):
        h = self.encoder(x)
        x_rec = self.decoder(h)
        return x_rec

    def denoise(self, x):
        h = self.encoder(x)
        x_rec = self.decoder(h)
        return x_rec

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


class VAEDenoiser(nn.Module):
    def __init__(self, input_dim, init_w=1e-3, min_variance=None):
        super(VAEDenoiser, self).__init__()

        if min_variance is None:
            self.log_min_variance = None
        else:
            self.log_min_variance = float(np.log(min_variance))

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )

        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 32)

        self.fc1.weight.data.uniform_(-init_w, init_w)
        self.fc1.bias.data.uniform_(-init_w, init_w)

        self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc2.bias.data.uniform_(-init_w, init_w)

        self.decoder = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim)
        )


        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dim, 128), nn.ReLU(),
        #     nn.Linear(128, 64), nn.ReLU(),
        #     nn.Linear(64, 32), nn.ReLU(),
        # )
        # self.decoder = nn.Sequential(
        #     nn.Linear(32, 64), nn.ReLU(),
        #     nn.Linear(64, 128), nn.ReLU(),
        #     nn.Linear(128, input_dim)
        # )

    def rsample(self, latent_distribution_params):
        mu, logvar = latent_distribution_params
        stds = (0.5 * logvar).exp()
        epsilon = torch.randn(*mu.size(), device=mu.device)
        latents = epsilon * stds + mu
        return latents

    def reparameterize(self, latent_distribution_params):
        if self.training:
            return self.rsample(latent_distribution_params)
        else:
            return latent_distribution_params[0]

    def kl_divergence(self, latent_distribution_params):
        mu, logvar = latent_distribution_params
        return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def forward(self, x):
        latent_distribution_params = self.encode(x)
        latents = self.reparameterize(latent_distribution_params)
        reconstructions, obs_distribution_params = self.decode(latents)
        return reconstructions, obs_distribution_params, latent_distribution_params

    def denoise(self, x):
        latent_distribution_params = self.encode(x)
        latents = latent_distribution_params[0]
        reconstructions, _ = self.decode(latents)
        return reconstructions

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc1(h)
        if self.log_min_variance is None:
            logvar = self.fc2(h)
        else:
            logvar = self.log_min_variance + torch.abs(self.fc2(h))
        return (mu, logvar)

    def decode(self, z):
        decoded = self.decoder(z)
        return decoded, [decoded, torch.ones_like(decoded)]

    def logprob(self, x, obs_distribution_params):
        log_prob = -1 * F.mse_loss(x, obs_distribution_params[0])
        return log_prob
