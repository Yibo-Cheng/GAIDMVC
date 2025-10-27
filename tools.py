import argparse
import numpy as np
import torch
import random
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def get_pre_q(zs, class_num,seed,  alpha,):
    device = zs[0].device
    # concatenate all views' features
    z_concat = torch.cat(zs, dim=1)  # shape: [n_samples, total_features]
    z_np = z_concat.detach().cpu().numpy()
    # K-means
    kmeans = KMeans(n_clusters=class_num,n_init=10,random_state = seed)
    cluster_assignment = kmeans.fit_predict(z_np)
    centers = kmeans.cluster_centers_  # shape: [class_num, total_features]
    centers = torch.tensor(centers, dtype=z_concat.dtype, device=device)
    # calculate q
    diff = z_concat.unsqueeze(1) - centers.unsqueeze(0)
    dist_sq = torch.sum(torch.pow(diff,2), dim=2)
    numerator = (1.0 + dist_sq / alpha).pow(-(alpha + 1.0) / 2.0)
    denominator = torch.sum(numerator, dim=1, keepdim=True)+1e-10
    q = numerator / denominator
    return q

def distance2(X, Y, square=True):

    # X: [d, n], Y: [d, m]
    n = X.shape[1]
    m = Y.shape[1]
    X_sq = torch.sum(X ** 2, dim=0, keepdim=True).t()  # [n, 1]
    Y_sq = torch.sum(Y ** 2, dim=0, keepdim=True)  # [1, m]
    crossing_term = torch.mm(X.t(), Y)  # [n, m]
    result = X_sq + Y_sq - 2 * crossing_term
    result = torch.clamp(result, min=0)

    return result if square else torch.sqrt(result + 1e-10)
def getB_via_CAN(distances, k):

        n, m = distances.shape
        # get the kth smallest distance for each row and their indices
        k_val = min(k + 1, m)  # solve the case that k is larger than m
        topk_vals, _ = torch.topk(distances, k=k_val, dim=1, largest=False, sorted=True)
        # get the kth smallest distance for each row
        top_k = topk_vals[:, k] if k < m else topk_vals[:, -1]
        top_k = top_k.view(-1, 1).expand(-1, m)  # [n, m]
        # get the sum of the kth smallest distances for each row
        sum_top_k = torch.sum(topk_vals[:, :k], dim=1).view(-1, 1).expand(-1, m)  # [n, m]
        # calculate the weights
        numerator = top_k - distances
        denominator = k * top_k - sum_top_k + 1e-10
        weights = numerator / denominator

        return torch.relu(weights)
def recons_c2(B, embedding):
        f1 = torch.mm(embedding.t(), B)
        Bsum = B.sum(dim=0)  # [m]
        anchors = f1 / (Bsum + 1e-10)
        return anchors.t()


def target_distribution(q,verbose=True):
    weight = q ** 2 / (torch.sum(q, dim=0, keepdim=True)+1e-10)
    p = weight / torch.sum(weight, dim=1, keepdim=True)
    return p

def construct_S2(q):
    q=torch.nn.functional.normalize(q,dim=1)
    S = torch.mm(q, q.t())  # [N, N]
    return S
def construct_S3(q):
    sigma=1
    dist = distance2(q.T, q.T, square=True)
    sim = torch.exp(-dist / (2.0 * sigma ** 2)).float()
    S_sym = (sim + sim) / 2
    return S_sym




