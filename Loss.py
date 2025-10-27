import torch.nn.functional as F
from tools import *
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def mask_correlated_samples(N):
    mask = torch.ones((N, N),device=device)
    mask = mask.fill_diagonal_(0)
    for i in range(N//2):
        mask[i, N//2 + i] = 0
        mask[N//2 + i, i] = 0
    mask = mask.bool()
    return mask

def orthogonal_loss_2(B_s_list, B_t_list, device,view):

    N=B_s_list[0].shape[0]
    B_s_norm = [F.normalize(B, p=2, dim=1, eps=1e-10) for B in B_s_list]
    B_t_norm = [F.normalize(B, p=2, dim=1, eps=1e-10) for B in B_t_list]
    #  [view, N, d]
    B_s_all = torch.stack(B_s_norm, dim=0) # shape: [V, N, d]
    B_t_all = torch.stack(B_t_norm, dim=0)  # shape: [V, N, d]
    dot_products = torch.einsum('vnd,vnd->vn', B_s_all, B_t_all)
    loss = torch.sum(dot_products) / N
    return loss

def cosine_alignment_loss_batched(Q, Bs_list, view, use_diag_mask=True, batch_size=512):
    n_samples = Q.shape[0]
    device = Q.device
    total_loss = torch.tensor(0.0, device=device)
    tmp_B_list = []
    tmp_B_t_list = []
    # precompute all B matrices' transpose
    for i in range(view):
        tmp_B_list.append(torch.nn.functional.normalize(Bs_list[i]).to(device))
        tmp_B_t_list.append((torch.nn.functional.normalize(Bs_list[i])).t().to(device))

    Q=torch.nn.functional.normalize(Q,dim=1)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = slice(start_idx, end_idx)
        batch_size_cur = end_idx - start_idx


        Q_batch = Q[batch_indices]
        S_q_batch = torch.mm(Q_batch, Q.t()) # [batch_size_cur, n_samples]


        if use_diag_mask:
            mask = torch.ones((batch_size_cur, n_samples), device=device)
            rows = torch.arange(batch_size_cur, device=device)
            cols = start_idx + torch.arange(batch_size_cur, device=device)
            mask[rows, cols] = 0
            S_q_batch = torch.mul(S_q_batch, mask)

        S_q_norm_batch = F.normalize(S_q_batch, p=2, dim=1, eps=1e-10)

        for i in range(view):
            B_batch = tmp_B_list[i][batch_indices]
            S_b_batch = torch.mm(B_batch, tmp_B_t_list[i])  # [batch_size_cur, n_samples]
            if use_diag_mask:
                S_b_batch = torch.mul(S_b_batch, mask)
            S_b_norm_batch = F.normalize(S_b_batch, p=2, dim=1, eps=1e-10)
            cos_sim = torch.sum(torch.mul(S_q_norm_batch, S_b_norm_batch), dim=1)  # [batch_size_cur]
            total_loss =  total_loss + torch.sum(1 - cos_sim)

    return   total_loss / n_samples


def loss_contrastive_learning(q_g, Ur, U_list, view, temperature_f, device, k, poch):
    loss_list = []
    M = U_list[0].shape[0]  # the number of samples
    criterion = nn.CrossEntropyLoss(reduction="sum")
    if poch == 0:
        S = construct_S(q_g, k).to(device)
    elif poch == 1:
        S = construct_S2(q_g).to(device)
    elif poch == 2:
        S = construct_S3(q_g).to(device)

    # construct the structure-guided similarity matrix S (M x M)
    batch_size=512
    for v in range(view):
        for i in range(2):
            if i == 0:
                for start_idx in range(0, M, batch_size):
                    end_idx = min(start_idx + batch_size, M)
                    batch_indices = slice(start_idx, end_idx)
                    batch_size_cur = end_idx - start_idx

                    row_idx = torch.arange(start_idx,end_idx, device=device).view(-1, 1)  # [M,1]

                    h_i=U_list[v][batch_indices]

                    candidates = torch.cat([U_list[v], Ur], dim=0)  # [M,2*M]

                    sim = torch.mm(h_i, candidates.t()) / temperature_f

                    pos_sim = sim.gather(1, torch.arange(M+start_idx, M+end_idx, device=device).view(-1,1))  #

                    self_mask = torch.zeros_like(sim, dtype=torch.bool, device=device)
                    self_mask.scatter_(1, row_idx, True)
                    self_mask.scatter_(1, row_idx + M, True)
                    neg_mask = ~self_mask

                    S_neg = 1 - S[batch_indices]
                    weight_matrix = torch.cat([
                        S_neg,
                        S_neg,
                    ], dim=1)

                    weighted_sim = sim * weight_matrix
                    neg_sim = weighted_sim[neg_mask].view(batch_size_cur, 2 * M - 2)
                    logits = torch.cat([pos_sim, neg_sim], dim=1)
                    labels = torch.zeros(batch_size_cur, dtype=torch.long, device=device)
                    loss_item = criterion(logits, labels)
                    loss_list.append(loss_item)
            if i==1:
                for start_idx in range(0, M, batch_size):
                    end_idx = min(start_idx + batch_size, M)
                    batch_indices = slice(start_idx, end_idx)
                    batch_size_cur = end_idx - start_idx

                    row_idx = torch.arange(start_idx,end_idx, device=device).view(-1, 1)  # [M,1]
                    h_i=Ur[batch_indices]
                    candidates = torch.cat([U_list[v], Ur], dim=0)  # [M,2*M]
                    sim = torch.mm(h_i, candidates.t()) / temperature_f
                    pos_sim = sim.gather(1, torch.arange(start_idx, end_idx, device=device).view(-1,1))

                    self_mask = torch.zeros_like(sim, dtype=torch.bool, device=device)
                    self_mask.scatter_(1, row_idx, True)
                    self_mask.scatter_(1, row_idx + M, True)
                    neg_mask = ~self_mask

                    S_neg = 1 - S[batch_indices]
                    weight_matrix = torch.cat([
                        S_neg,
                        S_neg,
                    ], dim=1)

                    weighted_sim = sim * weight_matrix
                    neg_sim = weighted_sim[neg_mask].view(batch_size_cur, 2 * M - 2)
                    logits = torch.cat([pos_sim, neg_sim], dim=1)
                    labels = torch.zeros(batch_size_cur, dtype=torch.long, device=device)
                    loss_item = criterion(logits, labels)
                    loss_list.append(loss_item)
    return torch.sum(torch.stack(loss_list))/(2*M)

def loss_reconstruct_init_anchor_graph(recon_b_list, B_list,device,view):
    loss_list=[]
    N=B_list[0].shape[0]
    for v in range(view):
        B = B_list[v].to(device)
        recon_b = recon_b_list[v].to(device)
        #get the non-zero indices of B
        nz_rows, nz_cols = torch.nonzero(B, as_tuple=True)
        if len(nz_rows) == 0:
            loss = torch.tensor(0.0).to(device)
        else:
            # get the non-zero values of B
            B_vals = B[nz_rows, nz_cols]
            recons_vals = recon_b[nz_rows, nz_cols]
            loss_terms = torch.multiply(B_vals, torch.log(recons_vals + 1e-9))
            loss_terms = torch.neg(loss_terms)
            loss = torch.sum(loss_terms)
            loss_list.append(loss)

    return torch.sum(torch.stack(loss_list))/N

def construct_S(q,k):

    q_g=torch.nn.functional.normalize(q,p=2,dim=1)
    S = torch.mm(q_g, q_g.t())  # [N, N]
    # top-k
    topk_vals, topk_idxs = torch.topk(S, k=k, dim=1)
    S_sparse = torch.zeros_like(S,device=device)
    S_sparse = torch.scatter(S_sparse, 1, topk_idxs, topk_vals)  #
    S_sym=(S_sparse+S_sparse.t())/2
    return S_sym