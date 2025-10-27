import torch.backends.cudnn as cudnn
from data_loader import *
from model import *
from metric import *
import os
import time
from tqdm import tqdm
import torch
from copy import deepcopy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def set_seed(seed):
    random.seed(seed)  # set the seed for Python random number generator
    np.random.seed(seed)  # set the seed for NumPy random number generator
    torch.manual_seed(seed)  # set the seed for PyTorch CPU random number generator
    torch.cuda.manual_seed(seed)  # set the seed for current GPU random number generator
    torch.cuda.manual_seed_all(seed)  # set the seed for all GPU random number generator
    # The setting of CuDNN to ensure reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
def train_step(hp1, hp2, anchor, zs, seed, view, class_num, labels,args):
    #training process
    nmi_best=0
    set_seed(seed)  # Set the seed to ensure reproducibility
    args.hp1 = hp1
    args.hp2 = hp2
    args.AnchorNum = int(anchor)
    sigma = 1
    m = args.AnchorNum
    print(f"data: {args.dataset}")
    print(f"anchor: {args.AnchorNum}")
    print(f"hp1 : {args.hp1}")
    print(f"hp2 : {args.hp2}")
    
    anchor_init_list = [None] * view  # the anchors of each view, anchor_init_list[0] denotes the anchors of the first view
    B_list = [None] * view  # the anchor graph of each view, B_list[0] denotes the anchor graph of the first view
    layers = []  # the dim of each view, layers[0] denotes the dim of the sample in first view
    start = time.time()

    #start GAGC
    for t in range(view):
        layers.append(zs[t].shape[1])
    for i in range(view):
        randomNum = random.sample(range(0, zs[i].shape[0]), m)
        anchor_init_list[i] = zs[i][randomNum, :]  # sample anchors
    for v in range(view):
        iterNum = args.num_iter_GAGC
        while (iterNum > 0):
            f = distance2(zs[v].T, anchor_init_list[v].T, True)  # calculate the distance matrix
            B_list[v] = getB_via_CAN(f, class_num)  # update the anchor graph
            anchor_init_list[v] = recons_c2(B_list[v], zs[v])  # update the anchors
            iterNum = iterNum - 1
    #end GAGC

    GAIDMVC = MVC_model(view, layers, args.hidden_dim, args.high_feature_dim, device, class_num, 1).to(device)  # get the model instance
    optimizer = torch.optim.Adam(GAIDMVC.parameters(), lr=args.learning_rate)  # the iterator
    
    q_g = get_pre_q(zs, class_num, seed, alpha=args.alpha)  # init the q_g
    r = target_distribution(q_g).to(device) #init target distribution
    k2 = class_num #the number of the connectable anchors
    loss_list=[]
    loss_list.append(float('-inf'))
    for i in tqdm(range(args.num_iter_GAIDMVC), desc="iterating...: "):
        GAIDMVC.train()
        optimizer.zero_grad()
        recon_b_list, U_list, Us_list, Ut_list, Bt_list, Bs_list, anchor_embedding_list, Ur = GAIDMVC(zs, B_list,anchor_init_list, view, sigma, k2)

        loss_ri= GAIDMVC.loss_reconstruct_information(Bt_list, Bs_list, zs, anchor_embedding_list, view)  # reconstruct information loss
        loss_rg = loss_reconstruct_init_anchor_graph(recon_b_list, B_list, device,view)  # reconstruct init anchor graph loss
        loss_cl = loss_contrastive_learning(q_g, Ur, Us_list, view, args.temperature_f, device,class_num,0)  # contrastive learning loss

        q = GAIDMVC.Cluster(Ur, device)# get the soft label
        
        loss_kl = torch.nn.functional.kl_div(q.log(), r, reduction='batchmean')  # kl divergence loss
        loss_acga = cosine_alignment_loss_batched(q_g, Bs_list, view )  # the anchor graph structure-clustering partition graph alignment mechanism
        loss_or = orthogonal_loss_2(Bs_list, Bt_list, device, view)  # the orthogonal loss
        loss = (loss_cl + loss_rg + loss_ri + loss_kl  + args.hp2 * loss_or+args.hp1*loss_acga)# total loss

        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

        if (i + 1) % 50 == 0:
            GAIDMVC.eval()
            with torch.no_grad():
                _, _, _, _, _, _, _, Ur = GAIDMVC(zs, B_list, anchor_init_list, view, sigma,  k2)
                # update the soft label which is used to guide the contrastive learning and structural alignment mechanism
                q_g = GAIDMVC.Cluster(Ur, device).detach().clone().to(device)

                # update the target distribution which is used to calculate the kl divergence loss
                r=target_distribution(q_g).to(device)

                # Evaluate the clustering performance of clustering guided by q_g and the target distribution R in this round(50 iterations)
                cluster_labels = torch.argmax(q_g, dim=1)

                nmi_tmp = v_measure_score(labels, cluster_labels.cpu().numpy())
                if nmi_tmp > nmi_best:
                    weights = deepcopy(GAIDMVC.state_dict())
                    nmi_best = nmi_tmp
        if (i+1)%200==0:
            k2+=class_num

        if abs(loss_list[-1]-loss_list[-2])<1e-6:
            break

    GAIDMVC.load_state_dict(weights)
    GAIDMVC.eval()
    with torch.no_grad():
        _, _, _, _, _, _, _, Ur = GAIDMVC(zs, B_list, anchor_init_list, view, sigma,k2)
        q_final = GAIDMVC.Cluster(Ur, device)
        cluster_labels = torch.argmax(q_final, dim=1)
    end = time.time()
    times = end - start
    nmi, ari, acc, pur = evaluate(labels, cluster_labels.cpu().numpy())
    return nmi, ari, acc, pur, times
