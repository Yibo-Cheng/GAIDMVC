from Loss import *
import torch.nn.init as init
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, view, device, dropout=0.1):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )
        self.to(device)
        self.initialize_parameters() 

    def initialize_parameters(self):
    
            init.kaiming_normal_(self.mlp[0].weight, mode='fan_in', nonlinearity='relu')
            if self.mlp[0].bias is not None:
                init.zeros_(self.mlp[0].bias)

            init.kaiming_normal_(self.mlp[4].weight, mode='fan_in', nonlinearity='relu')
            if self.mlp[4].bias is not None:
                init.zeros_(self.mlp[4].bias)

            init.xavier_normal_(self.mlp[8].weight)
            if self.mlp[8].bias is not None:
                init.zeros_(self.mlp[8].bias)

            init.ones_(self.mlp[1].weight)
            init.zeros_(self.mlp[1].bias)
            init.ones_(self.mlp[5].weight)
            init.zeros_(self.mlp[5].bias)

    def forward(self, x):
        return self.mlp(x)


class Decoder(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.decoder = nn.Sequential(

            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        self.to(device)
        self.initialize_parameters()

    def initialize_parameters(self):
        
        init.kaiming_normal_(self.decoder[0].weight, mode='fan_in', nonlinearity='relu')
        if self.decoder[0].bias is not None:
            init.zeros_(self.decoder[0].bias)

        init.xavier_normal_(self.decoder[2].weight)
        if self.decoder[2].bias is not None:
            init.zeros_(self.decoder[2].bias)


    def forward(self, x):
        return self.decoder(x)


class MLp_fusion(nn.Module):
    def __init__(self, high_feature_dim, view, device):
        super(MLp_fusion, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(high_feature_dim * view, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, high_feature_dim),
        )
        self.to(device)
        self.initialize_parameters()

    def initialize_parameters(self):
        
        init.kaiming_normal_(self.mlp[0].weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.mlp[2].weight, mode='fan_in', nonlinearity='relu')

        init.xavier_normal_(self.mlp[4].weight)

        for layer in [self.mlp[0], self.mlp[2], self.mlp[4]]:
            if layer.bias is not None:
                init.zeros_(layer.bias)

    def forward(self, x):
        return self.mlp(x)


class MVC_model(nn.Module):
    def __init__(self, view, input_feature, hidden_dim, high_feature_dim, device, cl_num, alpha):
        super(MVC_model, self).__init__()
        self.device = device
        self.view = view
        self.alpha = alpha

        self.W1 = nn.ModuleList()
        self.W2 = nn.ModuleList()
        self.MLP_s = nn.ModuleList()
        self.MLP_t = nn.ModuleList()
        self.Decoders = nn.ModuleList()


        for v in range(view):
            #init GCN layer
            w1 = nn.Linear(input_feature[v], hidden_dim, bias=True).to(device)
            torch.nn.init.kaiming_normal_(w1.weight, mode='fan_in', nonlinearity='relu') 
            if w1.bias is not None:
                nn.init.zeros_(w1.bias)  
            self.W1.append(w1)
            w2 = nn.Linear(hidden_dim, high_feature_dim, bias=True).to(device)
            torch.nn.init.kaiming_normal_(w2.weight, mode='fan_in', nonlinearity='relu')
            if w2.bias is not None:
                nn.init.zeros_(w2.bias)  
            self.W2.append(w2)
        
            #init decoupling layer
            mlps = MLP(high_feature_dim, hidden_dim, high_feature_dim, view, device).to(device)
            mlpt = MLP(high_feature_dim, hidden_dim, high_feature_dim, view, device).to(device)
            self.MLP_s.append(mlps)
            self.MLP_t.append(mlpt)
            self.Decoders.append(Decoder(high_feature_dim, input_feature[v]).to(device))


    
        self.mlp_fusion = MLp_fusion(high_feature_dim, view, device).to(device)#init fusion layer
        self.clusters = nn.Parameter(torch.empty(cl_num, high_feature_dim, device=device))#init clusters
        init.orthogonal_(self.clusters.data)
        # move all modules to device
        self.to(device)

    def initialize_with_kaiming(self, module_list):
        for module in module_list:
            for sub_module in module.modules():
                if isinstance(sub_module, nn.Linear):
                    nn.init.kaiming_normal_(sub_module.weight, nonlinearity='relu',mode='fan_in')
                    if sub_module.bias is not None:
                        nn.init.zeros_(sub_module.bias)
                # BatchNorm initialization
                elif isinstance(sub_module, nn.BatchNorm1d):
                    nn.init.ones_(sub_module.weight)
                    nn.init.zeros_(sub_module.bias)

    def initialize_mlp_layers(self, mlp_list):
        for mlp in mlp_list:
            for module in mlp.modules():
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_normal_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)

    def forward(self, zs, B_list, anchor_init_list,  view,  sigma,k2):
        U_list = [] #the list of sample embeddings of each view
        Us_list = [] #the list of view-shared factors
        Ut_list = [] #the list of view-specific factors
        Bt_list = [] #the list of view-specific anchor graphs
        Bs_list = [] #the list of view-shared anchor graphs
        anchor_embedding_list = [] #the list of embeddings of anchors
        recon_b_list = [] #the list of reconstructed anchor graphs



        # precompute degree vectors for each view
        degree_vectors = []
        for v in range(view):
            deg = torch.sum(B_list[v], dim=0)
            deg_inv = torch.where(deg > 0, deg.pow(-1), torch.zeros_like(deg))
            degree_vectors.append(deg_inv)

        for v in range(view):
            # first GCN layer
            layer1_out = self.W1[v](zs[v])
            temp = torch.mm(B_list[v].t(), layer1_out)
            temp = degree_vectors[v].unsqueeze(1) * temp
            embedding_sample = torch.mm(B_list[v], temp)
            embedding_sample = torch.relu(embedding_sample)

            # second GCN layer
            layer2_out = self.W2[v](embedding_sample)
            temp = torch.mm(B_list[v].t(), layer2_out)
            temp = degree_vectors[v].unsqueeze(1) * temp
            embedding_sample = torch.mm(B_list[v], temp)
            embedding_sample = F.normalize(embedding_sample, p=2, dim=1, eps=1e-10)
            U_list.append(embedding_sample)

            BtB = torch.mm(B_list[v].t(), B_list[v])
            M = degree_vectors[v].unsqueeze(1) * BtB
            row_sum = torch.sum(M, dim=1)
            Dt_diag = torch.where(row_sum > 0, row_sum.pow(-0.5), torch.zeros_like(row_sum))#[n]
            LA = Dt_diag.unsqueeze(1) * M * Dt_diag.unsqueeze(0)

            anchor_feat = self.W1[v](anchor_init_list[v])
            embedding_anchors = torch.mm(LA, anchor_feat)
            embedding_anchors = torch.relu(embedding_anchors)
            embedding_anchors = self.W2[v](embedding_anchors)
            embedding_anchors = torch.mm(LA, embedding_anchors)
            embedding_anchors = F.normalize(embedding_anchors, p=2, dim=1, eps=1e-10)
            anchor_embedding_list.append(embedding_anchors)

            # -----------compute the view-shared and view-specific factors---------------
            Us = self.MLP_s[v](U_list[v])
            Ut = self.MLP_t[v](U_list[v])
            Us = F.normalize(Us, p=2, dim=1, eps=1e-10)
            Ut = F.normalize(Ut, p=2, dim=1, eps=1e-10)
            Us_list.append(Us)
            Ut_list.append(Ut)

            #----------------reconstruct the anchor graph-----------------
            dists = torch.cdist(embedding_sample, embedding_anchors, p=2)
            dists_sq = torch.pow(dists, 2)
            sim = torch.exp(torch.div(-dists_sq, (2.0 * sigma ** 2)))
            row_sum = torch.sum(sim, dim=1, keepdim=True) + 1e-10
            recon_b = torch.div(sim, row_sum)
            recon_b_list.append(recon_b)

            #--------------------compute the view-shared and view-specific anchor graph-------------------
            ft=distance2(Ut_list[v].t(),anchor_embedding_list[v].t(),square=True)
            bt=getB_via_CAN(ft,k2)
            Bt_list.append(bt)
            fs = distance2( Us_list[v].t(),anchor_embedding_list[v].t(), square=True)
            bs = getB_via_CAN(fs, k2)
            Bs_list.append(bs)


        #-------------------------------compute the fused consensus representation---------------------------------
        Ur = torch.cat(U_list,dim=1)
        Ur = self.mlp_fusion(Ur)
        Ur=torch.nn.functional.normalize(Ur,p=2,dim=1,eps=1e-10)

        return recon_b_list, U_list,Us_list,Ut_list,Bt_list,Bs_list,anchor_embedding_list,Ur

    def Cluster(self, Ur, device):
            eps = 1e-10
            dist_sq = torch.cdist(Ur, self.clusters, p=2)
            dist_sq = torch.pow(dist_sq, 2)
            denom = torch.add(1.0, torch.div(dist_sq, self.alpha))
            q = torch.div(1.0, denom)
            exponent = (self.alpha + 1.0) / 2.0
            q = torch.pow(q, exponent)
            row_sum = torch.sum(q, dim=1, keepdim=True) + eps
            q_norm = torch.div(q, row_sum)
            return q_norm

    def loss_reconstruct_information(self, Bt_list, Bs_list, zs, anchor_embedding_list,view):
        loss_fun = torch.nn.functional.mse_loss
        loss_list=[]
        datasize = zs[0].shape[0]
        for v in range(self.view):
            com_B = torch.add(Bs_list[v], Bt_list[v])  # B_s + B_t
            anchor_comb = torch.matmul(com_B, anchor_embedding_list[v])  # (B_s + B_t) * anchor_embedding
            reconstruct_inf = self.Decoders[v](anchor_comb)
            loss_temp=loss_fun(reconstruct_inf, zs[v],reduction='sum')
            loss_list.append(loss_temp)
        return  torch.sum(torch.stack(loss_list))/datasize






























