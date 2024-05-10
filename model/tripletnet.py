import torch
import torch.nn as nn
import torch.nn.functional as F


# 無音判定なし
# input emb
class CS_Tripletnet(nn.Module):
    def __init__(self, csnnet):
        super(CS_Tripletnet, self).__init__()
        self.csnnet = csnnet

    def forward(self, emb_a, emb_p, emb_n, c):
        masked_embedded_a = self.csnnet(emb_a, c)
        masked_embedded_p = self.csnnet(emb_p, c)
        masked_embedded_n = self.csnnet(emb_n, c)

        dist_p = F.pairwise_distance(masked_embedded_a, masked_embedded_p, 2)
        dist_n = F.pairwise_distance(masked_embedded_a, masked_embedded_n, 2)

        return dist_p, dist_n