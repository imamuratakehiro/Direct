"""条件付けするモデル"""

import torch
import torch.nn as nn
import numpy as np

from utils.func import bin2list

# GPUが使用可能かどうか判定、使用可能なら使用する
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"\n=== Using {device}({__name__}). ===\n")

class ConditionalSimNet2d(nn.Module):
    def __init__(self, size, device):
        super(ConditionalSimNet2d, self).__init__()
        # maskを作成。embeddingするために１次元化
        mask_init = torch.ones(size).to(device)
        mask = []
        ch = 128
        mask.append(torch.flatten(torch.cat([mask_init[:, :ch, :, :],
                                        torch.zeros(size[0], size[1] - ch, size[2], size[3], device=device)], 1)))
        while (ch + 128) < size[1]:
            mask.append(torch.flatten(torch.cat([torch.cat([torch.zeros(size[0], ch, size[2], size[3], device=device),
                                                    mask_init[:, ch: ch+128, :, :]], 1),
                                        torch.zeros(size[0], size[1] - (ch+128), size[2], size[3], device=device)], 1)))
            ch += 128
        mask.append(torch.flatten(torch.cat([torch.zeros(size[0], ch, size[2], size[3], device=device),
                                        mask_init[:, ch: size[1], :, :]], 1)))
        mask = torch.stack(mask, dim=0)
        # embeddingを定義
        self.masks = torch.nn.Embedding(mask.shape[0], mask.shape[1])
        # 各条件に対するembedding結果(重み)を定義(学習なし)
        self.masks.weight = torch.nn.Parameter(mask, requires_grad=False)
        self.size = size

    def forward(self, input, c):
        mask_c = self.masks(c)
        # 重みとして埋め込むために1次元化していたのを復元
        mask_c = torch.reshape(mask_c, self.size)
        masked_embedding = input * mask_c
        return masked_embedding

class ConditionalSimNet2d768(nn.Module):
    def __init__(self, size, device):
        super(ConditionalSimNet2d768, self).__init__()
        # maskを作成。embeddingするために１次元化
        mask_init = torch.ones(size).to(device)
        mask = torch.stack([torch.flatten(torch.cat([mask_init[:, :128, :, :],
                                        torch.zeros(size[0], 640, size[2], size[3], device=device)], 1)),

                torch.flatten(torch.cat([torch.cat([torch.zeros(size[0], 128, size[2], size[3], device=device),
                                                    mask_init[:, 128: 256, :, :]], 1),
                                        torch.zeros(size[0], 512, size[2], size[3], device=device)], 1)),

                torch.flatten(torch.cat([torch.cat([torch.zeros(size[0], 256, size[2], size[3], device=device),
                                                    mask_init[:, 256: 384, :, :]], 1),
                                        torch.zeros(size[0], 384, size[2], size[3], device=device)], 1)),

                torch.flatten(torch.cat([torch.cat([torch.zeros(size[0], 384, size[2], size[3], device=device),
                                                    mask_init[:, 384: 512, :, :]], 1),
                                        torch.zeros(size[0], 256, size[2], size[3], device=device)], 1)),

                torch.flatten(torch.cat([torch.cat([torch.zeros(size[0], 512, size[2], size[3], device=device),
                                                    mask_init[:, 512: 640, :, :]], 1),
                                        torch.zeros(size[0], 128, size[2], size[3], device=device)], 1)),

                torch.flatten(torch.cat([torch.zeros(size[0], 640, size[2], size[3], device=device),
                                        mask_init[:, 640: 768, :, :]], 1))
        ], dim=0)
        #print(mask.size())
        # embeddingを定義
        self.masks = torch.nn.Embedding(mask.shape[0], mask.shape[1])
        # 各条件に対するembedding結果(重み)を定義(学習なし)
        self.masks.weight = torch.nn.Parameter(mask, requires_grad=False)
        self.size = size

    def forward(self, input, c):
        mask_c = self.masks(c)
        # 重みとして埋め込むために1次元化していたのを復元
        mask_c = torch.reshape(mask_c, self.size)
        masked_embedding = input * mask_c
        return masked_embedding

class ConditionalSimNet1dBatch(nn.Module):
    def __init__(self, batch):
        super(ConditionalSimNet1d, self).__init__()
        # maskを作成。embeddingするために１次元化
        mask_init = torch.ones(batch, 128, device=device)
        mask = torch.stack([torch.flatten(torch.cat([mask_init, torch.zeros(batch, 512, device=device)], dim=1)),

                torch.flatten(torch.cat([torch.cat([torch.zeros(batch, 128, device=device),mask_init], dim=1),
                                        torch.zeros(batch, 384, device=device)], dim=1)),

                torch.flatten(torch.cat([torch.cat([torch.zeros(batch, 256, device=device),mask_init], dim=1),
                                        torch.zeros(batch, 256, device=device)], dim=1)),

                torch.flatten(torch.cat([torch.cat([torch.zeros(batch, 384, device=device),mask_init], dim=1),
                                        torch.zeros(batch, 128, device=device)], dim=1)),

                torch.flatten(torch.cat([torch.zeros(batch, 512, device=device), mask_init], dim=1))
        ], dim=0)
        #print(mask.size())
        # embeddingを定義
        self.masks = torch.nn.Embedding(mask.shape[0], mask.shape[1])
        # 各条件に対するembedding結果(重み)を定義(学習なし)
        self.masks.weight = torch.nn.Parameter(mask, requires_grad=False)
        self.batch = batch

    def forward(self, input, c):
        mask_c = self.masks(c)
        # 重みとして埋め込むために1次元化していたのを復元
        mask_c = torch.reshape(mask_c, (self.batch, 640))
        masked_embedding = input * mask_c
        return masked_embedding

class ConditionalSimNet1d(nn.Module):
    def __init__(self):
        super(ConditionalSimNet1d, self).__init__()
        # maskを作成。embeddingするために１次元化
        mask_init = torch.ones(128)
        mask = torch.stack(
            [(torch.cat([mask_init, torch.zeros(512)], dim=0)),
            (torch.cat([torch.cat([torch.zeros(128),mask_init], dim=0),
                                    torch.zeros(384)], dim=0)),
            (torch.cat([torch.cat([torch.zeros(256),mask_init], dim=0),
                                    torch.zeros(256)], dim=0)),
            (torch.cat([torch.cat([torch.zeros(384),mask_init], dim=0),
                                    torch.zeros(128)], dim=0)),
            (torch.cat([torch.zeros(512), mask_init], dim=0))
        ], dim=0)
        #print(mask.size())
        # embeddingを定義
        self.masks = torch.nn.Embedding(mask.shape[0], mask.shape[1])
        # 各条件に対するembedding結果(重み)を定義(学習なし)
        self.masks.weight = torch.nn.Parameter(mask, requires_grad=False)
        #self.to(device)

    def forward(self, input, c):
        mask_c = self.masks(c)
        masked_embedding = input * mask_c
        return masked_embedding

class ConditionalSimNet1d31cases(nn.Module):
    def __init__(self, cfg, embedding_size=640):
        super(ConditionalSimNet1d31cases, self).__init__()
        conditions = [bin2list(i, len(cfg.inst_list)) for i in range(32)]
        n_category = len(cfg.inst_list)
        # create the mask
        self.n_conditions = len(conditions)  # 条件(組み合わせ含む)数
        # define masks
        self.masks = torch.nn.Embedding(self.n_conditions, embedding_size)
        # initialize masks
        mask_array = np.zeros([self.n_conditions, embedding_size])
        mask_len = int(embedding_size / n_category)
        # make the dimension assigned for each cndition 1
        #for i in range(self.n_conditions):
        #    for j in conditions[i][1]:
        #        mask_array[i, j * mask_len : (j + 1) * mask_len] = 1
        for i in range(self.n_conditions):
            #print(conditions[i])
            for idx, case in enumerate(conditions[i]):
                if case == 1:
                    mask_array[i, idx * mask_len : (idx + 1) * mask_len] = 1

        # no gradients for the masks
        self.masks.weight = torch.nn.Parameter(
            torch.Tensor(mask_array), requires_grad=False
        )

    def forward(self, embedded_x, c):
        # cが条件のtensorなら、まとめてできる
        self.mask = self.masks(c)
        masked_embedding = embedded_x * self.mask
        return masked_embedding

def main():
    #csn = ConditionalSimNet2d((16, 640, 9, 5), "cuda")
    csn1d = ConditionalSimNet1d()
    mask = csn1d.masks(torch.tensor([1], device=device))
    print(mask)
    
if "__main__" == __name__:
    main()