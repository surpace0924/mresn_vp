#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gc
import torch
import torchvision.transforms.functional as F
from models.torch_esn.esn import ESN
import torch.nn as nn

class NextFramePredictionModule(torch.nn.Module):
    def __init__(self, H, W, C, node_num = 500, trans_len = 0) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.H = H
        self.W = W
        self.C = C
        self.node_num = node_num
        self.trans_len = trans_len

        # ESN構築
        self.esns = []
        for c in range(C):
            self.esns.append(ESN(
                N_u = H*W,
                N_x = self.node_num,                 
                N_y = H*W,
                density=0.05,  
                input_scale=1.0,
                rho=0.95,
                leaking_rate=0.95,
                regularization_rate=1.0))
    
    # リザバーの訓練
    def fit(self, data_in) -> None:
        device = self.dummy.device
        N_b, T, C, H, W = data_in.shape

        # [N_b, T, C, H, W] -> [C, T, N_b, HW]
        inputs = data_in.permute(2, 1, 0, 3, 4).to(device)
        inputs = torch.reshape(inputs, (C, T, N_b, H*W))

        # チャネルごとに予測
        for c, u_array in enumerate(inputs):
            # [T-1, N_b, HW]
            u = u_array[:T-1]
            d = u_array[1:]
            
            self.esns[c].to(device)
            self.esns[c](u, self.trans_len, d)
            self.esns[c].fit()
                    

    # 映像を入力として次のフレーム画像を予測する
    # input: [N_b, T, C, H, W]
    # output: [N_b, C, H, W]
    def forward(self, data_in):
        N_b, T, C, H, W = data_in.shape

        # [N_b, T, C, H, W] -> [C, T, N_b, HW]
        inputs = data_in.permute(2, 1, 0, 3, 4)
        inputs = torch.reshape(inputs, (C, T, N_b, H*W))

        # チャネルごとに予測
        outputs = []
        for c, u_array in enumerate(inputs):
            output, x_n = self.esns[c](u_array, 0)
            outputs.append(torch.unsqueeze(output[-1], dim=0))
        outputs = torch.cat(outputs)
        
        next_frames = torch.reshape(outputs, (C, N_b, H, W))
        next_frames = next_frames.permute(1, 0, 2, 3)
        del outputs
        return next_frames



class ImageFusionModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        device = self.dummy.device

        C = 3

        # 重み計算用行列
        self.design_mat = None
        self.target_vec = None

        # 重み
        k = torch.zeros(C, 2).to(device)
        self.k = nn.Parameter(k, requires_grad=True)


    def fit(self):
        device = self.dummy.device
        C = self.design_mat.size()[0]

        k_list = torch.arange(0.0, 1.0, 0.5)
        k_res = torch.zeros(C, 2).to(device)
        
        for i, (A, b) in enumerate(zip(self.design_mat, self.target_vec)):
            min_E = float('inf')
            min_idx = 0
            for k_idx, k in enumerate(k_list):
                x = k*A[0] + (1.0-k)*A[1]
                E = torch.sum((x - b)**2) 
                if E < min_E:
                    min_E = E   
                    min_idx = k_idx
            # k_res[i][0] = k_list[min_idx]
            # k_res[i][1] = 1.0 - k_list[min_idx]
            k_res[i][0] = 0.5
            k_res[i][1] = 1.0 - k_res[i][0]
        # 更新
        self.k = nn.Parameter(k_res, requires_grad=True)
        
        
    # 画像を入力として画像を出力する
    # input dim: (N_b, C, H, W)
    # res_a > res_b
    def forward(self, img_a, img_b, img_d = None) -> torch.Tensor:
        device = self.dummy.device
        N_b, C, H_a, W_a = img_a.size()
        N_b, C, H_b, W_b = img_b.size()

        # res_a になるようにアップサンプリング
        img_b_upsampled = img_b.reshape(N_b*C, H_b, W_b)
        img_b_upsampled = F.resize(img=img_b_upsampled, size=(H_a, W_a))
        img_b_upsampled = img_b_upsampled.reshape(N_b, C, H_a, W_a)

        if img_d is not None:
            # チャネルを先頭に
            img_a_channel_first = img_a.permute(1, 0, 2, 3)
            img_b_upsampled_channel_first = img_b_upsampled.permute(1, 0, 2, 3)
            img_d_channel_first = img_d.permute(1, 0, 2, 3)

            # 行ごとに真ん中の値を抽出
            va = img_a_channel_first[:, :, :, int(W_a/2)]
            vb = img_b_upsampled_channel_first[:, :, :, int(W_a/2)]
            vd = img_d_channel_first[:, :, :, int(W_a/2)]
            del img_a_channel_first, img_b_upsampled_channel_first, img_d_channel_first
            gc.collect()

            # 真ん中に新しい軸を追加 dim=(C, N_b*res) -> (C, 1, N_b*res)
            va = torch.unsqueeze(va, dim=1)
            vb = torch.unsqueeze(vb, dim=1)
            vd = torch.unsqueeze(vd, dim=1)
            
            # 2つの画像を並べた行列
            # dim=(C, 2, N_b*res)
            design_mat = torch.cat((va, vb), dim=1)
            self.design_mat = design_mat
            del va, vb, design_mat
            gc.collect()

            # 正解の画素値を並べた行列
            # dim=(C, N_b*H_a*W_a)
            self.target_vec = vd
            del vd
            gc.collect()

        # 画像融合
        img_fured = torch.zeros(N_b, C, H_a, W_a).to(device)
        for c in range(C):
            img_fured[:, c, ...] = self.k[c][0]*img_a[:, c, ...] + self.k[c][1]*img_b_upsampled[:, c, ...]
        del img_b_upsampled
        gc.collect()
        return img_fured                





class MR_ESN_VP(torch.nn.Module):
    def __init__(self, arg) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        device = self.dummy.device


        self.T_out = arg['T_out']
        H = arg['H']
        W = arg['W']

        self.resolution_list = sorted([
            int(H/3.0),
            int(H/2.5),
            int(H/2.0),
            int(H/1.5),
            H,
        ])

        # 解像度別に次フレーム予測モジュールを構築
        self.nfpm_list = [
            NextFramePredictionModule(res, res, 3).to(device) for res in self.resolution_list 
        ]
        
        # 解像度別に画像融合モジュールを構築 (予測解像度-1個になる)
        self.ifm_list = [
            ImageFusionModule().to(device) for _ in range(len(self.resolution_list)-1)
        ]

    
    # 学習
    def fit(self) -> None:
        for ifm in self.ifm_list:
            ifm.fit()

        C=3
        for r, ifm in enumerate(self.ifm_list):
            k_res = torch.zeros(C, 2).to(self.dummy.device)
            for i in range(C):
                if r == 0:
                    k_res[i][0] = 0.1
                elif r == 1:
                    k_res[i][0] = 0.1
                elif r == 2:
                    k_res[i][0] = 0.1
                else:
                    k_res[i][0] = 0.8
                k_res[i][1] = 1.0 - k_res[i][0]
            self.ifm_list[r].k = nn.Parameter(k_res, requires_grad=True)
    

    # 入力映像から映像予測を行い，将来の映像を返す
    def forward(self, data_in, data_out, is_train=False):
        device = self.dummy.device
        N_b, T_in, C, H, W = data_in.size()
        
        # 解像度の数
        R = len(self.nfpm_list)

        # 次フレーム予測モジュールを学習
        for r, res in enumerate(self.resolution_list):
            self.nfpm_list[r].to(device)

            # リサイズ
            data_in_resized = data_in.reshape(N_b*T_in*C, H, W)
            data_in_resized = F.resize(img=data_in_resized, size=(res, res))
            data_in_resized = data_in_resized.reshape(N_b, T_in, C, res, res)

            # 学習
            self.nfpm_list[r].fit(data_in_resized.to(device))
        

        # 次フレームの予測を繰り返して映像を予測
        for t in range(self.T_out):
            N_b, T, C, H, W = data_in.size()   

            # 解像度ごとに次フレームを予測
            next_frame_list = []
            for r in range(R):
                res = self.nfpm_list[r].H
        
                # リサイズ
                data_in_resized = self.resize_dataset(data_in, res, res)
   
                # 次フレーム予測
                self.nfpm_list[r].fit(data_in_resized)
                next_frame = self.nfpm_list[r](data_in_resized)
                next_frame = torch.clamp(next_frame, min=0.0, max=1.0)
                next_frame_list.append(next_frame)

            # 解像度の異なる2つの画像を融合
            for r in range(R-1):
                # res_a > res_b
                res_a, res_b = self.resolution_list[r+1], self.resolution_list[r]
                img_a, img_b = next_frame_list[r+1], next_frame_list[r]
                data_true = self.resize_dataset(data_out, res_a, res_a)[:, t, ...]

                self.ifm_list[r].to(device)
                if is_train:
                    self.ifm_list[r](img_a, img_b, data_true)
                    I_fused = data_true
                else:
                    I_fused = self.ifm_list[r](img_a, data_true)
                I_fused = torch.clamp(I_fused, min=0.0, max=1.0)

            # 予測したフレームを時間方向に結合
            I_fused = torch.unsqueeze(I_fused, dim=1)
            data_in = torch.cat((data_in, I_fused), dim=1)
        del I_fused
        gc.collect()
        return data_in[:, T_in:, ...]


    # データセットのリサイズ
    # dim: (N, T, C, H, W) -> (N, T, C, toH, toW)
    def resize_dataset(self, dataset, toH, toW):
        N, T, C, H, W = dataset.shape
        dataset_resized = dataset.reshape(N*T*C, H, W)
        dataset_resized = F.resize(img=dataset_resized, size=(toH, toW))
        dataset_resized = dataset_resized.reshape(N, T, C, toH, toW)
        return dataset_resized
