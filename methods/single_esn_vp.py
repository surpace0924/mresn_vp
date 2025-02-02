#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from models.torch_esn.esn import ESN

class NextFramePredictionModule(torch.nn.Module):
    def __init__(self, H, W, C, node_num = 500, trans_len = 10) -> None:
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
    def train(self, data_in) -> None:
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
        return next_frames


class Single_ESN_VP(torch.nn.Module):
    def __init__(self, arg) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        device = self.dummy.device

        self.T_out = arg['T_out']
        H = arg['H']
        W = arg['W']

        # 次フレーム予測モジュールを構築
        self.nfpm = NextFramePredictionModule(H, W, 3).to(device)
    
    # 不要
    def fit(self) -> None:
        pass
    
    # 入力映像から映像予測を行い，将来の映像を返す
    def forward(self, data_in, data_out):
        device = self.dummy.device
        self.nfpm.to(device)

        # [N_b, T, C, H, W]
        N_b, T, C, H, W = data_in.size()

        # 次フレーム予測モジュールを学習
        self.nfpm.train(data_in.to(device))

        # 次フレームの予測を繰り返して映像を予測
        for t in range(self.T_out):
            # 次フレーム予測
            self.nfpm.train(data_in)
            next_frame = self.nfpm(data_in)
            next_frame = torch.clamp(next_frame, min=0.0, max=1.0)

            # 予測したフレームを時間方向に結合
            next_frame = torch.unsqueeze(next_frame, dim=1)
            data_in = torch.cat((data_in, next_frame), dim=1)

        # 全映像のうち予測した区間の映像のみを返す
        return data_in[:, T:, ...]