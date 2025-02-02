#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

class JarTestDataset(Dataset):
    def __init__(self,
                 is_train=True,
                 n_frames_input=10,
                 n_frames_output=10, 
                 valid_id=0):
        super(JarTestDataset, self).__init__()

        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.mean = 0
        self.std = 1

        # 水質データの読み込み
        dataset_dir_path = os.path.join('datasets')
        wq_data_path = os.path.join(dataset_dir_path, 'water_quality.csv')
        df_water = pd.read_csv(wq_data_path)

        # 水質データを最終濁度についてソートし，idのnumpy配列を得る
        idx_array = df_water.sort_values('fin_turbidity')['id'].values

        # idx配列をsplit_num行の2次元配列にする
        # あまりの部分は -1 で埋める
        split_num = 5
        rows = split_num
        cols = math.ceil(len(idx_array)/split_num)
        total_elements = rows * cols
        remainder_array = -1*np.ones(total_elements - len(idx_array))
        idx_array = np.concatenate((idx_array, remainder_array)).astype(np.int32)
        idx_mat = idx_array.reshape(cols, rows).T
        
        # データセットの読み込み
        dataset_path = os.path.join(dataset_dir_path, 'dataset.npy')
        self.dataset = np.load(dataset_path)

        # データセットの時間軸を間引く
        self.dataset = self.dataset[:, ::2, ...]

        # train test に分割
        if is_train:
            # 訓練データはidに対応する行以外を抜き出す
            video_id_array = np.delete(idx_mat, valid_id, 0).flatten()
        else:
            # 検証データはid に対応する行を抜き出す
            video_id_array = idx_mat[valid_id]
            
        # あまりの -1 は削除
        self.video_id_array = video_id_array[video_id_array != -1]
        self.dataset = self.dataset[video_id_array].astype(np.float32)

        # 最終濁度のリストを生成
        self.water = df_water['fin_turbidity'].values[video_id_array]


    def __getitem__(self, idx):
        data = self.dataset[idx]
        input = data[:self.n_frames_input, ...]
        output = data[self.n_frames_input:self.n_frames_input+self.n_frames_output, ...]
        output = torch.from_numpy(output / 255.0).contiguous().float()
        input = torch.from_numpy(input / 255.0).contiguous().float()
        return input, output


    def __len__(self):
        return len(self.dataset)


    def get_video_id_array(self):
        return self.video_id_array
