#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from configs import config
from evaluation import Metrics
from methods.methods import Methods
from datasets.dataset_jartest import JarTestDataset


def main():
    # 乱数固定
    torch.manual_seed(42)
    np.random.seed(42)
    
    # パーサーの読み込み
    arg = config.config()

    print("=== arg ===")
    print(arg)
    print()

    # 結果の保存ディレクトリの生成
    os.makedirs(arg['res_dir'], exist_ok=True)

    # GPUの設定
    device = torch.device(arg['device'] if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    # データセットの選択と読み込み
    dataset_train = JarTestDataset(
        is_train=True,
        n_frames_input=arg['T_in'],
        n_frames_output=arg['T_out'], 
        valid_id=arg['valid_id'])
    
    dataset_valid = JarTestDataset(
        is_train=False, 
        n_frames_input=arg['T_in'],
        n_frames_output=arg['T_out'], 
        valid_id=arg['valid_id'])

    print('video_id_array: ', dataset_valid.get_video_id_array())

    # データローダーの生成
    dataloader_train = DataLoader(dataset_train, batch_size=1)
    dataloader_valid = DataLoader(dataset_valid, batch_size=1)
    
    # メソッドの選択
    method = Methods.get(arg).to(device)
    
    print('train')
    for dataset_train in tqdm.tqdm(dataloader_train):
        data_in, data_true = dataset_train
        method(data_in.to(device), data_true.to(device), is_train=True)
        # break
    method.fit()
    
    print('valid')
    np_dataset_pred, np_dataset_true = [], []
    i = -1
    for dataset_valid in tqdm.tqdm(dataloader_valid):
        i += 1
        # if i != 9:
        #     continue
        data_in, data_true = dataset_valid
        data_pred = method(data_in.to(device), data_true.to(device))
        np_data_pred = data_pred.to('cpu').detach().numpy().copy()
        np_data_true = data_true.to('cpu').detach().numpy().copy()
        np_dataset_pred.append(np_data_pred)
        np_dataset_true.append(np_data_true)
    np_dataset_pred = np.concatenate(np_dataset_pred, axis=0)
    np_dataset_true = np.concatenate(np_dataset_true, axis=0)
    
    # 精度評価
    metrics = Metrics()
    metrics.update(np_dataset_pred, np_dataset_true)
    print('ave(ssim, psnr, rmse): ', metrics.get_ave())
    print('std(ssim, psnr, rmse): ', metrics.get_std())

    # 予測結果の保存
    print("saving result")
    os.makedirs(os.path.join(arg['res_dir'], 'saved'), exist_ok=True)
    np.save(os.path.join(arg['res_dir'], 'saved', f'preds.npy'), np_dataset_pred)
    np.save(os.path.join(arg['res_dir'], 'saved', f'trues.npy'), np_dataset_true)


if __name__ == '__main__':
    main()
