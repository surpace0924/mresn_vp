#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from skimage.metrics import structural_similarity as cal_ssim
import statistics
import pandas as pd

class Evalution():
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.ssim_list = []
        self.psnr_list = []
        self.rmse_list = []
        self.ssim_ave = 0
        self.psnr_ave = 0
        self.rmse_ave = 0
        self.ssim_std = 0
        self.psnr_std = 0
        self.rmse_std = 0

    def update(self, video_pred, video_true):
        # 画像ごとに評価値を計算し平均を映像の評価値とする
        ssim, psnr, rmse = 0.0, 0.0, 0.0
        for img_pred, img_true in zip(video_pred, video_true):
            ssim += Metrics.SSIM(img_pred, img_true) / len(video_pred)
            psnr += Metrics.PSNR(img_pred, img_true) / len(video_pred)
            rmse += Metrics.RMSE(img_pred, img_true) / len(video_pred)
        
        # 映像の評価値リストに追加
        self.ssim_list.append(ssim)
        self.psnr_list.append(psnr)
        self.rmse_list.append(rmse)

        # 平均と分散を更新
        self.ssim_ave = sum(self.ssim_list)/len(self.ssim_list)
        self.psnr_ave = sum(self.psnr_list)/len(self.psnr_list)
        self.rmse_ave = sum(self.rmse_list)/len(self.rmse_list)

        if len(self.ssim_list) > 1:
            self.ssim_std = statistics.stdev(self.ssim_list)
            self.psnr_std = statistics.stdev(self.psnr_list)
            self.rmse_std = statistics.stdev(self.rmse_list)

    def get_df(self):
        data = {
            'ssim': self.ssim_list,
            'psnr': self.psnr_list,
            'rmse': self.rmse_list
        }
        return pd.DataFrame(data)



class Metrics:
    def __init__(self):
        self.ssim_mat = np.array([])
        self.psnr_mat = np.array([])
        self.rmse_mat = np.array([])

    def update(self, dataset_pred, dataset_true):
        ssim_list, psnr_list, rmse_list = [], [], []
        for video_pred, video_true in zip(dataset_pred, dataset_true):
            for img_pred, img_true in zip(video_pred, video_true):
                # 画像とに計算
                ssim = Metrics.SSIM(img_pred, img_true)
                psnr = Metrics.PSNR(img_pred, img_true)
                rmse = Metrics.RMSE(img_pred, img_true)
            
                # 映像の評価値リストに追加
                ssim_list.append(ssim)
                psnr_list.append(psnr)
                rmse_list.append(rmse)
        self.ssim_mat = np.array(ssim_list)
        self.psnr_mat = np.array(psnr_list)
        self.rmse_mat = np.array(rmse_list)

    def get_ave(self):
        ssim_ave = np.average(self.ssim_mat)
        psnr_ave = np.average(self.psnr_mat)
        rmse_ave = np.average(self.rmse_mat)
        return ssim_ave, psnr_ave, rmse_ave
    
    def get_std(self):
        ssim_std = np.std(self.ssim_mat.flatten())
        psnr_std = np.std(self.psnr_mat.flatten())
        rmse_std = np.std(self.rmse_mat.flatten())
        return ssim_std, psnr_std, rmse_std

    @staticmethod
    def PSNR(img_pred, img_true):
        mse = np.mean((np.uint8(img_pred * 255)-np.uint8(img_true * 255))**2)
        return 20 * np.log10(255) - 10 * np.log10(mse)

    @staticmethod
    def MSE(img_pred, img_true):
        return np.mean((255*img_pred-255*img_true)**2)

    @staticmethod
    def RMSE(img_pred, img_true):
        return np.sqrt(Metrics.MSE(img_pred, img_true))

    @staticmethod
    def SSIM(img_pred, img_true):
        img_pred = np.clip(img_pred*255, 0, 255).astype(np.uint8)
        img_true = np.clip(img_true*255, 0, 255).astype(np.uint8)
        return cal_ssim(img_pred, img_true, channel_axis = 0)
