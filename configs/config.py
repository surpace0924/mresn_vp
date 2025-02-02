#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import json

# 1. パーサから設定ファイルのパスを取得
# 2. 設定ファイルの内容を読み込み
# 3. 設定ファイルの内容をパーサの記述で上書き
def config():
    # argパース
    args = vars(parser().parse_args())
    
    # 設定ファイルの読み込み
    config_file = args['config_file']
    with open(config_file) as file:
        config = json.load(file)

    # 上書き処理
    arg_key_names = ['config_file', 'omit', 'Height', 'Width', 'Channnel', 'Comp', 'Nx', 'T_in', 'T_out', 'valid_id', 'ex_name', 'device']
    config_key_names = ['config_file', 'omit', 'H', 'W', 'C', 'D', 'Nx', 'T_in', 'T_out', 'valid_id', 'ex_name', 'device']
    for arg_key_name, config_key_name in zip(arg_key_names, config_key_names):
        if args[arg_key_name] != None:
            config[config_key_name] = args[arg_key_name]
    
    # 結果の保存ディレクトリの生成
    # res_dir_name = f"{config['dataset']}_{config['method'].replace('_', '')}_{str(config['T_in']).zfill(2)}_{str(config['T_out']).zfill(2)}"
    config['res_dir'] = os.path.join('.', 'results', config['ex_name'])
    
    return config
    

def parser():
    parser = argparse.ArgumentParser(description='Video predictor based on ESN')
    
    parser.add_argument(
        '--config_file', '-c', 
        default='configs/jartest/mr_esn_vp.json', 
        type=str,
        help='Path to the default config file')
    
    parser.add_argument(
        '--omit', '-o', 
        default=False, 
        type=bool,
        help='Run in test mode?')
    
    parser.add_argument(
        '--Height', 
        type=int,
        help='(H) Height of video')
    
    parser.add_argument(
        '--Width', 
        type=int,
        help='(W) Width of video')
    
    parser.add_argument(
        '--Channnel',
        type=int,
        help='(C) Channnel num of video')
    
    parser.add_argument(
        '--Comp',
        type=int,
        help='(D) Compression dimension when dimension reduction is used')
    
    parser.add_argument(
        '--Nx',
        type=int,
        help='Node num of reservoir')
    
    parser.add_argument(
        '--T_in',
        type=int,
        help='Num of video input frames')
    
    parser.add_argument(
        '--T_out',
        type=int,
        help='Num of video output frames')
    
    parser.add_argument(
        '--valid_id',
        type=int,
        default=0, 
        help='Valid Id of dataset')
    
    parser.add_argument(
        '--ex_name',
        type=str,
        default='test', 
        help='ex_name')
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0', 
        help='device')
    
    return parser
