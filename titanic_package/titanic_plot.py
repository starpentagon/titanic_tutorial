#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys,os
sys.path.append(os.path.dirname(__file__))

from titanic_preprocess import get_binned_data

def plot_train_test_histogram(col_name, titanic_all, bins=10):
    '''
    学習用データと評価用データのヒストグラムを描画する

    Parameters
    ----------
    col_name : str
        ヒストグラムを描画する列名
    titanic_all : pd.DataFrame
        全データ
    bins : int
        ヒストグラムのbinの数
    '''
    
    # ビン分割
    all_values = titanic_all[col_name]
    all_binned_values = get_binned_data(all_values, bins)
    
    train_flg = titanic_all['Type'] == 'train'
    train_binned_values = all_binned_values[train_flg]
    
    test_flg = titanic_all['Type'] == 'test'
    test_binned_values = all_binned_values[test_flg]
    
    # カテゴリごとに件数を集計
    train_plot_data = pd.DataFrame({'train': train_binned_values.value_counts().sort_index() / sum(train_flg)})
    test_plot_data = pd.DataFrame({'test': test_binned_values.value_counts().sort_index() / sum(test_flg)})
    
    all_plot_data = pd.DataFrame({'all': all_binned_values.value_counts().sort_index()})
    
    # 全体カテゴリのindexに合わせる
    train_plot_data = pd.concat([all_plot_data, train_plot_data], axis=1, sort=True).fillna(0)['train']
    test_plot_data = pd.concat([all_plot_data, test_plot_data], axis=1, sort=True).fillna(0)['test']
    
    x = np.arange(len(all_plot_data))
    w = 0.4
    
    plt.bar(x, train_plot_data, width=w, label='train', color='blue')
    plt.bar(x+w, test_plot_data, width=w, label='test', color='red')
    plt.xticks(x+w/2, all_plot_data.index, rotation=90)
    plt.legend(loc='best')
    