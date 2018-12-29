#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def get_binned_data(x, bins=10):
    '''
    指定したbin数にビン分割したデータを生成する
    * ユニーク数が10未満の場合は値をそのままカテゴリ化する
    * binsにリストを指定した場合はその値で分割する
    * 欠損値は'NaN'に変換する

    Parameters
    ----------
    x : list
        値のリスト(データ型はstr, int, floatを想定)

    bins : int or list
        binの数(default: 10)
    '''
    if len(x) == 0:
        return []
    
    # データ型チェック
    v = x[x.index[0]]

    if not(type(v) is str or type(v) is int or type(v) is float or type(v) is np.float64 or type(v) is np.int64 or type(v) is np.uint8):
        print('Unexpected type: {}'.format(type(v)))
        return x

    # カテゴリデータの場合
    if type(v) is str:
        binned_x = x.fillna('NaN')
        return binned_x

    # unique数が10未満の場合は文字列に変換する
    if len(x.unique()) < 10:
        binned_x = pd.Series([str(val) for val in x])
    else:
        if type(bins) is not list:
            binned_value, bin_def = pd.qcut(x, bins, retbins=True, duplicates='drop')
        else:
            bin_def = bins
        
        labels = ['{:02}_{:.0f}-{:.0f}'.format(i, bin_def[i], bin_def[i+1]) for i in range(len(bin_def)-1)]

        if type(bins) is not list:
            binned_x = pd.qcut(x, bins, labels=labels, duplicates='drop')
        else:
            binned_x = pd.cut(x, bins, labels=labels)

        binned_x = pd.Series([str(val) for val in binned_x])

    return binned_x

def is_age_estimated(age):
    '''
    年齢が推定値（小数部が0.5）かを判定する
    
    Parameters
    -----------
    age : float
        年齢
    '''
    
    if pd.isnull(age):
        return False
    
    fraction = age - int(age)
    
    if abs(2 * fraction - 1.0) < 0.001:
        return True
    else:
        return False