#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def get_binned_data(x, bins=10, label_format='{:02}_{:.0f}-{:.0f}'):
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

    label_format : str
        labelの表示形式
    '''
    if len(x) == 0:
        return []
    
    # データ型チェック
    if type(x) not in (pd.Series, pd.DataFrame):
        x = pd.Series(x)
        
    v = x[x.index[0]]

    if not(type(v) in (str, int, float, np.float64, np.int64, np.uint8)):
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
        if type(bins) is int:
            binned_value, bin_def = pd.qcut(x, bins, retbins=True, duplicates='drop')
        else:
            bin_def = bins
        
        labels = [label_format.format(i, bin_def[i], bin_def[i+1]) for i in range(len(bin_def)-1)]

        if type(bins) is int:
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

def get_age_ctgr(age, pclass):
    '''
    年齢のカテゴリ名を取得する。
    * 「5才以下」
    * 「5〜15才(2nd以上)」
    * 「5〜10才（3rd）」
    * 「10〜15才（3rd）」
    * 「15〜60才」
    * 「60才以上」
    * 「欠損値」
    * 「推定値」
    
    Parameters
    -----------
    age : float
        年齢    
    '''
    age_ctgr = ''
    
    if pd.isnull(age):
        age_ctgr = 'nan'
    elif is_age_estimated(age):
        age_ctgr = 'estimated'
    elif age <= 5:
        age_ctgr = '00-05'
    elif age <= 15:
        if pclass <= 2:
            age_ctgr = '05-15(2nd+)'
        elif age <= 10:
            age_ctgr = '05-10(3rd)'
        else:
            age_ctgr = '10-15(3rd)'
    elif age <= 60:
        age_ctgr = '15-60'
    else:
        age_ctgr = '60+'

    return age_ctgr