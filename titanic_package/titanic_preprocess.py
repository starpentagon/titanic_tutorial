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

def calc_label_count_rank_encoding(titanic_all, feature, key='PassengerId'):
    '''
    Label Count Rankエンコーディングを行う

    Parameters
    -----------
    titanic_all : pd.DataFrame
        データセット全体
        
    feature : str
        エンコーディングを行う特徴量名（カテゴリ変数）
        
    key : str
        集計する際のキー
    '''
    feature_rank = titanic_all.groupby(feature).count()[key].rank(ascending=False).astype(int)
    feature_dict = feature_rank.to_dict()
    
    return titanic_all[feature].map(lambda x: feature_dict[x])

def get_title(name):
    '''
    敬称(Mr. Mrs. Miss. Master. Dr.　Rev.など)を取得する
    '''
    
    if 'Mr.' in name:
        return 'Mr.'
    
    if 'Mrs.' in name:
        return 'Mrs.'
    
    # Mme.は既婚女性の敬称のため'Mrs.'にする
    if 'Mme.' in name:
        return 'Mrs.'
    
    if 'Miss.' in name:
        return 'Miss.'
    
    if 'Ms.' in name:
        return 'Miss.'
    
    # Mlle.は未婚女性の敬称のため'Miss.'にする
    if 'Mlle.' in name:
        return 'Miss.'
    
    if 'Master.' in name:
        return 'Master.'
    
    # Dr.は件数が少ないのでEliteに集約
    if 'Dr.' in name:
        return 'Elite'
    
    # Rev., Don. Dona. は聖職者の敬称
    if 'Rev.' in name:
        return 'Priest'
    
    if 'Don.' in name:
        return 'Priest'
    
    if 'Dona.' in name:
        return 'Priest'
    
    # Sir., Lady. Countess.　Jonkheer.貴族の敬称
    # -> 件数が少ないのでEliteに集約
    if 'Sir.' in name:
        return 'Elite'
    
    if 'Lady.' in name:
        return 'Elite'
    
    if 'Countess.' in name:
        return 'Elite'
    
    if 'Jonkheer.' in name:
        return 'Elite'
    
    # Col., Capt. は軍の敬称
    # -> 件数が少ないのでEliteに集約
    if 'Col.' in name:
        return 'Elite'
    
    if 'Capt.' in name:
        return 'Elite'
    
    if 'Major.' in name:
        return 'Elite'
    
    return 'None'

def get_family_name(name):
    '''
    名前から苗字を取得する
    '''
    return name.split(',')[0]