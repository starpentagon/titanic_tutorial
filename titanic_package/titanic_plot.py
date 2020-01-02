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

def plot_survival_rate(col_name, titanic_all, bins=10, label_format='{:02}_{:.0f}-{:.0f}'):
    '''
    特徴量の値ごとの生存率を描画する

    Parameters
    ----------
    col_name : str
        ヒストグラムを描画する列名
    titanic_all : pd.DataFrame
        全データ
    bins : int
        ヒストグラムのbinの数。valuesがstr型の場合は無視される
    '''
    
    # ビン分割
    all_values = titanic_all[col_name]
    all_binned_values = get_binned_data(all_values, bins=bins, label_format=label_format)

    train_flg = titanic_all['Type'] == 'train'
    train_binned_values = all_binned_values[train_flg]

    # カテゴリごとに集計する
    feature_df = pd.DataFrame({col_name : train_binned_values, 'Survived' : titanic_all['Survived']})
    survival_rate_df = feature_df.groupby(col_name).mean()
    count_df = feature_df.groupby(col_name).count()
    count_df.columns = ['count']
    
    category_survival_df = survival_rate_df.join(count_df)

    # ヒストグラムと生存率をplot
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.bar(category_survival_df.index, category_survival_df['count'], alpha=0.5)
    ax1.set_ylabel('count')
    ax1.set_xticklabels(category_survival_df.index, rotation=90)

    ax2 = ax1.twinx()
    ax2.plot(category_survival_df.index, category_survival_df['Survived'], color='red', label='Survival')
    ax2.set_ylabel('Survival rate')
    ax2.set_ylim([0, 1.2])
    ax2.legend(loc='best')

    ax1.set_title('Survival rate by {col}'.format(col=col_name))
    ax1.set_xlabel(col_name)

    print(category_survival_df.to_string(formatters={'Survived': '{:.1%}'.format}))

def plot_model_coefficient(model, feature_names):
    '''
    モデル係数の大小関係を描画する
    
    Parameters
    ----------------
    
    model : modelオブジェクト
    
    feature_names : list
        特徴量名のリスト
    '''
    coef_df = pd.DataFrame(model.coef_.T)
    feature_df = pd.DataFrame(feature_names)

    model_df = pd.concat([feature_df, coef_df, abs(coef_df)], axis=1)
    model_df.columns = ['feature_name', 'coef', 'coef_abs']
    model_df = model_df.sort_values(by='coef_abs')

    plt.xticks(rotation=90)
    plt.bar(model_df.feature_name, model_df.coef)
    
def get_plot_data(feature, X, y_pred, y_train, bins=10):
    '''
    特徴量と生存率（予測／実績）のグラフ描画用のデータセットを生成する

    Parameters
    ----------
    X : pd.DataFrame
        特徴量のデータセット
        
    y_pred : list
        予測値
        
    y_train : list
        実績値

    bins : int
        binの数(default: 10)
    '''
    x_val = X[feature]
    x = get_binned_data(x_val, bins)

    feature_df = pd.DataFrame({feature : x, 'Survived(fact)' : y_train, 'Survived(pred)' : y_pred})
    survival_rate_df = feature_df.groupby(feature).mean()
    count_df = feature_df.groupby(feature).count()[['Survived(fact)']]
    count_df.columns = ['count']

    plot_df = survival_rate_df.join(count_df)
    
    return plot_df

def plot_feature_result(axe, plot_df, loc='upper right'):
    '''
    特徴量ごとの予測／実績の傾向を描画する

    Parameters
    ----------
    axe : subplot
        subplotオブジェクト
        
    plot_df : pd.DataFrame
        グラフ描画用のデータセット
    '''
    x_axis = np.array(plot_df.index)
    feature = plot_df.index.name

    axe.bar(x_axis, plot_df['count'], alpha=0.5)
    axe.set_ylabel('count')
    axe.set_xticklabels(x_axis, rotation=90)

    ax2 = axe.twinx()

    ax2.plot(x_axis, plot_df['Survived(fact)'], color='red', label='Survival(fact)')
    ax2.plot(x_axis, plot_df['Survived(pred)'], color='blue', label='Survival(pred)')

    ax2.set_ylabel('Survival rate')
    ax2.legend(loc=loc)

    axe.set_title('Survival rate by {feature}'.format(feature=feature))
    axe.set_xlabel(feature)    

def plot_single_regression_result(model, feature, X_train, X_orig, y_train):
    '''
    モデルの予測と実績の傾向を描画する

    Parameters
    ----------
    model : 
        学習済みオブジェクト
    
    feature : str
        特徴量名
        
    X_train : pd.DataFrame
        学習データ（正規化等の加工済）

    X_orig : pd.DataFrame
        学習データ（元データ）
        
    y_train : list
        実績値
    '''
    y_pred = [p[1] for p in model.predict_proba(X_train)]

    plot_df = get_plot_data(feature, X_orig, y_pred, y_train)

    fig, axe = plt.subplots(1, 1)
    plot_feature_result(axe, plot_df)
    