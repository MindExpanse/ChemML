# Import necessary packages
# ------------
# -------------
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import smogn
import seaborn as sns

from model import catboost, xgboost, lightgbm, rforest

def check_and_create_folder(folder_path):
    # 使用os.path.exists()函数判断文件夹是否存在
    if not os.path.exists(folder_path):
        try:
            # 使用os.makedirs()函数创建文件夹，包括创建多级目录
            os.makedirs(folder_path)
            print(f"文件夹 '{folder_path}' 创建成功！")
        except OSError as e:
            print(f"创建文件夹 '{folder_path}' 失败：{e}")
    else:
        print(f"文件夹 '{folder_path}' 已经存在！")

import importlib
importlib.reload(catboost)
importlib.reload(xgboost)
importlib.reload(lightgbm)
importlib.reload(rforest)

pca_decomposition = False
log_preprocess = False
train_mode = False # True/False
smogn_mode = False
pre_smogn = False

# Parameters
# ----------
# Data folder
# 读取数据
data = pd.read_excel('dataset.xlsx', sheet_name=0)[['compound', 'Ueff/K', 'θ1/°', 'd1/Å', 'θ2/°', 'd2/Å']]
model_path_dir = "output/416/xgboost/"
output_file = os.path.join(model_path_dir, "bayes_opt.csv")
error_analysis_file = os.path.join(model_path_dir, "error_analysis.csv")

check_and_create_folder(model_path_dir)

# test
# seed = 100
# seed = 200+
# seed = 400
seed = 400
# seed = 200
# seed = 200
# seed = 200
# seed = 200
# model_path = f"output/seed-{seed}.cbm"
model_path = os.path.join(model_path_dir, f'seed-{seed}.bin')
# model_path = os.path.join(model_path_dir, f'seed-{seed}.cbm')
# model_path = os.path.join(model_path_dir, f'seed-{seed}.txt')
# model_path = os.path.join(model_path_dir, f'seed-{seed}.joblib')

# 删除自变量或因变量为空的样本
data.dropna(subset=['Ueff/K', 'θ1/°', 'd1/Å', 'θ2/°', 'd2/Å'], inplace=True)

# 提取自变量和因变量
X = data[['θ1/°', 'd1/Å', 'θ2/°', 'd2/Å']].values
# print("取对数前：")
y = data['Ueff/K'].values
compound = data['compound'].values
if log_preprocess:
    print("取对数后：")
    y = np.log(y)

print("总样本维度：", np.shape(X))

if pca_decomposition:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=100)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_)
    X = pca.transform(X)

    print("PCA降维之后的样本维度：", np.shape(X))

# In this case, rmse serves as the criteria: roc_auc_test_ave/rmse_test_ave
criterion_col = "rmse_test_ave"

# Performance sort order, smaller is better for rmse, so ascending=True
ascending = True

# Random seed
rnds = range(0, 500, 100)

# Model random seed
model_rnd = 42

# X_train, X_test, y_train, y_test, compound_train, compound_test = train_test_split(X, y, compound, test_size=0.3, train_size=0.7, random_state=42)
# X_train, X_test, y_train, y_test, compound_train, compound_test = train_test_split(X, y, compound, test_size=0.3, train_size=0.7, random_state=42)

# Task type: binary_classification/regression
task_type = "regression"

# Bayes Optimization
# Initial iteration, more is better
init_iter = 3

# Optimization times, more is better
n_iters = 5

if pre_smogn:
    print('++++++++++++++++++SMOGN starting++++++++++++++++++')
    train_data = np.concatenate((X, y.reshape(-1, 1)), axis=-1)
    # reset_index
    train_pd = pd.DataFrame(train_data, columns=['θ1/°', 'd1/Å', 'θ2/°', 'd2/Å', 'Ueff/K']).reset_index(drop=True)

    n_tries = 0
    done = False
    while not done:
        try:
            # smogn.smoter(data=foo, y="bar")
            train_smogn = smogn.smoter(
                ## main arguments
                data=train_pd,  ## pandas dataframe
                y='Ueff/K',  ## string ('header name')
                k=9,  ## positive integer (k < n)
                samp_method='extreme',  ## string ('balance' or 'extreme')

                ## phi relevance arguments
                rel_thres=0.80,  ## positive real number (0 < R < 1)
                rel_method='auto',  ## string ('auto' or 'manual')
                rel_xtrm_type='high',  ## string ('low' or 'both' or 'high')
                rel_coef=2.25  ## positive real number (0 < R)
            )
            train_smogn = train_smogn.dropna()
            done = True

        except ValueError:
            if n_tries < 10:
                n_tries += 1
                X_train, X_test, y_train, y_test, compound_train, compound_test = train_test_split(X, y,
                                                                                                   compound,
                                                                                                   test_size=0.3,
                                                                                                   train_size=0.7,
                                                                                                   random_state=seed + n_tries)
            else:
                raise

    # 假设你的DataFrame叫做 train_smogn
    train_smogn.to_csv(os.path.join(model_path_dir, 'synt.csv'), index=False)
    X = train_smogn[['θ1/°', 'd1/Å', 'θ2/°', 'd2/Å']].values
    y = train_smogn['Ueff/K'].values
    compound = np.concatenate([compound, np.ones([603 - 449])], axis=0)



if train_mode:
# Start hyper opt:
#------------------------------------------
    mae_test_list, rmse_test_list, r2_test_list, roc_auc_test_list = [], [], [], []
    result = []
    for seed in rnds:

        # target_model = catboost
        target_model = xgboost
        # target_model = lightgbm
        # target_model = rforest

        # score function
        feval, ascending = target_model.feval_value(criterion_col)

        n_tries = 0

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=seed)
        X_train, X_test, y_train, y_test, compound_train, compound_test = train_test_split(X, y, compound,
                                                                                           test_size=0.3,
                                                                                           train_size=0.7,
                                                                                           random_state=seed+n_tries)
        if smogn_mode:
            print('++++++++++++++++++SMOGN starting++++++++++++++++++')
            train_data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=-1)
            # reset_index
            train_pd = pd.DataFrame(train_data, columns=['θ1/°', 'd1/Å', 'θ2/°', 'd2/Å', 'Ueff/K']).reset_index(drop=True)

            n_tries = 0
            done = False
            while not done:
                try:
                    # smogn.smoter(data=foo, y="bar")
                    train_smogn = smogn.smoter(
                        ## main arguments
                        data=train_pd,  ## pandas dataframe
                        y='Ueff/K',  ## string ('header name')
                        k=9,  ## positive integer (k < n)
                        samp_method='extreme',  ## string ('balance' or 'extreme')

                        ## phi relevance arguments
                        rel_thres=0.80,  ## positive real number (0 < R < 1)
                        rel_method='auto',  ## string ('auto' or 'manual')
                        rel_xtrm_type='high',  ## string ('low' or 'both' or 'high')
                        rel_coef=2.25  ## positive real number (0 < R)
                    )
                    train_smogn = train_smogn.dropna()
                    done = True

                except ValueError:
                    if n_tries < 10:
                        n_tries += 1
                        X_train, X_test, y_train, y_test, compound_train, compound_test = train_test_split(X, y,
                                                                                                           compound,
                                                                                                           test_size=0.3,
                                                                                                           train_size=0.7,
                                                                                                           random_state= seed + n_tries)
                    else:
                        raise

            X_train = train_smogn[['θ1/°', 'd1/Å', 'θ2/°', 'd2/Å']].values
            y_train = train_smogn['Ueff/K'].values

        print(f"X_train.shape: {X_train.shape}, X_test.shape: {X_test.shape}, y_train.shape: {y_train.shape}, y_test.shape: {y_test.shape}")

        # Optimization

        best_params = target_model.bayesion_opt(X_train, y_train, init_iter, n_iters, feval, criterion_col, pds='default', random_state=model_rnd, seed=seed, task=task_type)
        print(f"Best_params of LightBGM by Bayes Optimization: {best_params}")
        # Model training
        criteria = target_model.train(X_train, X_test, y_train, y_test, best_params, feval, model_path_dir, seed=seed, task=task_type)
        rmse_test = criteria['rmse_test']
        mae_test = criteria['mae_test']
        r2_test = criteria['r2_test']

        mae_test_list.append(mae_test)
        rmse_test_list.append(rmse_test)
        r2_test_list.append(r2_test)

        temp_result = [seed, mae_test, rmse_test, r2_test, best_params]
        result.append(temp_result)

    # Summary
    mae_test_ave = np.average(mae_test_list)
    rmse_test_ave = np.average(rmse_test_list)
    r2_test_ave = np.average(r2_test_list)
    mae_test_std = np.std(mae_test_list)
    rmse_test_std = np.std(rmse_test_list)
    r2_test_std = np.std(r2_test_list)

    result.append(["ave", mae_test_ave, rmse_test_ave, r2_test_ave, "--"])
    result.append(["std", mae_test_std, rmse_test_std, r2_test_std, "--"])

    # Save results
    # --------------
    df = pd.DataFrame(result, columns=['entry', 'mae_test_ave', 'rmse_test_ave', 'r2_test_ave', 'best-params'])
    # 使用os.path.dirname()函数获取文件所在目录
    directory = os.path.dirname(output_file)
    print(directory)
    check_and_create_folder(directory)

    df.to_csv(output_file)

else:
    # target_model = catboost
    target_model = xgboost
    # target_model = lightgbm
    # target_model = rforest

    # score function
    feval, ascending = target_model.feval_value(criterion_col)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=seed)
    X_train, X_test, y_train, y_test, compound_train, compound_test = train_test_split(X, y, compound,
                                                                                       test_size=0.3,
                                                                                       train_size=0.7,
                                                                                       random_state=seed)
    if smogn_mode:
        print('++++++++++++++++++SMOGN starting++++++++++++++++++')
        train_data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=-1)
        train_data_1 = np.concatenate((compound_train.reshape(-1, 1), X_train, y_train.reshape(-1, 1)), axis=-1)
        # reset_index
        train_pd = pd.DataFrame(train_data, columns=['θ1/°', 'd1/Å', 'θ2/°', 'd2/Å', 'Ueff/K']).reset_index(drop=True)
        train_pd_1 = pd.DataFrame(train_data_1, columns=['Compound', 'θ1/°', 'd1/Å', 'θ2/°', 'd2/Å', 'Ueff/K']).reset_index(drop=True)
        train_pd_1.to_csv(os.path.join(model_path_dir, 'origin.csv'), index=False)

        # sns.histplot(train_pd['Ueff/K'], bins=50)
        # plt.xlabel('Ueff/K')
        # plt.ylabel('Frequency')
        # plt.title('Histogram of Ueff/K for original')
        # plt.show()

        n_tries = 0
        done = False
        while not done:
            try:
                # smogn.smoter(data=foo, y="bar")
                train_smogn = smogn.smoter(
                    ## main arguments
                    data=train_pd,  ## pandas dataframe
                    y='Ueff/K',  ## string ('header name')
                    k=9,  ## positive integer (k < n)
                    samp_method='extreme',  ## string ('balance' or 'extreme')

                    ## phi relevance arguments
                    rel_thres=0.80,  ## positive real number (0 < R < 1)
                    rel_method='auto',  ## string ('auto' or 'manual')
                    rel_xtrm_type='high',  ## string ('low' or 'both' or 'high')
                    rel_coef=2.25  ## positive real number (0 < R)
                )
                done = True
                train_smogn = train_smogn.dropna()

            except ValueError:
                if n_tries < 10:
                    n_tries += 1
                    X_train, X_test, y_train, y_test, compound_train, compound_test = train_test_split(X, y,
                                                                                                       compound,
                                                                                                       test_size=0.3,
                                                                                                       train_size=0.7,
                                                                                                       random_state=seed + n_tries)
                else:
                    raise

        # 绘制单变量直方图
        # sns.histplot(train_pd['Ueff/K'], bins=50)
        # plt.xlabel('Ueff_log/K')
        # plt.ylabel('Frequency')
        # plt.title('Histogram of Ueff/K for modified')
        # plt.show()

        train_smogn.to_csv(os.path.join(model_path_dir, 'smogn.csv'), index=False)

        X_train = train_smogn[['θ1/°', 'd1/Å', 'θ2/°', 'd2/Å']].values
        y_train = train_smogn['Ueff/K'].values

    # structure_list, Ueff_list = [], []
    result = []
    print(f"X_train.shape: {X_train.shape}, X_test.shape: {X_test.shape}, y_train.shape: {y_train.shape}, y_test.shape: {y_test.shape}")
    y_test_predict = target_model.test(X_test, model_path, task=task_type)
    # print(y_test_predict)
    if log_preprocess:
        y_test_predict = np.exp(y_test_predict)
        y_test = np.exp(y_test)
    print(f"y_test_predict: {y_test_predict.shape}")
    rmse_test, mae_test, r2_test = target_model.metric_cal(y_test, y_test_predict, precise=4)
    print(rmse_test, mae_test, r2_test)
    # 创建散点图
    plt.scatter(y_test, y_test_predict, color='blue', label='Predicted vs True', s=10, alpha=0.5)
    # 添加标签和标题
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    # plt.title('Scatter Plot of True vs Predicted Values for RandomForest')
    plt.title('Scatter Plot of True vs Predicted Values for Xgboost')
    # plt.title('Scatter Plot of True vs Predicted Values for Catboost')
    # plt.title('Scatter Plot of True vs Predicted Values for LightGBM')
    # 添加一条表示理想情况的对角线
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2,
             label='Ideal')
    # 添加图例
    plt.legend()
    # 在图片右下角添加文本标签 "R2=0.924"
    plt.text(0.8, 0.05, r'$R^2={}$'.format(r2_test), color='green', fontsize=12, ha='right', va='bottom', transform=plt.gca().transAxes)
    # 显示图形
    plt.savefig(os.path.join(model_path_dir, 'scatter.png'), dpi=900)
    plt.show()

    # 计算误差绝对值
    error_abs = np.abs(y_test - y_test_predict)
    # 创建一个DataFrame保存数据
    data = {'Compound': compound_test, 'Real Value': y_test, 'Predicted Value': y_test_predict,
            'Error Absolute': error_abs}
    df = pd.DataFrame(data)
    # 按照误差绝对值从大到小排序
    df_sorted = df.sort_values(by='Error Absolute', ascending=False)
    # 将数据保存为CSV文件
    df_sorted.to_csv(error_analysis_file, index=False)
