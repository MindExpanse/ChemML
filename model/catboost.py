# coding: utf-8
import numpy as np
import catboost as cbt
from bayes_opt import BayesianOptimization
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def train(X_train, X_test, y_train, y_test, best_params, feval, model_path, seed=42, verbose_eval=50, bagging_seed=42, verbosity=-1, task='binary_classification'):

    model_path = model_path + "seed-{}.cbm".format(seed)
    params = {
        'max_depth': int(best_params["params"]["max_depth"]),
        "rsm": best_params["params"]["rsm"],
        "subsample":best_params["params"]["subsample"],
        "eta":best_params["params"]["eta"],
    }


    if task == "binary_classification":
        # 如果是二分类任务
        params["objective"] = "CrossEntropy"
        params["eval_metric"] = "CrossEntropy"

    elif task == "regression":
        # 如果是回归任务
        # params["objective"] = "reg:squarederror"
        # params["eval_metric"] = "rmse"
        params["objective"] = "RMSE"
        params["eval_metric"] = "RMSE"
        print(params)

    # 设置模型的参数
    #params["max_depth"] = int(round(best_params["params"]["max_depth"]))
    #1params['learning_rate'] = best_params["params"]["learning_rate"]
    #1params['colsample_bytree'] = max(min(best_params["params"]["colsample_bytree"], 1), 0)
    #params['rsm'] = max(min(best_params["params"]["rsm"], 1), 0)
    #params['subsample'] = max(min(best_params["params"]["subsample"], 1), 0)
    #params['eta'] = best_params["params"]["eta"]
    #params["random_seed"] = best_params["params"]["random_seed"]
    #params['gamma'] = best_params["params"]["gamma"]

    # 创建训练集
    train_set = cbt.Pool(X_train, label=y_train)

    # 训练模型
    model = cbt.train(
        train_set,
        params,
        num_boost_round=int(best_params["params"]["num_boost_round"]),
        verbose_eval=verbose_eval,
        #feval=feval,
    )

    # 保存训练好的模型
    model.save_model(model_path)

    # 预测
    dtrain = cbt.Pool(X_train)
    dtest = cbt.Pool(X_test)

    if task == "binary_classification":
        # 如果是二分类任务
        y_train_pred = model.predict(dtrain)
        y_test_pred = model.predict(dtest)
        print(y_test_pred)
        roc_auc_train = roc_auc_score(y_train, y_train_pred)
        roc_auc_test = roc_auc_score(y_test, y_test_pred)

        criteria = {
            'roc_auc_train': roc_auc_train,
            'roc_auc_test':  roc_auc_test,
        }

    elif task == "regression":
        # 如果是回归任务
        y_train_pred = model.predict(dtrain)
        y_test_pred = model.predict(dtest)
        rmse_train, mae_train, r2_train = metric_cal(y_train, y_train_pred, precise=4)
        rmse_test, mae_test, r2_test = metric_cal(y_test, y_test_pred, precise=4)

        criteria = {
            'rmse_train': rmse_train,
            'mae_train': mae_train,
            'r2_train': r2_train,
            'rmse_test': rmse_test,
            'mae_test': mae_test,
            'r2_test': r2_test,
        }

    return criteria

def test(X_test, model_path, task='binary_classification'):
    # 加载训练好的模型
    model = cbt.CatBoost()
    model.load_model(model_path)

    dtest = cbt.Pool(X_test)

    if task == "binary_classification":
        y_test_pred = model.predict(dtest)
        # roc_auc_test = roc_auc_score(y_test, y_test_pred)

        # criteria = {
        #     'roc_auc_test': roc_auc_test,
        # }

    elif task == "regression":
        y_test_pred = model.predict(dtest)
        # rmse_test, mae_test, r2_test = metric_cal(y_test, y_test_pred, precise=4)

        # criteria = {
        #     'rmse_test': rmse_test,
        #     'mae_test': mae_test,
        #     'r2_test': r2_test,
        # }

    return y_test_pred

def bayesion_opt(X_train, y_train, init_iter, n_iters, feval, criterion_col, pds='default', random_state=42, seed=42, task='binary_classification', verbosity=-1):

    if pds == 'default':
        pds = {
            'num_boost_round': (200, 2000),#2000
            'max_depth': (3, 10),#10
            #'learning_rate': (0.005, 0.3),
            #'colsample_bytree': (0.5, 1),
            'subsample': (0.6, 1),
            'eta': (0.001, 0.1),#0.1
            #'gamma': (0, 25),
            #添加了这一行
            'rsm': (0.5, 1),  # 1修改这一行
        }
    #评估模型性能
    def cbt_cv(num_boost_round, max_depth, rsm, subsample, eta):#增加了rsm
        #train_set = cbt.DMatrix(data=X_train, label=y_train)
        #catboost中使用Pool加载数据
        train_set = cbt.Pool(data=X_train, label=y_train)

        params = {
            'verbose': 100,
            #'eval_metric': 'Accuracy',
        }

        if task == "binary_classification":
            params["objective"] = "CrossEntropy"
            params["eval_metric"] = "CrossEntropy"
        elif task == "regression":
            params["objective"] = "RMSE"#"regression"
            params["eval_metric"] ="RMSE"# "rmse"

        params["max_depth"] = int(round(max_depth))#指定决策树最大深度
        #params['learning_rate'] = learning_rate#指定学习率，和eta只需要存在一个就可以
        #params['colsample_bytree'] = max(min(colsample_bytree, 1), 0)#把范围控制在0-1
        params['rsm'] = max(min(rsm, 1), 0)  # 修改这一行
        params['subsample'] = max(min(subsample, 1), 0)#控制使用样本比例
        params['eta'] = eta#控制学习率
        #params['gamma'] = gamma#叶子节点分裂所需的最小损失减少值的参数
        params["random_seed"] = seed
        #进行交叉验证，这里面如果提供了pool参数和dtrain参数只需要提供一个就可以了
        cv_results = cbt.cv(
            # pool=train_set,
            # params=params,
            #num_boost_round=int(round(num_boost_round)),
            # nfold=5,
            # stratified=False,#注意设置成False而不是None
            # folds=None,
            # as_pandas=True,
            # seed=seed,
            # shuffle=True,
            pool=train_set,
            params=params,
            iterations=int(round(num_boost_round)),
            fold_count=5,
            stratified=False,
            as_pandas=True,
            #random_seed=seed,
            shuffle=True,
        )
        if criterion_col == "rmse_test_ave":
            result = -np.min(cv_results["test-RMSE-mean"])
        elif criterion_col == "mae_test_ave":
            result = -np.min(cv_results["test-MAE-mean"])
        elif criterion_col == "r2_test_ave":
            result = np.max(cv_results["test-R2-mean"])
        elif criterion_col == "roc_auc_test_ave":
            print(cv_results)
            result = -np.min(cv_results["test-CrossEntropy-mean"])

        return result

    optimizer = BayesianOptimization(cbt_cv, pds, random_state=random_state)
    optimizer.maximize(init_points=init_iter, n_iter=n_iters)


    return optimizer.max

def cbt_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)


def cbt_mse_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'mse', mean_squared_error(labels, preds)


def cbt_mae_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(labels, preds)


def cbt_roc_auc_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'roc_auc', roc_auc_score(labels, preds)


def metric_cal(y_true, y_pred, precise=3):
    # y_true = np.exp(y_true)
    # y_pred = np.exp(y_pred)
    rmse = round(np.sqrt(mean_squared_error(y_true, y_pred)), precise)
    mae = round(mean_absolute_error(y_true, y_pred), precise)
    r2 = round(r2_score(y_true, y_pred), precise)

    return rmse, mae, r2

# 根据评估指标选择相应的评估函数和排序方式
def feval_value(criterion_col):
    """To determine the score function.

    Args:
        criterion_col (str): criterion_type

    Returns:
        (func, bool): (feval, ascending)
    """
    if criterion_col == "rmse_test_ave":
        ascending = True
        feval = cbt_mse_score
    elif criterion_col == "mae_test_ave":
        ascending = True
        feval = cbt_mae_score
    elif criterion_col == "r2_test_ave":
        ascending = False
        feval = cbt_r2_score
    elif criterion_col == "roc_auc_test_ave":
        ascending = False
        feval = cbt_roc_auc_score

    return feval, ascending
