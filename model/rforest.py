import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from bayes_opt import BayesianOptimization
from sklearn.metrics import roc_auc_score, r2_score, mean_absolute_error, mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score

def train(X_train, X_test, y_train, y_test, best_params, feval, model_path, seed=42, verbose_eval=50, bagging_seed=42, verbosity=-1, task='binary_classification'):
    model_path = model_path + "seed-{}.joblib".format(seed)

    if task == 'binary_classification':
        model = RandomForestClassifier(
            n_estimators=int(round(best_params['params']['n_estimators'])),
            max_depth=int(round(best_params['params']['max_depth'])),
            min_samples_split=int(round(best_params['params']['min_samples_split'])),
            min_samples_leaf=int(round(best_params['params']['min_samples_leaf'])),
            max_features=max(min(best_params['params']['max_features'], 1), 0),
            random_state=seed
        )
    else:
        model = RandomForestRegressor(
            n_estimators=int(round(best_params['params']['n_estimators'])),
            max_depth=int(round(best_params['params']['max_depth'])),
            min_samples_split=int(round(best_params['params']['min_samples_split'])),
            min_samples_leaf=int(round(best_params['params']['min_samples_leaf'])),
            max_features=max(min(best_params['params']['max_features'], 1), 0),
            random_state=seed
        )

    model.fit(X_train, y_train)

    # 保存模型
    joblib.dump(model, model_path)

    if task == 'binary_classification':
        y_train_pred = model.predict_proba(X_train)[:, 1]
        y_test_pred = model.predict_proba(X_test)[:, 1]
        roc_auc_train = roc_auc_score(y_train, y_train_pred)
        roc_auc_test = roc_auc_score(y_test, y_test_pred)

        criteria = {
            'roc_auc_train': roc_auc_train,
            'roc_auc_test': roc_auc_test,
        }

    else:
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
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
    model = joblib.load(model_path)

    dtest = X_test

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
            'n_estimators': (50, 200),           # 森林中树的数量
            'max_depth': (3, 10),                # 每棵树的最大深度
            'min_samples_split': (2, 20),        # 分割内部节点所需的最小样本数
            'min_samples_leaf': (1, 10),         # 叶子节点的最小样本数
            'max_features': (0.1, 1.0),          # 寻找最佳分割时考虑的特征比例
        }

    def rf_cv(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):

        if task == 'binary_classification':
            model = RandomForestClassifier(
                n_estimators=int(round(n_estimators)),
                max_depth=int(round(max_depth)),
                min_samples_split=int(round(min_samples_split)),
                min_samples_leaf=int(round(min_samples_leaf)),
                max_features=max(min(max_features, 1), 0),
                random_state=seed
            )
        else:
            model = RandomForestRegressor(
                n_estimators=int(round(n_estimators)),
                max_depth=int(round(max_depth)),
                min_samples_split=int(round(min_samples_split)),
                min_samples_leaf=int(round(min_samples_leaf)),
                max_features=max(min(max_features, 1), 0),
                random_state=seed
            )

        if task == 'binary_classification':
            score_func = make_scorer(roc_auc_score)
        else:
            score_func = make_scorer(r2_score)

        scores = cross_val_score(model, X_train, y_train, cv=5, scoring=score_func)
        return scores.mean()

    optimizer = BayesianOptimization(rf_cv, pds, random_state=seed)
    optimizer.maximize(init_points=init_iter, n_iter=n_iters)

    return optimizer.max


# 帮助函数用于评估指标
def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)


def xgb_mse_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'mse', mean_squared_error(labels, preds)


def xgb_mae_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(labels, preds)


def xgb_roc_auc_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'roc_auc', roc_auc_score(labels, preds)


def metric_cal(y_true, y_pred, precise=3):
    rmse = round(np.sqrt(mean_squared_error(y_true, y_pred)), precise)
    mae = round(mean_absolute_error(y_true, y_pred), precise)
    r2 = round(r2_score(y_true, y_pred), precise)

    return rmse, mae, r2


def feval_value(criterion_col):
    """To determine the score function.

    Args:
        criterion_col (str): criterion_type

    Returns:
        (func, bool): (feval, ascending)
    """
    if criterion_col == "rmse_test_ave":
        ascending = True
        feval = xgb_mse_score
    elif criterion_col == "mae_test_ave":
        ascending = True
        feval = xgb_mae_score
    elif criterion_col == "r2_test_ave":
        ascending = False
        feval = xgb_r2_score
    elif criterion_col == "roc_auc_test_ave":
        ascending = False
        feval = xgb_roc_auc_score

    return feval, ascending