import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import shap


def split_data(df: DataFrame, target: str, test_size: float = 0.2, valid_size: float = None, shuffle: bool = True, random_state: int = 1, stratify_on_target: bool = True):
    X = df.copy()
    y = X.pop(target)
    stratify = y if stratify_on_target and shuffle else None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle, stratify=stratify)
    
    if valid_size is not None:
        valid_size = round((X.shape[0] * valid_size) / X_train.shape[0], 2)
        stratify = y_train if stratify_on_target else None
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_size, random_state=random_state, shuffle=shuffle, stratify=stratify)
        
        return X_train, X_test, X_valid, y_train, y_test, y_valid
    
    return X_train, X_test, y_train, y_test


def split_data_on_index(df: DataFrame, target: str, index_level: int = 0):
    """split data based on device ids"""
    index_name = df.index.names[index_level]
    df_index = DataFrame(zip(df.index.get_level_values(index_level), df[target]), columns=[index_name, target])

    X_train, X_test, y_train, y_test = split_data(df_index, target, shuffle=False)
    
    X_train = df[df.index.isin(X_train["device"].values, level=index_level)].copy()
    y_train = X_train.pop(target)

    X_test = df[df.index.isin(X_test["device"].values, level=index_level)].copy()
    y_test = X_test.pop(target)

    return X_train, X_test, y_train, y_test

def _get_shap_values(model, X, labels):
    X = X if labels is None else DataFrame(X, columns=labels)
    
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    
    return X, explainer.shap_values(X), explainer.expected_value

def shap_explain(model, X, labels=None):
    X, shap_values, _ = _get_shap_values(model, X, labels)
    shap.summary_plot(shap_values, X)
    
def shap_explain_prediction(model, X, index, labels=None):
    X, shap_values, expected_value = _get_shap_values(model, X, labels)
    return shap.force_plot(
        expected_value, shap_values[index], X.iloc[index]
    )
