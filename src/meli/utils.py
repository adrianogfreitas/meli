# import os

# import joblib
import numpy as np
from pandas import DataFrame
# import shap
# import yaml
# # from pdpbox import pdp
# from sklearn.feature_selection import mutual_info_regression
# from sklearn.metrics import classification_report
# from sklearn.model_selection import cross_validate, train_test_split
from sklearn.model_selection import train_test_split
# from textwrap import dedent

# from meli.plot import make_confusion_matrix


# def get_conf(conf_file_path="../config.yaml"):
#     conf = yaml.safe_load(open(conf_file_path, "r"))
#     return conf


# def make_mi_scores(X, y, discrete_features):
#     mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
#     mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
#     mi_scores = mi_scores.sort_values(ascending=False)
#     return mi_scores


# def plot_mi_scores(X, y, discrete_features):
#     scores = make_mi_scores(X, y, discrete_features)
#     scores = scores.sort_values(ascending=True)
#     width = np.arange(len(scores))
#     ticks = list(scores.index)
#     plt.barh(width, scores)
#     plt.yticks(width, ticks)
#     plt.title("Mutual Information Scores")
#     plt.show()


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


# def evaluate_model(y_train, y_pred_train, y_test, y_pred_test):
#     print(classification_report(y_train, y_pred_train))
#     print(classification_report(y_test, y_pred_test))




# def score_model(model, X: DataFrame, y: DataFrame, cv: int = 5, scoring: str = 'roc_auc'):
#     scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
#     best_estimator = np.argmax(scores['test_score'])
    
#     print(dedent(f"""
#     Mean Score: {scores['test_score'].mean():.4}
#     High Score: {max(scores['test_score']):.4}
#     Low Score: {min(scores['test_score']):.4}"""))

#     return best_estimator


# def encode(df, cols, encode_dict_path=None):
#     encode_dict = {}
#     for col in cols:
#         df[col], encode_dict[col] = df[col].factorize()

#     if encode_dict_path is None:
#         conf = get_conf()
#         encode_dict_path = os.path.join(
#             conf["artifacts"]["path"], conf["artifacts"]["encode"]
#         )

#     joblib.dump(encode_dict, encode_dict_path)


# def get_encode_dict(encode_dict_path=None):
#     if encode_dict_path is None:
#         conf = get_conf()
#         encode_dict_path = os.path.join(
#             conf["artifacts"]["path"], conf["artifacts"]["encode"]
#         )

#     encode_dict = joblib.load(encode_dict_path)
#     return encode_dict


# def to_numeric(df, cols):
#     for col in cols:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
#     return df

# def split_cols(df, target):
#     categorical_cols = [col for col in df.select_dtypes('object') if col != target]
#     dummy_cols = [col for col in df.select_dtypes('int64') if col != target]
#     numerical_cols = [col for col in df.select_dtypes('float64') if col != target]
#     return categorical_cols, dummy_cols, numerical_cols


# class Explainer:
#     def __init__(
#         self,
#         model,
#         X,
#         labels=None,
#         model_type="classification",
#         shap_method="TreeExplainer",
#     ):
#         self._model = model
#         self._shap_method = shap_method
#         self._model_type = model_type
#         self._X = X if labels is None else DataFrame(X, columns=labels)
#         self._shap_values = None
#         self._explainer = None

#         shap.initjs()
#         self._build_shap_values()

#     def _predict(self, data_asarray):
#         data_asframe = DataFrame(data_asarray, columns=self._X.columns)
#         return self._model.predict(data_asframe)

#     def _build_shap_values(self):
#         method = getattr(shap, self._shap_method)
#         self._explainer = method(self._model)
#         self._shap_values = self._explainer.shap_values(self._X)

#     @property
#     def shap_values(self):
#         # return self._shap_values[1] if self._model_type == 'classification' else self._shap_values
#         return (
#             self._shap_values
#         )  # [1] if self._model_type == 'classification' else self._shap_values

#     @property
#     def expected_value(self):
#         # return self._explainer.expected_value[1] if self._model_type == 'classification' else self._explainer.expected_value
#         return (
#             self._explainer.expected_value
#         )  # [1] if self._model_type == 'classification' else self._explainer.expected_value

#     def explain(self):
#         shap.summary_plot(self.shap_values, self._X)
#         # shap.summary_plot(self.shap_values, self._X, plot_type="bar")

#     # def explain_feature(self, feature, feature_name=None):
#     #     feature_names = list(self._X.columns)
#     #     pdp_feature = pdp.pdp_isolate(
#     #         model=self._model,
#     #         dataset=self._X,
#     #         model_features=feature_names,
#     #         feature=feature,
#     #     )
#     #     plot = pdp.pdp_plot(pdp_feature, feature if not feature_name else feature_name)
#     #     plot[0].set_size_inches(8, 5)
#     #     return plot

#     def explain_features(self, features):
#         return shap.dependence_plot(
#             features[0], self.shap_values, self._X, interaction_index=features[1]
#         )

#     def explain_prediction(self, index):
#         return shap.force_plot(
#             self.expected_value, self.shap_values[index], self._X.iloc[index]
#         )
