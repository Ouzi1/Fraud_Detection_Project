import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
                             precision_score, recall_score, mean_absolute_error,
                             mean_squared_error, root_mean_squared_error, r2_score)

import importlib

# Try importing root_mean_squared_error (available in sklearn >= 1.6)
if importlib.util.find_spec("sklearn.metrics"):
    try:
        from sklearn.metrics import root_mean_squared_error
    except ImportError:
        # Fallback: define manually if sklearn < 1.6
        def root_mean_squared_error(y_true, y_pred):
            return np.sqrt(mean_squared_error(y_true, y_pred))

#%% 1. Def Check inConsistency 
def check_inconsistency(data, Categorical_Variables, Ordinal_Variables, dependant_name = None):
    if dependant_name != None and data[dependant_name].dtype in ['object', 'category']:
        All = list(Categorical_Variables) + list(Ordinal_Variables) + [dependant_name]
    else:
        All = list(Categorical_Variables) + list(Ordinal_Variables) 
    
    for feature in All:
        Item_categories = data[feature].unique()
        print(feature, 'has', len(Item_categories), 'categories. They are: ', Item_categories)
        print('-----------------------------------------------------------------------------------------')
    
    return
    
#%% 2. Define a custom transformer for outlier treatment
class OutlierTrimmer(BaseEstimator, TransformerMixin):
    """
    Capping outliers using IQR method.
    1. Values below Q1 - 1.5*IQR are capped at that limit,
    2. Values above Q3 + 1.5*IQR are capped at that limit.
    """
    def __init__(self):
        self.bounds_ = {}

    def fit(self, X, y = None):
        X_df = pd.DataFrame(X)
        for col in X_df.columns:
            Q1 = X_df[col].quantile(0.25)
            Q3 = X_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            self.bounds_[col] = (lower, upper)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            lower, upper = self.bounds_[col]
            X_df[col] = np.clip(X_df[col], lower, upper)
        return X_df.values
    
    
#%% 3. Evaluate 

def Evaluate(Models_List, X_train, X_test, y_train, y_test, classification = True):
    # ---------------- Classification ----------------
    if classification == True:
        metrics = {"Accuracy": "accuracy",
                   "Balanced_Accuracy": "balanced_accuracy",
                   "F1_Score": "f1",
                   "ROC-AUC": "roc_auc",
                   "Precision": "precision",
                   "Sensitivity": "recall"}
        cv_strategy = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

    # ---------------- Regression ----------------
    else:
        metrics = {"R2_Score": "r2",
                   "MAE": "neg_mean_absolute_error",
                   "MSE": "neg_mean_squared_error",
                   "RMSE": "neg_root_mean_squared_error"}
        cv_strategy = KFold(n_splits = 5, shuffle = True, random_state = 42)

    Testing_Scores = []
    Validation_Scores = []

    for Name, model in Models_List:
        print(f"Evaluating: {Name}")

        # Cross-validation scores
        Evaluation_Metrics = {"Technique": Name}
        for score_name, scorer in metrics.items():
            scores = cross_val_score(model, X_train, y_train, cv = cv_strategy, scoring = scorer)
            mean_score = np.mean(scores)
            # Convert negative regression metrics back to positive
            if 'neg_' in scorer:
                mean_score = -mean_score  # negate back to positive
            Evaluation_Metrics[score_name] = np.round(mean_score, 3)
        Validation_Scores.append(Evaluation_Metrics)

        # Train and test evaluation
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        if classification:
            # Handle models without predict_proba()
            if hasattr(model, "predict_proba"):
                prob_predictions = model.predict_proba(X_test)[:, 1]
                Test_AUC = round(roc_auc_score(y_test, prob_predictions), 3)
            else:
                Test_AUC = np.nan

            Testing_Scores.append({"Technique": Name,
                                   "Accuracy": round(accuracy_score(y_test, predictions), 3),
                                   "Balanced_Accuracy": round(balanced_accuracy_score(y_test, predictions), 3),
                                   "F1_Score": round(f1_score(y_test, predictions), 3),
                                   "AUC": Test_AUC,
                                   "Precision": round(precision_score(y_test, predictions, zero_division=0), 3),
                                   "Sensitivity": round(recall_score(y_test, predictions), 3)})

        else:
            Testing_Scores.append({"Technique": Name,
                                   "R2_Score": round(r2_score(y_test, predictions), 3),
                                   "MAE": round(mean_absolute_error(y_test, predictions), 3),
                                   "MSE": round(mean_squared_error(y_test, predictions), 3),
                                   "RMSE": round(root_mean_squared_error(y_test, predictions), 3)})

    # Convert to DataFrames
    Validation_Scores_df = pd.DataFrame(Validation_Scores)
    Testing_Scores_df = pd.DataFrame(Testing_Scores)

    # Save to Excel
    Validation_Scores_df.to_excel('Validation_Scores_df.xlsx', index = False)
    Testing_Scores_df.to_excel('Testing_Scores_df.xlsx', index = False)

    return Testing_Scores_df, Validation_Scores_df
