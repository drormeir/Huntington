from typing import Tuple
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import sklearn.metrics

def simple_feature_enginearing(df: pd.DataFrame, feature_columns: list[str], y_true_binary,\
                               engineer: bool = False, poly_deg: int = 1, verbose: int = 1) -> Tuple[pd.DataFrame,pd.DataFrame]:
    df = df[feature_columns]
    if engineer:
        temp_df = {}
        # Step 1: Apply Inverse, Log, Square Root, Exponent, and Rank transformations
        if verbose:
            shape = df.shape
            num_positive = np.sum(y_true_binary)
            num_negative = shape[0] - num_positive
            print(f'Feature Engineering: {shape=} {num_positive=} {num_negative=}')
        for col_name in tqdm(feature_columns, disable=verbose<1):
            col = df[col_name].to_numpy()
            min_col = col.min()
            if min_col >= 0:
                # Inverse Transformation
                temp_df[f'inv_{col_name}'] = 1 / (col + 1e-6)  # Avoid division by zero
                # Logarithmic Transformation
                temp_df[f'log_{col_name}'] = np.log(col) if min_col > 1e-6 else np.log1p(col)
                # Square Root Transformation
                temp_df[f'sqrt_{col_name}'] = np.sqrt(col)
            # Exponentiation
            if col.max() < 10:
                temp_df[f'exp_{col_name}'] = np.exp(col)
            # Rank Transformation
            temp_df[f'rank_{col_name}'] = np.argsort(np.argsort(col)) + 1

        if verbose:
            print(f'Feature Engineering: Number of additional engineered features: {len(temp_df)}')

        df = pd.concat([df, pd.DataFrame(temp_df, index=df.index)], axis=1) 
    if verbose:
        print(f'Feature Engineering: Number of total features before polynum features: {df.shape[1]}')
    if poly_deg > 1:    
        # Step 2: Apply Polynomial Features
        poly = PolynomialFeatures(degree=poly_deg, include_bias=False)
        X_poly = poly.fit_transform(df)
        poly_feature_names = poly.get_feature_names_out(df.columns)
        
        # Convert the polynomial features back into a DataFrame
        df = pd.DataFrame(X_poly, columns=poly_feature_names)
        if verbose:
            print(f'Feature Engineering: Number of engineered features after poly(deg=2): {df.shape[1]}')
    
    # Step 3: Calculate PR-AUC for each engineered feature
    pr_auc_scores = []

    for feature in tqdm(list(df.columns), disable=verbose<1):
        # Try to calculate PR-AUC for each feature
        try:
            # Precision-Recall curve
            x_values = df[feature]
            precision_pos, recall_pos, _ = sklearn.metrics.precision_recall_curve(y_true=y_true_binary, y_score=x_values)
            pr_auc_pos = sklearn.metrics.auc(recall_pos, precision_pos)
            # Calculate PR-AUC for the negative of the feature (flipped sign)
            precision_neg, recall_neg, _ = sklearn.metrics.precision_recall_curve(y_true=y_true_binary, y_score=-x_values)
            pr_auc_neg = sklearn.metrics.auc(recall_neg, precision_neg)
        except ValueError:
            # In case there's an error calculating PR-AUC (e.g., constant feature), skip
            continue

        def max_f1_score(precisions,recalls) -> float:
            f1_scores = []
            for p, r in zip(precisions, recalls):
                if p == 0 and r == 0:
                    f1_scores.append(0)  # Set F1 to 0 if both precision and recall are 0
                else:
                    f1 = 2 * (p * r) / (p + r)
                    f1_scores.append(f1)
            return np.max(f1_scores)
        
        f1_score_pos = max_f1_score(precision_pos, recall_pos) 
        f1_score_neg = max_f1_score(precision_neg, recall_neg)
        pr_auc = float(max(pr_auc_pos, pr_auc_neg))
        f1_score = float(max(f1_score_pos,f1_score_neg))
        pr_auc_scores.append((feature, pr_auc, f1_score))


    # Step 4: Sort by PR-AUC
    features_scores = sorted(pr_auc_scores, key=lambda x: x[1], reverse=True)
    features_scores = pd.DataFrame(features_scores, columns=['Features', 'PR-AUC', 'Max F1-score'])
    return features_scores, df


def evaluate_features_by_xgboost(df, feature_columns, y_true_binary,\
                                 verbose: int = 1, test_size: float = 0.2, cv_monte_carlo: int = 5) -> Tuple[list[str], float]:
    initial_features_scores, df = simple_feature_enginearing(df=df, feature_columns=feature_columns,y_true_binary=y_true_binary,\
                                                     engineer=False,poly_deg=1,verbose=max(0,verbose-1))
    initial_features_scores = initial_features_scores.sort_values(by='Max F1-score', ascending=False)
    ordered_features = initial_features_scores['Features'].tolist()
    if verbose:
        print(f'Evaluating features using XGBoost')
    xgb_model = XGBClassifier()
    ordered_features_score = []
    for num_features in tqdm(range(1,len(ordered_features)+1), disable=verbose<1):
        X = df[ordered_features[:num_features]].values
        model_scores = []
        for ind_monte_carlo in range(cv_monte_carlo):
            X_train, X_test, y_train, y_test = train_test_split(X, y_true_binary, test_size=test_size, random_state=ind_monte_carlo)
            # Fit the model
            xgb_model.fit(X_train, y_train)
            # Predict on the test set
            y_pred = xgb_model.predict(X_test)
            # Evaluate the model
            model_score = sklearn.metrics.f1_score(y_test, y_pred)
            model_scores.append(model_score)
        ordered_features_score.append(np.median(model_scores))
    ind_max = np.argmax(ordered_features_score)
    num_features = ind_max + 1
    selected_features = ordered_features[:num_features]
    final_score = ordered_features_score[ind_max]
    if verbose:
        plt.plot(range(1, len(ordered_features_score) + 1), ordered_features_score)
        plt.vlines(num_features, ymin=0.5, ymax=1.0, linestyles='dashed', colors='orange',\
                   label=f'Optimal number of features: {num_features}\nScore: {final_score}')
        plt.xlabel('Number of features')
        plt.ylabel('F1 Score of XGBoots')
        plt.title(f'Median F1 Score based on Cross-Validation of XGBoots as a function of number of features.\n'
                  +f'Validation size: {test_size*100:.2f}%   Monte-Carlo: {cv_monte_carlo}')
        plt.legend()
        plt.show()
    return selected_features, final_score
