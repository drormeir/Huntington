from typing import Tuple
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import PolynomialFeatures
import sklearn.metrics
from sklearn.model_selection import ParameterSampler, train_test_split

def simple_feature_enginearing(df: pd.DataFrame, feature_columns: list[str], y_true_binary,\
                               engineer: bool = False, poly_deg: int = 1, verbose: int = 1) -> Tuple[pd.DataFrame,pd.DataFrame]:
    if verbose < 0:
        verbose = 0    
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


def simple_evaluate_features_by_wells_classifiction_using_xgboost(df, feature_columns, y_true_binary,\
                                 verbose: int = 1, test_size: float = 0.2, cv_monte_carlo: int = 5) -> Tuple[list[str], float]:
    if verbose < 0:
        verbose = 0    
    initial_features_scores, df = simple_feature_enginearing(df=df, feature_columns=feature_columns,y_true_binary=y_true_binary,\
                                                     engineer=False,poly_deg=1,verbose=verbose-1)
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


def complex_evaluate_features_by_wells_classifiction_using_xgboost(\
        df, feature_columns, y_true_binary,\
        verbose: int = 1, test_size: float = 0.2, cv_monte_carlo: int = 5) -> Tuple[list[str], float]:
    if verbose < 0:
        verbose = 0
    df = df[feature_columns]
    if verbose:
        print(f'Evaluating features using XGBoost')
    feature_2_ind_column = {feature:ind for ind,feature in enumerate(feature_columns)}
    num_features = len(feature_columns)
    
    # Define the hyperparameter grid
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 1, 5],
        'min_child_weight': [1, 5, 10],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2]
    }

    random.seed(42)
    np.random.seed(42)
    param_combinations = list(ParameterSampler(param_grid, n_iter=100, random_state=42))
    best_model_score = -np.inf
    best_ind_param = -1
    ordered_features = []
    for ind_params, params in tqdm(list(enumerate(param_combinations)), disable=verbose<1):
        total_features_ranks = np.zeros(shape=num_features, dtype=np.int32)
        feature_in_use = total_features_ranks.copy()
        model_scores = []
        for ind_monte_carlo in range(cv_monte_carlo):
            # Split the dataset using stratified split to maintain class balance
            X_train, X_val, y_train, y_val = train_test_split(df, y_true_binary, test_size=test_size,\
                                                            random_state=ind_monte_carlo, stratify=y_true_binary)
            xgb_model = XGBClassifier(random_state=ind_monte_carlo, **params)
            xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            y_pred = xgb_model.predict(X_val)
            model_score = max(sklearn.metrics.f1_score(y_true=y_val, y_pred=y_pred),
                              sklearn.metrics.f1_score(y_true=~y_val, y_pred=y_pred))
            model_scores.append(model_score)
            # calculates features ranks, works always, even if model do not use all possible features
            features_importance = xgb_model.get_booster().get_score(importance_type='gain')
            features_importance = sorted(features_importance.items(), key=lambda x: x[1], reverse=True)
            for rank, (feature,gain) in enumerate(features_importance):
                ind_column = feature_2_ind_column[feature]
                total_features_ranks[ind_column] += rank
                feature_in_use[ind_column] += 1
        model_score = np.median(model_scores)
        if model_score > best_model_score:
            best_ind_param = ind_params
            best_model_score = model_score
            ordered_features = [feature_columns[ind_feature] for ind_feature in np.argsort(total_features_ranks) if feature_in_use[ind_feature] == cv_monte_carlo]
    if verbose:
        print(f'The best XGBoost model has parameters: {param_combinations[best_ind_param]}')
        print(f'Model F1 score: {best_model_score}')
        print('Ordered features in descending order:')
        for i_feature, feature in enumerate(ordered_features):
            print(f'Feature[{i_feature+1}] = {feature}')
    return ordered_features, best_model_score

def complex_evaluate_features_by_patients_classifiction_using_xgboost(\
        df, feature_columns, y_true_binary, patients_ids,\
        num_combinations: int = 10, verbose: int = 1) -> Tuple[list[str], float]:
    if verbose < 0:
        verbose = 0
    df = df[feature_columns]
    feature_2_ind_column = {feature:ind for ind,feature in enumerate(feature_columns)}
    num_features = len(feature_columns)
    
    # Define the hyperparameter grid
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 1, 5],
        'min_child_weight': [1, 5, 10],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2]
    }
    random_state = 42
    random.seed(random_state)
    np.random.seed(random_state)
    param_combinations = list(ParameterSampler(param_grid, n_iter=num_combinations, random_state=random_state))
    if isinstance(patients_ids, pd.Series):
        patients_ids = patients_ids.to_numpy()
    unique_patient_ids = np.unique(patients_ids)
    num_patient_ids = unique_patient_ids.size
    xgb_random_state = 0
    total_features_ranks = np.zeros(shape=num_features, dtype=np.int32)
    feature_in_use = total_features_ranks.copy()
    test_scores = []
    with tqdm(total=num_patient_ids * (num_combinations*(num_patient_ids-1)+1), disable=verbose<1,\
              desc='Evaluating features using XGBoost') as pbar:
        for patient_id_test in unique_patient_ids:
            is_testing_rows = patients_ids == patient_id_test
            is_non_testing_rows = ~is_testing_rows
            best_validation_params = None
            best_validation_score = -np.inf
            for params in param_combinations:
                validation_scores = []
                for patient_id_validation in unique_patient_ids:
                    if patient_id_validation == patient_id_test:
                        continue
                    is_training_rows = is_non_testing_rows & (patients_ids != patient_id_validation)
                    is_validation_rows = is_non_testing_rows & (patients_ids == patient_id_validation)
                    X_train = df.iloc[is_training_rows]
                    y_train = y_true_binary[is_training_rows]
                    X_val = df.iloc[is_validation_rows]
                    y_val = y_true_binary[is_validation_rows]
                    xgb_model = XGBClassifier(random_state=xgb_random_state, **params)
                    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)        
                    y_pred = xgb_model.predict(X_val)
                    validation_score = sum(y_pred == y_val)/y_val.size*100 # accuracy for single 
                    validation_scores.append(validation_score)
                    pbar.update(1)
                validation_score = np.median(validation_scores)
                if validation_score > best_validation_score:
                    best_validation_score = validation_score
                    best_validation_params = params
                pass # next params to check
            # at this point we have the best params for the validation
            # we will execute a final model to predict the test set
            X_train = df.iloc[is_non_testing_rows]
            y_train = y_true_binary[is_non_testing_rows]
            X_test = df.iloc[is_testing_rows]
            y_test = y_true_binary[is_testing_rows]
            xgb_model = XGBClassifier(random_state=xgb_random_state, **best_validation_params)
            xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            y_pred = xgb_model.predict(X_test)
            test_score = sum(y_pred == y_test)/y_test.size*100 # accuracy for single 
            test_scores.append(test_score)
            # calculates features ranks, works always, even if model do not use all possible features
            features_importance = xgb_model.get_booster().get_score(importance_type='gain')
            features_importance = sorted(features_importance.items(), key=lambda x: x[1], reverse=True)
            for rank, (feature,gain) in enumerate(features_importance):
                ind_column = feature_2_ind_column[feature]
                total_features_ranks[ind_column] += rank
                feature_in_use[ind_column] += 1
            pbar.update(1)
    total_features_ranks += num_features*(num_patient_ids-feature_in_use)
    ordered_features = [feature_columns[ind_feature] for ind_feature in np.argsort(total_features_ranks)]
    test_score = np.median(test_scores)
    return ordered_features, test_score
