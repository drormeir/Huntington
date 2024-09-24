import os
import pandas as pd
from python.csv_pd import pd_display, csv_raw_data, csv_header_body_2_dataframe
from python.Patients_Catalog import Patients_Catalog
from python.feature_engineering import simple_feature_enginearing, complex_evaluate_features_by_wells_classifiction_using_xgboost
from python.feature_engineering import complex_evaluate_features_by_patients_classifiction_using_xgboost

class Live_Cell_Imaging:
    def __init__(self, file_name: str|None = None, verbose: int = 0, class_print_read_data: bool = False) -> None:
        if file_name is None:
            file_name = os.path.join(os.getcwd(),'datasets','Live_Cell_Imaging.csv')
        if verbose or class_print_read_data:
            print(f'Reading Live_Cell_Imaging from {file_name=}')
        csv_data = csv_raw_data(file_name, verbose=max(0,verbose-2))
        df = csv_header_body_2_dataframe(csv_data[0], csv_data[1:], verbose=max(0,verbose-2), file_name=file_name)
        df = df.drop(columns=['Cell_Type', 'Severity', 'PC', 'Cell_Type_PC', 'Severity_PC', 'Experiment', 'plate',\
                              'well'], errors='ignore')
        patient_ID = df.pop('ID')
        self.features_columns = list(df.columns)
        patients_catalog = Patients_Catalog(verbose=max(0,verbose-2))
        patient_ID = patients_catalog.find_ID(patient_ID, verbose=max(0,verbose-1))
        patient_ID = pd.DataFrame({'Patient_ID':patient_ID})
        self.df = pd.concat([df.reset_index(drop=True), patient_ID.reset_index(drop=True)], axis=1)
        self.patients_wells_counts = self.df.groupby('Patient_ID').size()
        self.patients_wells_counts.name = 'Num_Wells'
        if verbose:
            print('\nLive_Cell_Imaging Wells Count\n' + '='*50)
            pd_display(self.patients_wells_counts)
            pd_display(self.df.head())
            print(f'Features columns:\n{self.features_columns}')
    
    def simple_feature_enginearing_classify_HD_without_HGPS(self, verbose: int = 2) -> None:
        df = self.df
        df_no_HGPS = df[Patients_Catalog().is_not_HGPS(df['Patient_ID'])]
        is_healthy = Patients_Catalog().is_healthy(df_no_HGPS.pop('Patient_ID'))
        self.engineered_features_classify_HD_without_HGPS, _ =\
            simple_feature_enginearing(df_no_HGPS, self.features_columns, is_healthy,\
                                       verbose=max(0,verbose-1), engineer=True, poly_deg=2)
        if verbose:
            pd_display(self.engineered_features_classify_HD_without_HGPS.head(10))
    
    def evaluate_features_by_xgboost_classify_HD_without_HGPS(self, cv_monte_carlo: int = 5, verbose: int = 1) -> None:
        df = self.df
        df_no_HGPS = df[Patients_Catalog().is_not_HGPS(df['Patient_ID'])]
        is_healthy = Patients_Catalog().is_healthy(df_no_HGPS.pop('Patient_ID'))
        selected_features, score = complex_evaluate_features_by_wells_classifiction_using_xgboost(\
            df=df_no_HGPS, feature_columns=self.features_columns, y_true_binary=is_healthy,\
            cv_monte_carlo=cv_monte_carlo, verbose=max(0,verbose-1))
        self.selected_features_xgboost_classify_HD_without_HGPS = selected_features
        self.score_xgboost_classify_HD_without_HGPS = score
        if verbose:
            print(f'Selected model has {len(selected_features)} features with final score: {score}')
            for ind, feature_name in enumerate(selected_features):
                print(f'Feature[{ind+1:2d}] = {feature_name}')

    def evaluate_features_by_xgboost_to_classify_patients_HD_without_HGPS(self, verbose: int = 1) -> None:
        df = self.df
        df_no_HGPS = df[Patients_Catalog().is_not_HGPS(df['Patient_ID'])]
        patients_ids = df_no_HGPS.pop('Patient_ID')
        is_healthy = Patients_Catalog().is_healthy(patients_ids)
        selected_features, score = complex_evaluate_features_by_patients_classifiction_using_xgboost(\
            df=df_no_HGPS, feature_columns=self.features_columns, y_true_binary=is_healthy,\
            patients_ids=patients_ids, verbose=max(0,verbose-1))
        self.selected_features_xgboost_classify_HD_without_HGPS = selected_features
        self.score_xgboost_classify_HD_without_HGPS = score
        if verbose:
            print(f'Selected model has {len(selected_features)} features with final accuracy: {score:6.2f}%')
            for ind, feature_name in enumerate(selected_features):
                print(f'Feature[{ind+1:2d}] = {feature_name}')
