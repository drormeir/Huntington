import warnings
import os
import pandas as pd
from python.csv_pd import pd_display, csv_raw_data, csv_header_body_2_dataframe
from python.patients_wells import Patient_Wells_Collecter
from python.Patients_Catalog import Patients_Catalog
from python.feature_engineering import complex_evaluate_features_by_patients_classifiction_using_xgboost


class Protein_Data:
    def __init__(self, file_name, verbose: int = 0, class_print_read_data: bool = False) -> None:
        if verbose < 0:
            verbose = 0
        self.name, _ = os.path.splitext(os.path.basename(file_name))
        if verbose or class_print_read_data:
            print(f'Reading Mitochondrial protein data: {self.name}', flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            csv_data = csv_raw_data(file_name, verbose=verbose-1)
            df = csv_header_body_2_dataframe(csv_data[0], csv_data[1:], verbose=verbose-1)
        df = df.drop(columns=['group', 'group_with_id', 'group_with_p', 'group_p', 'Local_Outlier_Factor_30_outliers',\
                      'Local_Outlier_Factor_8_outliers'], errors='ignore')
        patient_ID = df.pop('Cell_id')
        self.features_columns = list(df.columns)
        patient_ID = Patients_Catalog(verbose=verbose-3).find_ID(patient_ID, verbose=verbose-2)
        patient_ID = pd.DataFrame({'Patient_ID':patient_ID})
        self.df = pd.concat([df.reset_index(drop=True), patient_ID.reset_index(drop=True)], axis=1)
        self.patients_wells_counts = self.df.groupby('Patient_ID').size()
        self.patients_wells_counts.name = 'Num_Wells'
        if verbose:
            print(f'DataFrame of {self.name} has shape: {self.df.shape}')
            pd_display(self.df.head())
            pd_display(self.patients_wells_counts)

    def evaluate_features_by_xgboost_to_classify_patients_HD_without_HGPS(self, verbose: int = 2) -> None:
        patients_catalog = Patients_Catalog()
        df = self.df
        df_no_HGPS = df[patients_catalog.is_not_HGPS(df['Patient_ID'])]
        patients_ids = df_no_HGPS.pop('Patient_ID')
        is_healthy = patients_catalog.is_healthy(patients_ids)
        selected_features, score = complex_evaluate_features_by_patients_classifiction_using_xgboost(\
            df=df_no_HGPS, feature_columns=self.features_columns, y_true_binary=is_healthy,\
            patients_ids=patients_ids, verbose=verbose-1)
        self.selected_features_xgboost_classify_HD_without_HGPS = selected_features
        self.score_xgboost_classify_HD_without_HGPS = score
        if verbose:
            print(f'Selected model has {len(selected_features)} features with final accuracy: {score:6.2f}%')
            for ind, feature_name in enumerate(selected_features):
                print(f'Feature[{ind+1:2d}] = {feature_name}')


class Morphology_Data:
    def __init__(self, file_name, verbose: int = 0, class_print_read_data: bool = False) -> None:
        if verbose < 0:
            verbose = 0
        if verbose or class_print_read_data:
            print(f'Reading Mitochondrial Morphology data: {file_name}', flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            csv_data = csv_raw_data(file_name, verbose=verbose-2)
            self.df = csv_header_body_2_dataframe(csv_data[0], csv_data[1:], verbose=verbose-2)
            patients_id = Morphology_Data.group_with_id_2_patient_id(self.df['group_with_id'], verbose=verbose-1)
        self.df['Patient_ID'] = patients_id
        self.patients_wells_counts = self.df.groupby('Patient_ID').size()
        self.patients_wells_counts.name = 'Num_Wells'
        if verbose:
            pd_display(self.patients_wells_counts)
    
    @staticmethod
    def group_with_id_2_patient_id(column_group_with_id, verbose: int = 0) -> list[str]:
        column_group_with_id = list(column_group_with_id)
        patients_catalog = Patients_Catalog(verbose=verbose-1)
        raw_data_2_raw_patient_id = {}
        raw_patient_id_2_catalog_id = {}
        raw_patients_id = []
        for group_with_id in set(column_group_with_id):
            low_str = group_with_id.lower().replace('hc','').replace('mild','').replace('severe','').replace('premanifest','')
            assert isinstance(group_with_id, str)
            raw_patient_id = group_with_id[-len(low_str):]
            assert isinstance(raw_patient_id, str)
            raw_data_2_raw_patient_id[group_with_id] = raw_patient_id
            id_2_check = [raw_patient_id, raw_patient_id.replace('0663', '0633')]
            catalog_id = patients_catalog.find_typo_ID(id_2_check)
            raw_patient_id_2_catalog_id[raw_patient_id] = catalog_id
        patients_ids = []
        for group_with_id in column_group_with_id:
            raw_patient_id = raw_data_2_raw_patient_id[group_with_id]
            catalog_id = raw_patient_id_2_catalog_id[raw_patient_id]
            patients_ids.append(catalog_id)
        return patients_ids

class All_Mitochondrial_Data:
    def __init__(self, morphology_file_name: str, protain_file_names: list[str], verbose: int = 0, class_print_read_data: bool = False) -> None:
        if verbose < 0:
            verbose = 0
        self.wells_collecter = Patient_Wells_Collecter(class_print_read_data=class_print_read_data)
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            self.morphology_data = Morphology_Data(morphology_file_name, verbose=verbose-1, class_print_read_data=class_print_read_data)
            protein_data = []
            for protein_name in protain_file_names:
                protein_data.append(Protein_Data(protein_name, verbose=verbose-1, class_print_read_data=class_print_read_data))
        self.protein_data = {protein.name: protein for protein in protein_data}
        self.wells_collecter.add_experiment('Mito morphology', self.morphology_data.patients_wells_counts, verbose=verbose)
        for protein_name, protein_data in self.protein_data.items():
             self.wells_collecter.add_experiment(f'Mito {protein_name}', protein_data.patients_wells_counts, verbose=verbose)
        if verbose:
            self.wells_collecter.display_table()
        