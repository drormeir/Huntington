import pandas as pd
import re
import os
from python.csv_pd import pd_display, csv_raw_data, csv_header_body_2_dataframe
from python.Patients_Catalog import Patients_Catalog
import warnings

class Migration_Data:
    def __init__(self, file_name, verbose: int = 0, class_print_read_data: bool = False) -> None:
        assert os.path.isfile(file_name), f'Migration_Data: File does not exists: {file_name}'
        if verbose or class_print_read_data:
            print(f'Reading Migration_Data: {file_name}')
        csv_data = csv_raw_data(file_name, verbose=max(0,verbose-2))
        self.df = csv_header_body_2_dataframe(csv_data[0], csv_data[1:], verbose=max(0,verbose-2))
        cap_patient_2_true_id = {}
        patients_wells_count = {}
        patients_catalog = Patients_Catalog()
        cap_patient_column = self.df.pop('CAPPatient')
        cap_patient_column2 = self.df.pop('CAPPatient2')
        unique_cap_patient_column = cap_patient_column.unique()
        failed_cap_patient_2_test_id = {}
        for cap_patient in unique_cap_patient_column:
            raw_id = cap_patient.replace('\'',' ').split(' ')[-1]
            assert raw_id
            raw_id = raw_id.replace('IC','0').replace('pro','HGADFN')
            test_id_fix = [raw_id, raw_id.replace('4484', '484'), raw_id.replace('5687', '4687'), raw_id.replace('46146', '16146')]
            test_id_fix = list(set(test_id_fix)) # keep only unique options
            patient_id = patients_catalog.find_any_ID(test_id_fix, verbose=max(0,verbose-1), original_ID=raw_id)
            if not patient_id:
                failed_cap_patient_2_test_id[cap_patient] = test_id_fix
                patient_id = cap_patient
            else:
                catalog_cap = patients_catalog.get_CAP(patient_id)
                migration_cap = float(cap_patient.split('\'')[0])
                if abs(migration_cap - catalog_cap) >= 1:
                    warnings.warn(f'{cap_patient=} --> {patient_id=}  -> {catalog_cap=}')
            cap_patient_2_true_id[cap_patient] = patient_id
            patients_wells_count[patient_id] = cap_patient_column2[cap_patient_column==cap_patient].nunique()
            pass
        
        patient_id_col = [cap_patient_2_true_id[cap_patient] for cap_patient in cap_patient_column]
        new_df = pd.DataFrame({'Patient_ID':patient_id_col})
        self.df = pd.concat([self.df.reset_index(drop=True), new_df.reset_index(drop=True)], axis=1)
        self.patients_wells_count = pd.Series(patients_wells_count, name='Num_Wells').rename_axis('Patient_ID')
        self.cap_patient_2_true_id = pd.Series(cap_patient_2_true_id, name='Patient_ID').rename_axis('CAPPatient')
        if verbose:
            pd_display(self.cap_patient_2_true_id)
        if verbose:
            pd_display(self.patients_wells_count)
        if failed_cap_patient_2_test_id:
            failed_cap_patient = list(failed_cap_patient_2_test_id.keys())
            str_error = 'Failed to find true Patient_ID for:\n'
            if len(failed_cap_patient) == 1:
                str_error += f'cap patient: {failed_cap_patient[0]}\n'
            else:
               str_error += f'{len(failed_cap_patient)} cap patients: {failed_cap_patient}\n'
            str_error += f'Failed Cap Patients to ID options:\n{failed_cap_patient_2_test_id}\n'
            if len(failed_cap_patient) < len(cap_patient_2_true_id):
                str_error += f'out of {len(cap_patient_2_true_id)} cap patients; {list(cap_patient_2_true_id.keys())}\n'
            str_error += f'All possible candidates: {patients_catalog.get_ID()}'
            raise ValueError(str_error)

    @staticmethod
    def remove_leading_integer_and_quote(s: str|list[str]|pd.Series): # remove prefix cap
        pattern = r"^\d+'"
        if isinstance(s,str):
            return re.sub(pattern, '', s)
        return [re.sub(pattern, '', s0) for s0 in s]
    

class HGPS_Plate_Results:
    def __init__(self, file_name: str, verbose: int = 0, class_print_read_data: bool = False) -> None:
        assert os.path.isfile(file_name), f'HGPS_Plate_Results: File does not exists: {file_name}'
        if verbose or class_print_read_data:
            print(f'Reading HGPS_Plate_Results: {file_name}')
        csv_data = csv_raw_data(file_name, verbose=max(0,verbose-2))
        self.df = csv_header_body_2_dataframe(csv_data[0], csv_data[1:], verbose=max(0,verbose-2)).drop('CAP', axis=1)
        patient_type = self.df.pop('Cell_Type').reset_index(drop=True)
        col_patient_id = self.df.pop('Cell_ID').reset_index(drop=True)
        raw_unique_patient_id = col_patient_id.unique().tolist()
        catalog_patient_id = [str(id) for id in raw_unique_patient_id]
        catalog_patient_id = ['0'*max(0,4-len(id)) + id for id in catalog_patient_id]
        catalog_patient_id = Patients_Catalog().find_ID(catalog_patient_id, verbose=max(0,verbose-1))
        catalog_patient_type = Patients_Catalog.df.loc[catalog_patient_id, 'Cell_Type']
        catalog_patient_id2type = {id:t for id,t in zip(catalog_patient_id,catalog_patient_type)}
        convert_patient_id_2_catalog = {u:c for u,c in zip(raw_unique_patient_id,catalog_patient_id)}
        col_catalog_patient_id = []
        for row_patient_id, row_patient_type in zip(col_patient_id,patient_type):
            catalog_row_patient_id = convert_patient_id_2_catalog[row_patient_id]
            catalog_row_patient_type = catalog_patient_id2type[catalog_row_patient_id]
            assert row_patient_type == catalog_row_patient_type
            col_catalog_patient_id.append(catalog_row_patient_id)
        new_df = pd.DataFrame({'Patient_ID':col_catalog_patient_id})
        self.df = pd.concat([self.df.reset_index(drop=True), new_df.reset_index(drop=True)], axis=1)
        self.patients_wells_count = new_df['Patient_ID'].value_counts().rename('Num_Wells')
        if verbose:
            pd_display(self.patients_wells_count)

class HGPS_Data_APRW:
    def __init__(self, file_name: str, verbose: int = 0, class_print_read_data: bool = False) -> None:
        assert os.path.isfile(file_name), f'HGPS_Data_APRW: File does not exists: {file_name}'
        if verbose or class_print_read_data:
            print(f'Reading HGPS_Data_APRW: {file_name}')
        csv_data = csv_raw_data(file_name, verbose=max(0,verbose-2))
        self.df = csv_header_body_2_dataframe(csv_data[0], csv_data[1:], verbose=max(0,verbose-2)).drop('CAP', axis=1)
        col_patient_id = self.df['Patient/CELLID3']
        raw_unique_patient_id = col_patient_id.unique().tolist()
        patients_catalog = Patients_Catalog()
        convert_raw_patient_id_2_catalog = {}
        for raw_id in raw_unique_patient_id:
            raw_id = str(raw_id)
            original_id = raw_id.replace('\'','').replace(' ','').replace('-','')
            original_id0 = original_id.replace('IC', '0')
            test_id_fix = [original_id0, original_id0.replace('4484', '484'), original_id0.replace('5687', '4687'), original_id0.replace('46146', '16146')]
            test_id_fix = list(set(test_id_fix)) # keep only unique options
            patient_id = patients_catalog.find_any_ID(test_id_fix, verbose=max(0,verbose-1), original_ID=original_id)
            convert_raw_patient_id_2_catalog[raw_id] = patient_id
        col_catalog_patient_id = [convert_raw_patient_id_2_catalog[str(raw_patient_id)] for raw_patient_id in col_patient_id]
        new_df = pd.DataFrame({'Patient_ID':col_catalog_patient_id})
        self.df = pd.concat([self.df.reset_index(drop=True), new_df.reset_index(drop=True)], axis=1)
        self.patients_wells_count = new_df['Patient_ID'].value_counts().rename('Num_Wells')
        if verbose:
            pd_display(self.patients_wells_count)


class LatB_LowHigh_MitoQ_APRW:
    def __init__(self, file_name: str, verbose: int = 0, class_print_read_data: bool = False) -> None:
        assert os.path.isfile(file_name), f'LatB_LowHigh_MitoQ_APRW: File does not exists: {file_name}'
        if verbose or class_print_read_data:
            print(f'Reading LatB_LowHigh_MitoQ_APRW: {file_name}')
        csv_data = csv_raw_data(file_name, verbose=max(0,verbose-2))
        self.df = csv_header_body_2_dataframe(csv_data[0], csv_data[1:], verbose=max(0,verbose-2)).drop('CAP', axis=1)
        col_patient_id = self.df.pop('Patient/CELLID3')
        col_patient_id_well = self.df.pop('Patient/CELLID2')
        patients_catalog = Patients_Catalog()
        convert_raw_patient_id_2_catalog = {}
        for raw_id in col_patient_id.unique().tolist():
            raw_id = str(raw_id)
            raw_id0 = raw_id.split(' ')[-1]
            test_id_fix = [raw_id0, raw_id0.replace('4484', '484'), raw_id0.replace('5687', '4687'), raw_id0.replace('46146', '16146')]
            test_id_fix = list(set(test_id_fix)) # keep only unique options
            patient_id = patients_catalog.find_any_ID(test_id_fix, verbose=max(0,verbose-1))
            convert_raw_patient_id_2_catalog[raw_id] = patient_id
        col_catalog_patient_id = [convert_raw_patient_id_2_catalog[str(raw_patient_id)] for raw_patient_id in col_patient_id]
        new_df = pd.DataFrame({'Patient_ID':col_catalog_patient_id}).reset_index(drop=True)
        self.df = pd.concat([self.df.reset_index(drop=True), new_df], axis=1)
        patients_wells_count = {}
        for raw_id, cat_id in convert_raw_patient_id_2_catalog.items():
            patients_wells_count[cat_id] = col_patient_id_well[col_patient_id==raw_id].nunique()
        self.patients_wells_count = pd.Series(patients_wells_count, name='Num_Wells').rename_axis('Patient_ID')
        if verbose:
            pd_display(self.patients_wells_count)


