import pandas as pd
import re
import os
from python.csv_pd import pd_display, csv_raw_data, csv_header_body_2_dataframe, unique_list
from python.Patients_Catalog import Patients_Catalog
import warnings

class Migration_Data:
    def try_different_id(id0: str) -> list[str]:
        id0 = id0.replace('IC','0').replace('pro','HGADFN')
        test_id_fix = []
        id0_2_check = unique_list([id0, id0.replace('NA', 'GM'), id0.replace('NA', 'AG')])
        assert id0_2_check[0] == id0
        for id in id0_2_check:
            test_id_fix += [id, id.replace('4484', '484').replace('5687', '4687').replace('46146', '16146').replace('0170','1170')]
        if id0 in ['16', 'NA0016']:
            test_id_fix.append('CM16')
        elif id0 == '13':
            test_id_fix.append('HGADFN014313')
        return unique_list(test_id_fix)

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
            test_id_fix = Migration_Data.try_different_id(raw_id)
            patient_id = patients_catalog.find_typo_ID(test_id_fix, verbose=max(0,verbose-1), original_ID=raw_id)
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
        patients_catalog = Patients_Catalog()

        csv_data = csv_raw_data(file_name, verbose=max(0,verbose-2))
        self.df = csv_header_body_2_dataframe(csv_data[0], csv_data[1:], verbose=max(0,verbose-2)).drop('CAP', axis=1)
        col_patient_type = self.df.pop('Cell_Type').reset_index(drop=True)
        col_patient_id = self.df.pop('Cell_ID').reset_index(drop=True).astype(str)
        raw_data_2_catalog_id = {}
        col_catalog_patient_id = []
        for raw_patient_id, patient_type in zip(col_patient_id,col_patient_type):
            patient_id_type = raw_patient_id + '_' + patient_type
            if patient_id_type not in raw_data_2_catalog_id:
                test_id_fix = Migration_Data.try_different_id(raw_patient_id)
                catalog_id = patients_catalog.find_typo_ID(test_id_fix, verbose=max(0,verbose-1), original_ID=raw_patient_id, patient_type=patient_type)
                raw_data_2_catalog_id[patient_id_type] = catalog_id
            else:
                catalog_id = raw_data_2_catalog_id[patient_id_type]
            col_catalog_patient_id.append(catalog_id)

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
            test_id_fix = Migration_Data.try_different_id(original_id)
            patient_id = patients_catalog.find_typo_ID(test_id_fix, verbose=max(0,verbose-1), original_ID=original_id)
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
        original_id_2_catalog_id = {}
        for raw_id in col_patient_id.unique().tolist():
            raw_id = str(raw_id)
            original_id = raw_id.split(' ')[-1]
            if original_id not in original_id_2_catalog_id:
                test_id_fix = Migration_Data.try_different_id(original_id)
                original_id_2_catalog_id[original_id] = patients_catalog.find_typo_ID(test_id_fix, verbose=max(0,verbose-1), original_ID=original_id)
            convert_raw_patient_id_2_catalog[raw_id] = original_id_2_catalog_id[original_id]
        col_catalog_patient_id = [convert_raw_patient_id_2_catalog[str(raw_patient_id)] for raw_patient_id in col_patient_id]
        new_df = pd.DataFrame({'Patient_ID':col_catalog_patient_id}).reset_index(drop=True)
        self.df = pd.concat([self.df.reset_index(drop=True), new_df], axis=1)
        patients_wells_count = {}
        for raw_id, cat_id in convert_raw_patient_id_2_catalog.items():
            patients_wells_count[cat_id] = col_patient_id_well[col_patient_id==raw_id].nunique()
        self.patients_wells_count = pd.Series(patients_wells_count, name='Num_Wells').rename_axis('Patient_ID')
        if verbose:
            pd_display(self.patients_wells_count)


