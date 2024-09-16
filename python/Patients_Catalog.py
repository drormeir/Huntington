import os
import pandas as pd
from python.csv_pd import pd_display, csv_raw_data, csv_header_body_2_dataframe, unique_list
import warnings
from typing import Union

class Patients_Catalog:
    df = None
    
    def __init__(self, file_name=None, verbose: int = 0, class_print_read_data: bool = False) -> None:
        if Patients_Catalog.df is not None:
            return
        if file_name is None:
            file_name=os.path.join('python','resources', 'Patients_Catalog.csv')
        if verbose or class_print_read_data:
            print(f'Reading Patients Catalog from: {file_name=}', flush=True)
        csv_data = csv_raw_data(file_name, verbose=verbose>1)
        Patients_Catalog.df = csv_header_body_2_dataframe(csv_data[0], csv_data[1:], verbose=verbose>1).set_index('Patient_ID')
        if verbose:
            print(f'Patients_Catalog CSV body length: {len(csv_data)-1} rows')
            self.display()
            
    def get_ID(self) -> list[str]:
        return list(Patients_Catalog.df.index)

    def get_CAP(self, id: str) -> float:
        disease_status = Patients_Catalog.df.loc[id, 'Disease_Status']
        if disease_status == 'Healthy':
            return 0.0
        if disease_status == 'HGPS':
            return 999.0
        return Patients_Catalog.df.loc[id, 'CAP_Score']
    
    def get_Age(self, id: str) -> float:
        return Patients_Catalog.df.loc[id, 'Age']

    def find_typo_ID(self, id_options: list[str], verbose: int = 0, original_ID: str|None = None, disease_status: str|None = None) -> str|None:
        assert id_options
        id_options = unique_list(id_options)
        if len(id_options) < 2:
            return self.find_ID(id_options[0], verbose=verbose, original_ID=original_ID, disease_status=disease_status)
        ret_id = None
        selected_id = None
        for id in id_options:
            try:
                ret_id = self.find_ID(id, verbose=verbose, original_ID=original_ID, disease_status=disease_status)
                if ret_id:
                    selected_id = id
                    break
            except ValueError as e:
                pass
        if selected_id and selected_id != id_options[0]:
            str_warn = 'Found typo!! '
            if original_ID and original_ID != id_options[0]:
                str_warn += f'{original_ID=} --> '
            str_warn += f'base_id={id_options[0]} --> {selected_id=} --> {ret_id=}'
            warnings.warn(str_warn)
        return ret_id

    def all_ID_correct(self, ids: list[str]) -> bool:
        my_IDs = self.get_ID()
        return all(id in my_IDs for id in ids)
    
    def find_ID(self, id: pd.Index|pd.Series|str|list[str]|pd.arrays.IntegerArray, verbose: int = 0, original_ID: str|None = None, disease_status: str|None = None) -> str|list[str]:
        if isinstance(id, (list, pd.Series, pd.Index, pd.arrays.IntegerArray)):
            id_elements = unique_list(list(id))
            assert not isinstance(id_elements[0], (list,tuple)), f'Original ID to find: {id}'
            mapping = {}
            failed_id = []
            for id_element in id_elements:
                try:
                    mapping[id_element] = self.find_ID(id_element, verbose=verbose, original_ID=None, disease_status=disease_status)
                except ValueError as e:
                    failed_id.append(id_element)
            if failed_id:
                if len(failed_id) < len(id_elements):
                    str_error = \
                        f'Failed to find true Patient_ID for:\n' +\
                        f'{len(failed_id)} data ID: {failed_id}\n'+\
                        f'out of {len(id_elements)} data IDs; {id_elements}\n'+\
                        f'All possible candidates are: {self.get_ID()}'
                else:
                    str_error = \
                        f'Failed to find true Patient_ID for:\n' +\
                        f'{len(failed_id)} data ID: {failed_id}\n'+\
                        f'All possible candidates are: {self.get_ID()}'
                raise ValueError(str_error)
            if verbose>1:
                print(f'{mapping=}')
            return [mapping[id_element] for id_element in id]
        
        original_type = type(id)
        id = str(id).replace('\'','').replace(' ','').replace('-','')
        if not original_ID or original_ID == id:
            str_id = f'{id=}'
        else:
            str_id = f'{original_ID=} --> {id=}'

        if len(id) > 2 and id[-1].isdigit() and id[-2].isdigit() and int(id[-2:]) > 0:
            ids_trim_passage = [id[:len(id)-trim] for trim in range(0,3)]
        else:
            ids_trim_passage = [id]
        if verbose > 1:
            print(f'{ids_trim_passage=}')
        for id_2_check in ids_trim_passage:
            if id_2_check in Patients_Catalog.df.index:
                zero_age = self.get_Age(id=id_2_check) < 1e-6
                if verbose or zero_age:
                    if original_ID and original_ID != id_2_check:
                        str_out = f'Patients_Catalog: match {str_id} --> catalog_id={id_2_check}'
                    else:
                        str_out = 'Patients_Catalog: extact match ' + str_id
                    if zero_age:
                        str_out += '  Zero age!'
                    if verbose:
                        print(str_out)
                    else:
                        warnings.warn(str_out)
                return id_2_check
                
        def count_leading_letters(s):
            count = 0
            for char in s:
                if char.isalpha():
                    count += 1
                else:
                    break
            return count
        
        if disease_status:
            assert disease_status in ['Healthy', 'HGPS', 'HD_Premanifest', 'HD_Mild', 'HD_Severe']
            mask = [disease_status in status for status in Patients_Catalog.df['Disease_Status']]
            mask = pd.Series(mask, index=Patients_Catalog.df.index)
            all_catalog_ids = Patients_Catalog.df.index[mask]
        else:
            all_catalog_ids = Patients_Catalog.df.index
        modified_catalog_ids = []
        for catalog_id in all_catalog_ids:
            num_letters = count_leading_letters(catalog_id)
            if num_letters < len(catalog_id):
                modified_catalog_ids.append((catalog_id[:num_letters] + '0' + catalog_id[num_letters:],catalog_id))
                if catalog_id[num_letters] == '0':
                    modified_catalog_ids.append((catalog_id[:num_letters] + catalog_id[num_letters+1:],catalog_id))
        all_catalog_ids = list(zip(all_catalog_ids,all_catalog_ids)) + modified_catalog_ids
        for id_2_check in ids_trim_passage:
            # first check if "id" is inside any of the original catalog_id
            for modified_catalog_id,original_catalog_id in all_catalog_ids:
                if original_catalog_id == modified_catalog_id:
                    str_catalog_id = f'catalog_id={original_catalog_id}'
                else:
                    str_catalog_id = f'{modified_catalog_id=} --> {original_catalog_id=}'
                if verbose > 1:
                    print(f'Check if {str_id} --> {id_2_check=} in {str_catalog_id}')
                if modified_catalog_id.endswith(id_2_check):
                    str_warn = '  Zero age!' if self.get_Age(id=original_catalog_id) < 1e-6 else ''
                    if verbose or str_warn:
                        str_out = f'Patients_Catalog: match {str_id} --> {id_2_check=} --> {str_catalog_id}' + str_warn
                        if verbose:
                            print(str_out)
                        else:
                            warnings.warn(str_out)
                    return original_catalog_id
            
        raise ValueError(f'Could not find Patient_ID for type={original_type} {id}')
        
    def is_not_HGPS(self, patient_ids: list[str]|pd.Series) -> Union[list[bool], pd.Series]:
        ret = [Patients_Catalog.df.loc[id, 'Disease_Status'] != 'HGPS' and Patients_Catalog.df.loc[id, 'Age'] >= 18.0 for id in patient_ids]
        if isinstance(patient_ids, list):
            return ret
        return pd.Series(data=ret, index=patient_ids.index)
    
    def is_healthy(self, patient_ids: list[str]|pd.Series) -> Union[list[bool], pd.Series]:
        ret = [Patients_Catalog.df.loc[id, 'Disease_Status'] == 'Healthy' for id in patient_ids]
        if isinstance(patient_ids, list):
            return ret
        return pd.Series(data=ret, index=patient_ids.index)
    
    def display(self) -> None:
        pd_display(Patients_Catalog.df)
