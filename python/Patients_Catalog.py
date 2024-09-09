import os
import pandas as pd
from python.csv_pd import pd_display, csv_raw_data, csv_header_body_2_dataframe, unique_list
import warnings

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
        cell_type = Patients_Catalog.df.loc[id, 'Cell_Type'].upper()
        if cell_type == 'HC':
            return 0.0
        if cell_type == 'HGPS':
            return 999.0
        return Patients_Catalog.df.loc[id, 'CAP_Score']
    
    def get_Age(self, id: str) -> float:
        return Patients_Catalog.df.loc[id, 'Age']

    def find_typo_ID(self, id_options: list[str], verbose: int = 0, original_ID: str|None = None) -> str|None:
        id_options = unique_list(id_options)
        if len(id_options) < 2:
            return self.find_ID(id_options[0], verbose=verbose, original_ID=original_ID)
        ret_id = None
        selected_id = None
        for id in id_options:
            try:
                ret_id = self.find_ID(id, verbose=verbose, original_ID=original_ID)
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
    
    def find_ID(self, id: pd.Index|pd.Series|str|list[str]|pd.arrays.IntegerArray, verbose: int = 0, original_ID: str|None = None) -> str|list[str]:
        if isinstance(id, (list, pd.Series, pd.Index, pd.arrays.IntegerArray)):
            id_elements = unique_list(list(id))
            assert not isinstance(id_elements[0], (list,tuple)), f'Original ID to find: {id}'
            mapping = {}
            failed_id = []
            for id_element in id_elements:
                try:
                    mapping[id_element] = self.find_ID(id_element, verbose=verbose, original_ID=None)
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
            return [mapping[id_element] for id_element in id]
        
        original_type = type(id)
        id = str(id).replace('\'','').replace(' ','').replace('-','')
        if not original_ID or original_ID == id:
            str_id = f'{id=}'
        else:
            str_id = f'{original_ID=} --> {id=}'
        if id in Patients_Catalog.df.index:
            str_warn = f'  Zero age!' if self.get_Age(id=id) < 1e-6 else ''
            if verbose or str_warn:
                if original_ID and original_ID != id:
                    str_verbose = 'Patients_Catalog: match '
                else:
                    str_verbose = 'Patients_Catalog: extact match '
                str_out = str_verbose + str_id + str_warn
                if verbose:
                    print(str_out)
                else:
                    warnings.warn(str_out)
            return id
                
        def count_leading_letters(s):
            count = 0
            for char in s:
                if char.isalpha():
                    count += 1
                else:
                    break
            return count
        
        if len(id) > 2 and id[-1].isdigit() and id[-2].isdigit() and int(id[-2:]) > 0:
            digits_to_trim = 2
        else:
            digits_to_trim = 0
        for trim in range(0,digits_to_trim+1):
            id_2_check = id[:len(id)-trim]
            if verbose > 1:
                print(f'{digits_to_trim=}  --> {id_2_check=}')

            # first check if "id" is inside any of the original catalog_id
            for catalog_id in Patients_Catalog.df.index:
                if verbose > 1:
                    print(f'Check if {str_id} --> {id_2_check=} in {catalog_id=}')
                if id_2_check in catalog_id:
                    str_warn = '  Zero age!' if self.get_Age(id=catalog_id) < 1e-6 else ''
                    if verbose or str_warn:
                        str_out = f'Patients_Catalog: match {str_id} --> {id_2_check=} --> {catalog_id}' + str_warn
                        if verbose:
                            print(str_out)
                        else:
                            warnings.warn(str_out)
                    return catalog_id
            # only later!!! check if "id" is inside any of modifications of catalog_id
            for catalog_id in Patients_Catalog.df.index:
                if verbose > 1:
                    print(f'Check if {str_id} --> {id_2_check=} in {catalog_id=}')
                catalog_id_list = []
                num_letters = count_leading_letters(catalog_id)
                if num_letters:
                    catalog_id_list.append(catalog_id[:num_letters] + '0' + catalog_id[num_letters:])
                    if catalog_id[num_letters] == '0':
                        catalog_id_list.append(catalog_id[:num_letters] + catalog_id[num_letters+1:])
                for catalog_id_2_check in catalog_id_list:
                    if id_2_check in catalog_id_2_check:
                        str_warn = f'  Zero age!' if self.get_Age(id=catalog_id) < 1e-6 else ''
                        if verbose or str_warn:
                            assert id != catalog_id
                            str_out = f'Patients_Catalog: match {str_id} --> {id_2_check=} --> {catalog_id_2_check=} --> {catalog_id}' + str_warn
                            if verbose:
                                print(str_out)
                            else:
                                warnings.warn(str_out)
                        return catalog_id
            
        raise ValueError(f'Could not find Patient_ID for type={original_type} {id}')
        
    def display(self) -> None:
        pd_display(Patients_Catalog.df)
