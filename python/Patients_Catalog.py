import os
import pandas as pd
from python.csv_pd import pd_display, csv_raw_data, csv_header_body_2_dataframe

class Patients_Catalog:
    df = None
    
    def __init__(self, file_name=None, verbose: int = 0) -> None:
        if Patients_Catalog.df is not None:
            return
        if file_name is None:
            file_name=os.path.join('python','resources', 'Patients_Catalog.csv')
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
    
    def find_any_ID(self, id_options: list[str], verbose: int = 0, original_ID: str|None = None) -> str|None:
        ret = None
        for id in id_options:
            try:
                ret = self.find_ID(id, verbose=verbose, original_ID=original_ID)
                break
            except ValueError as e:
                pass
        return ret

    def find_ID(self, id: pd.Index|pd.Series|str|list[str]|pd.arrays.IntegerArray, verbose: int = 0, original_ID: str|None = None) -> str|list[str]:
        if isinstance(id, (list, pd.Series, pd.Index, pd.arrays.IntegerArray)):
            id_elements = list(set(list(id)))
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
            if verbose:
                if original_ID and original_ID != id:
                    print(f'Patients_Catalog: match {str_id}')
                else:
                    print(f'Patients_Catalog: extact match {str_id}')
            return id
        def count_positive_digits_at_end(s):
            any_digits = 0
            found_zero = False
            count_positive = 0
            for char in reversed(s):
                if not char.isdigit():
                    break
                any_digits += 1
                if not found_zero:
                    if char == '0':
                        found_zero = True
                    else:
                        count_positive += 1
            return count_positive if any_digits else 0
        def count_leading_letters(s):
            count = 0
            for char in s:
                if char.isalpha():
                    count += 1
                else:
                    break
            return count
        if count_positive_digits_at_end(id[:-2]):
            digits_to_trim = 2
        else:
            digits_to_trim = 0
        for id_letters in set([id, id[count_leading_letters(id):]]):
            for trim in range(0,digits_to_trim+1):
                id_2_check = id_letters[:len(id_letters)-trim]
                # first check if "id" is inside any of the original catalog_id
                for catalog_id in Patients_Catalog.df.index:
                    if verbose > 1:
                        print(f'Check if {str_id} --> {id_2_check=} in {catalog_id=}')
                    if id_2_check in catalog_id:
                        if verbose:
                            assert original_ID != catalog_id
                            print(f'Patients_Catalog: match {str_id} --> {catalog_id}')
                        return catalog_id
                # only later!!! check if "id" is inside any of modifications of catalog_id
                for catalog_id in Patients_Catalog.df.index:
                    if verbose > 1:
                        print(f'Check if {str_id=} --> {id_2_check=} in {catalog_id=}')
                    catalog_id_list = []
                    num_letters = count_leading_letters(catalog_id)
                    if num_letters:
                        catalog_id_list.append(catalog_id[:num_letters] + '0' + catalog_id[num_letters:])
                        if catalog_id[num_letters] == '0':
                            catalog_id_list.append(catalog_id[:num_letters] + catalog_id[num_letters+1:])
                    for catalog_id_2_check in catalog_id_list:
                        if id_2_check in catalog_id_2_check:
                            if verbose:
                                assert id != catalog_id
                                print(f'Patients_Catalog: match {str_id} --> {catalog_id}')
                            return catalog_id
            
        # try removing prefix NA and add postfix 'C'
        def try_remove_prefix_NA(id: str):
            if not id.startswith('NA'):
                return None
            new_id = id[2:]
            if not new_id.endswith('C'):
                new_id = new_id + 'C'
            if new_id in Patients_Catalog.df.index:
                return new_id
            new_id = 'GM' + id[2:]
            if new_id in Patients_Catalog.df.index:
                return new_id
            return None
            
        res = try_remove_prefix_NA(id)
        if res is not None:
            return res
        # try to remove last two characters
        curr_len = len(id)
        for new_len in range(max(curr_len-2,1),curr_len):
            new_id = id[:new_len]
            if new_id in Patients_Catalog.df.index:
                return new_id
            res = try_remove_prefix_NA(new_id)
            if res is not None:
                return res
        raise ValueError(f'Could not find Patient_ID for type={original_type} {id}')
        
    def display(self) -> None:
        pd_display(Patients_Catalog.df)
