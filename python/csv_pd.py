from typing import Tuple
import warnings
import os
import csv
import numpy as np
import pandas as pd
from tabulate import tabulate
import re


pd.set_option('display.max_colwidth', 200)
pd.set_option('display.precision', 4)
pd.set_option('display.max_rows', 1000)


def pd_display(df: pd.DataFrame):
    try:
        from IPython.display import display
        display(df)
    except ImportError:
        print(tabulate(df, headers='keys', tablefmt='pretty'))



def csv_raw_data(file_name: str, verbose: bool = True) -> list[list[str]]:
    rows = []
    try:
        # The encoding parameter is to make sure that the data is without BOM characters
        with open(file_name, mode='r', encoding='utf-8-sig') as file:
            # Iterate through the rows in the CSV file
            for row in csv.reader(file):
                if not row:
                    continue
                row = [col.strip() for col in row]
                if not any([len(col) for col in row]):
                    continue # all cells are empty
                rows.append(row)
    except FileNotFoundError as e:
        print(f"Current working directory: {os.getcwd()}")
        raise e
    if verbose:
        print(f'Done reading raw csv file: {file_name} into {len(rows)} rows and {len(rows[0])} columns')
    return rows


def is_type_str_int_float(x):
    # Check for numeric types directly
    if isinstance(x, (int, np.int32, np.int64)):
        return pd.Int32Dtype()
    if isinstance(x, (float, np.float32)):
        return np.float32

    # Handle non-string and special string cases
    if not isinstance(x, str):
        return object
    x = x.strip().lower()
    
    if x in {'na', '<na>', 'nan', 'none', ''}:
        return None
    if x in {'inf', '-inf'}:
        return np.float32
    
    # Attempt to parse as an integer
    try:
        int_val = int(x)
        return pd.Int32Dtype()
    except ValueError:
        pass
    
    # Attempt to parse as a float
    try:
        float_val = float(x)
    except ValueError:
        return str

    # Check if the float value is close to an integer
    if abs(float_val - round(float_val)) < 1e-10:
        return pd.Int32Dtype()

    return np.float32

def convert_non_string_columns(df: pd.DataFrame, verbose: int = 0) -> Tuple[list[str], list[str], list[str]]:
    int_columns, float_columns, string_columns = [], [], []
    
    for col in df.columns:
        col_types = set(is_type_str_int_float(x) for x in df[col])
        
        if verbose == 2:
            print(f'Column {col} counting occurrences...')
            for u_type in col_types:
                print(f'Type: {u_type} has {sum(1 for t in col_types if t == u_type)} occurrences.')

        col_types.discard(None)
        
        target_list = string_columns
        if len(col_types) == 1:
            single_type = col_types.pop()
            if single_type == pd.Int32Dtype():
                target_list = int_columns
            elif single_type == np.float32:
                target_list = float_columns
        elif col_types <= {pd.Int32Dtype(), np.float32}:
            single_type = np.float32
            target_list = float_columns
        else:
            single_type = None
        target_list.append(col)

        if single_type in [pd.Int32Dtype(), np.float32]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                df[col] = pd.to_numeric(df[col].astype(str), errors='coerce').astype(single_type)
                

    if verbose:
        print(f'{len(int_columns)} Integers columns: {int_columns}')
        print(f'{len(float_columns)} Float columns: {float_columns}')
        print(f'{len(string_columns)} Strings columns: {string_columns}')
    return int_columns, float_columns, string_columns


def csv_header_body_2_dataframe(header: list[str], body: list[list[str]],\
                                rename_header: dict = {},\
                                replace_header_value: list = [(' - ','_'), ('- ','_'), (' ','_')],\
                                verbose: int = 1,\
                                file_name: str = '') -> pd.DataFrame:
    if len(set(header)) < len(header):
        # remove duplicate columns by header name only
        ind_unique = []
        seen = set()
        duplicates = []
        for ind, col in enumerate(header):
            if col in seen:
                duplicates.append(col)
            else:
                seen.add(col)
                ind_unique.append(ind)
        str_file = f'For file: "{file_name}" ' if file_name else ''
        warnings.warn(f'{str_file}There are {len(duplicates)} duplicate columns to be removed:\n{duplicates}')
        header = [header[ind_col] for ind_col in ind_unique]
        new_body = []
        for row in body:
             new_body.append([row[ind_col] for ind_col in ind_unique])
        body = new_body
        
    if rename_header:
        rename_before = all([h in header for h in rename_header.keys()])
        if rename_before:
            header = [rename_header[h] if h in rename_header else h for h in header]
    for rep0, rep1 in replace_header_value: # underscoring the header
        header = [h.replace(rep0, rep1) for h in header]
    if rename_header and not rename_before:
        # second try to rename
        assert all([h in header for h in rename_header.keys()])
        header = [rename_header[h] if h in rename_header else h for h in header]
    df = pd.DataFrame(body, columns=header)
    _, _, _ = convert_non_string_columns(df, verbose=verbose)
    if verbose:
        if file_name:
            print(f'Printing columns of file: {file_name}')
        for column in df.columns:
            print(f"Column: {column}, Data Type: {df[column].dtype}")
        pd_display(df.head(5))
    return df


def extract_integer_from_string_end(s: str):
    assert isinstance(s, str)
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    else:
        return None

def unique_list(data: list[str]) -> list[str]:
    unique_data = set()
    ret = []
    for x in data:
        if x in unique_data:
            continue
        ret.append(x)
        unique_data.add(x)
    return ret
