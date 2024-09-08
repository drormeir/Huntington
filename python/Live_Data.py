import os
import pandas as pd
from python.csv_pd import pd_display, csv_raw_data, csv_header_body_2_dataframe
from python.Patients_Catalog import Patients_Catalog

class Live_Data:
    def __init__(self, file_name: str|None = None, verbose: int = 0, class_print_read_data: bool = False) -> None:
        if file_name is None:
            file_name = os.path.join(os.getcwd(),'datasets','Live_data_noam.csv')
        if verbose or class_print_read_data:
            print(f'Reading Live data from {file_name=}')
        csv_data = csv_raw_data(file_name, verbose=verbose>2)
        df = csv_header_body_2_dataframe(csv_data[0], csv_data[1:], verbose=verbose>2, file_name=file_name).reset_index(drop=True)
        patients_catalog = Patients_Catalog(verbose=verbose>2)
        patient_ID = patients_catalog.find_ID(df['ID'], verbose=verbose>1)
        patient_ID = pd.DataFrame({'Patient_ID':patient_ID}).reset_index(drop=True)
        self.df = pd.concat([df, patient_ID], axis=1)
        self.patients_wells_counts = self.df.groupby('Patient_ID').size()
        self.patients_wells_counts.name = 'Num_Wells'
        if verbose:
            print('\nLive_Data Wells Count\n' + '='*50)
            pd_display(self.patients_wells_counts)
