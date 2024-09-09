import os
import pandas as pd
from collections import defaultdict
from python.csv_pd import pd_display, csv_raw_data, csv_header_body_2_dataframe
from python.patients_wells import Patient_Wells_Collecter
from python.Patients_Catalog import Patients_Catalog
import warnings

class Protein_Data:
    def __init__(self, file_name, verbose: int = 0, class_print_read_data: bool = False) -> None:
        self.name, _ = os.path.splitext(os.path.basename(file_name))
        if verbose or class_print_read_data:
            print(f'Reading Mitochondrial protein data: {self.name}', flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            csv_data = csv_raw_data(file_name, verbose=max(0,verbose-2))
            self.df = csv_header_body_2_dataframe(csv_data[0], csv_data[1:], verbose=max(0,verbose-2))
        raw_patients_id = self.df.pop('Cell_id')
        catalog_id = Patients_Catalog(verbose=False).find_ID(raw_patients_id, verbose=max(0,verbose-1))
        self.df['Patient_ID'] = catalog_id
        self.patients_wells_counts = self.df.groupby('Patient_ID').size()
        self.patients_wells_counts.name = 'Num_Wells'
        if verbose:
            pd_display(self.patients_wells_counts)


class Morphology_Data:
    def __init__(self, file_name, verbose: int = 0, class_print_read_data: bool = False) -> None:
        if verbose or class_print_read_data:
            print(f'Reading Mitochondrial Morphology data: {file_name}', flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            csv_data = csv_raw_data(file_name, verbose=max(0,verbose-2))
            self.df = csv_header_body_2_dataframe(csv_data[0], csv_data[1:], verbose=max(0,verbose-2))
            patients_id = Morphology_Data.group_with_id_2_patient_id(self.df['group_with_id'], verbose=max(0,verbose-1))
        self.df['Patient_ID'] = patients_id
        self.patients_wells_counts = self.df.groupby('Patient_ID').size()
        self.patients_wells_counts.name = 'Num_Wells'
        if verbose:
            pd_display(self.patients_wells_counts)
    
    @staticmethod
    def group_with_id_2_patient_id(column_group_with_id, verbose: int = 0) -> list[str]:
        column_group_with_id = list(column_group_with_id)
        patients_catalog = Patients_Catalog(verbose=max(0,verbose-1))
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
        self.wells_collecter = Patient_Wells_Collecter(class_print_read_data=class_print_read_data)
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            self.morphology_data = Morphology_Data(morphology_file_name, verbose=max(0,verbose-1), class_print_read_data=class_print_read_data)
            protein_data = []
            for protein_name in protain_file_names:
                protein_data.append(Protein_Data(protein_name, verbose=max(0,verbose-1), class_print_read_data=class_print_read_data))
        self.protein_data = {protein.name: protein for protein in protein_data}
        self.wells_collecter.add_experiment('Mito morphology', self.morphology_data.patients_wells_counts, verbose=verbose)
        for protein_name, protein_data in self.protein_data.items():
             self.wells_collecter.add_experiment(f'Mito {protein_name}', protein_data.patients_wells_counts, verbose=verbose)
        if verbose:
            self.wells_collecter.display()
        