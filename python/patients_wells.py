import pandas as pd
from python.csv_pd import pd_display
from python.Patients_Catalog import Patients_Catalog

class Patient_Wells_Collecter:
    def __init__(self) -> None:
        all_patients_ID = Patients_Catalog(verbose=False).get_ID()
        self.data = pd.DataFrame(index=pd.Index([], name='Patient_ID')).reindex(all_patients_ID)
        
    def add_experiment(self, experiment_name: str, data: pd.Series|pd.DataFrame, verbose: bool = False) -> None:
        if data.index.name == 'Patient_ID':
            # input data is ready
            pass
        elif isinstance(data, pd.Series):
            assert data.name == 'Num_Wells'
            assert data.index.name == 'Patient_ID', f'Inpu data Series index name is {data.index.name=} instead of \'Patient_ID\''
        elif isinstance(data, pd.DataFrame):
            assert 'Patient_ID' in data.columns, f'{data.index.name=} {data.columns=}'
            data = data.groupby('Patient_ID').size()
            data.name = 'Num_Wells'
        else:
            assert False, 'Invalid data type for add_experiment()'
        experiment_patients = Patients_Catalog(verbose=False).find_ID(data.index)
        # adding None rows for new patients
        new_patients = [patient_id for patient_id in experiment_patients if patient_id not in self.data.index]
        if new_patients:
            if verbose:
                print(f'Patient_Wells_Collecter.add_experiment() new patient_ids: {new_patients}')
            if self.data.columns.empty:           
                self.data = self.data.reindex(list(self.data.index) + new_patients)
                assert self.data.index.name == 'Patient_ID'
            else:
                new_data = [[pd.NA] * len(self.data.columns)]* len(new_patients)
                new_patients = pd.DataFrame(index=new_patients, columns=self.data.columns, data=new_data, dtype=pd.Int64Dtype())
                new_patients.index.name = 'Patient_ID'
                self.data = pd.concat([self.data, new_patients])
                assert self.data.index.name == 'Patient_ID'
        
        if isinstance(data, pd.DataFrame):
            experiment_columns_names = data.columns
            experiment_columns_data = [data[col] for col in data.columns]
        else:
            assert isinstance(data, pd.Series)
            experiment_columns_names = [experiment_name]
            experiment_columns_data = [data]
        
        for col_name, col_data in zip(experiment_columns_names,experiment_columns_data):
            # adding column of None for a new experiment
            if col_name not in self.data.columns:
                self.data[col_name] = pd.Series([pd.NA] * len(self.data), dtype=pd.Int64Dtype())            
            # copy experiment data
            try:
                self.data.loc[experiment_patients, col_name] = col_data
            except ValueError as e:
                new_error = str(e) + '\n'
                new_error += 'When calling add_experiment:\n'
                new_error += f'experiment_name={experiment_name}\n'
                new_error += f'current column={col_name}\n'
                new_error += f'columns={experiment_columns_names}\n'
                new_error += f'new_patients={new_patients}\n'
                new_error += f'Existing columns: {self.data.columns}\n'
                new_error += f'experiment_patients={experiment_patients}\n'
                new_error += f'col_data={col_data}'
                raise ValueError(new_error)
        
    def display(self) -> None:
        pd_display(self.data)

    def to_csv_file(self, filename) -> None:
        self.data.to_csv(filename, index=False)
        
