import pandas as pd
import numpy as np
from python.csv_pd import pd_display
from python.Patients_Catalog import Patients_Catalog
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Patient_Wells_Collecter:
    def __init__(self, class_print_read_data: bool = False) -> None:
        self.data = Patients_Catalog(verbose=False, class_print_read_data=class_print_read_data).df.copy()
        self.data['Count_Experiments'] = pd.Series([0] * len(self.data), index=self.data.index, dtype=pd.Int32Dtype())
        self.num_base_features_per_patient = self.data.shape[1]

    def add_experiment(self, experiment_name: str, data: pd.Series|pd.DataFrame, verbose: int = 0) -> None:
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
        experiment_patients = list(data.index)
        if not Patients_Catalog().all_ID_correct(experiment_patients):
            experiment_patients = Patients_Catalog(verbose=max(verbose-2,0)).find_ID(experiment_patients, verbose=max(verbose-1,0))
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
                new_patients = pd.DataFrame(index=new_patients, columns=self.data.columns, data=new_data, dtype=pd.Int32Dtype())
                print(f'{new_patients.dtypes=}')
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
                self.data[col_name] = pd.Series([pd.NA] * len(self.data), dtype=pd.Int32Dtype())
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
        self.data.loc[experiment_patients,'Count_Experiments'] += len(experiment_columns_data)

    def display_table(self) -> None:
        pd_display(self.__dataTable2export())

    def save_table_as_csv(self, filename: str) -> None:
        if not filename.endswith('.csv'):
            filename += '.csv'
        self.__dataTable2export().to_csv(filename, index=True)
        
    def save_table_as_excel(self, filename: str) -> None:
        if not filename.endswith('.xlsx'):
            filename += '.xlsx'
        self.__dataTable2export().to_excel(filename, index=True)

    def save_table_as_markdown(self, filename: str) -> None:
        if not filename.endswith('.md'):
            filename += '.md'

        markdown_table = self.__dataTable2export().to_markdown(index=True)

        with open(filename, 'w') as f:
            f.write(markdown_table)
    
    def __dataTable2export(self) -> pd.DataFrame:
        df_copy = self.data.copy()
        for column in df_copy.columns:
            if pd.api.types.is_integer_dtype(df_copy[column]) and df_copy[column].isna().any():
                df_copy[column] = df_copy[column].apply(lambda x: 'NA' if pd.isna(x) else str(int(x))).astype(str)
        return df_copy
    
    def display_cross_patients_all(self) -> None:
        self.display_cross_patients_diff(prefix_title='All Data - ')
        self.display_cross_patients_ratio(prefix_title='All Data - ')

    def display_cross_patients_No_Progeria(self) -> None:
        df = self.data.loc[self.data['Disease_Status'] != 'HGPS']
        df = df.loc[df['Age'] >= 18.0]
        self.display_cross_patients_diff(df=df, prefix_title='Without Progeria - ')
        self.display_cross_patients_ratio(df=df, prefix_title='Without Progeria - ')

    def display_cross_patients_diff(self, df: pd.DataFrame|None = None, prefix_title: str = '') -> None:
        if df is None:
            df = self.data
        experiments_names = list(df.columns)[self.num_base_features_per_patient:]
        num_experiments = len(experiments_names)
        cross_patients = np.empty(shape=(num_experiments,num_experiments), dtype=np.int32)
        for ind_experiment_source, name_experiment_source in enumerate(experiments_names):
            data_experiments_source = df[name_experiment_source]
            positive_source = (data_experiments_source > 0) & (data_experiments_source.notna())
            for ind_experiment_target, name_experiment_target in enumerate(experiments_names):
                if ind_experiment_source == ind_experiment_target:
                    cross_patients[ind_experiment_source,ind_experiment_target] = 0
                    continue
                data_experiments_target = df[name_experiment_target]
                positive_target = (data_experiments_target > 0) & (data_experiments_target.notna())
                count_missing = sum((positive_source) & (~positive_target))
                if count_missing:
                    res = -count_missing
                else:
                    res = sum((positive_target) & (~positive_source))
                cross_patients[ind_experiment_source,ind_experiment_target] = res
        # Plot the result
        # Calculate dynamic figure size based on the number of experiments
        fig_size = max(8, num_experiments * 0.8)  # Increase size based on number of experiments
        # Create the figure and axes
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        ax.set_title(prefix_title + 'Cross-Patients Differences')
        # Define custom colormap: red for negative, black for zero, green for positive
        cmap = ListedColormap(['red', 'black', 'green'])
        ind_color_map = np.sign(cross_patients)+1
        im = ax.imshow(ind_color_map, cmap=cmap)
        # Add text annotations (value of each cell) in white color
        for i in range(num_experiments):
            for j in range(num_experiments):
                plt.text(j, i, f'{cross_patients[i, j]}', ha='center', va='center', color='white')
        # Draw horizontal and vertical gridlines manually
        plt.hlines(np.arange(-0.5, num_experiments, 1), xmin=-0.5, xmax=num_experiments-0.5, colors='gray', linewidth=2, alpha=0.2)
        plt.vlines(np.arange(-0.5, num_experiments, 1), ymin=-0.5, ymax=num_experiments-0.5, colors='gray', linewidth=2, alpha=0.2)

        # Add row and column names (rotating the column names)
        ax.set_xticks(np.arange(num_experiments))
        ax.set_yticks(np.arange(num_experiments))
        ax.set_xticklabels(experiments_names, rotation=45, ha='right', rotation_mode='anchor', fontsize=10)
        ax.set_yticklabels(experiments_names, fontsize=10)

        # Add more space around the plot for better visualization of labels
        plt.subplots_adjust(left=0.3, bottom=0.3, top=0.95, right=0.95)
        # Show the plot
        plt.show()

    def display_cross_patients_ratio(self, df: pd.DataFrame|None = None, prefix_title: str = '') -> None:
        if df is None:
            df = self.data
        experiments_names = list(df.columns)[self.num_base_features_per_patient:]
        # filter out invalid experiments
        experiments_names = [col for col in experiments_names if (df[col] > 0 & df[col].notna()).any()]
        num_experiments = len(experiments_names)
        cross_patients = np.empty(shape=(num_experiments,num_experiments), dtype=np.int32)
        for ind_experiment_source, name_experiment_source in enumerate(experiments_names):
            data_experiments_source = df[name_experiment_source]
            positive_source = (data_experiments_source > 0) & (data_experiments_source.notna())
            assert positive_source.any()
            num_source = sum((positive_source))
            for ind_experiment_target, name_experiment_target in enumerate(experiments_names):
                if ind_experiment_source == ind_experiment_target:
                    cross_patients[ind_experiment_source,ind_experiment_target] = 100
                    continue
                data_experiments_target = df[name_experiment_target]
                positive_target = (data_experiments_target > 0) & (data_experiments_target.notna())
                count_missing = sum((positive_source) & (~positive_target))
                res = int((num_source - count_missing)/num_source*100 + 0.5)
                cross_patients[ind_experiment_source,ind_experiment_target] = res
        # Plot the result
        # Calculate dynamic figure size based on the number of experiments
        fig_size = max(8, num_experiments * 0.8)  # Increase size based on number of experiments
        # Create the figure and axes
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        ax.set_title(prefix_title + 'Cross-Patients Overlap Percentage (%)')
        # Use a continuous colormap (red for 0%, green for 100%)
        cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap
        cax = ax.imshow(cross_patients, cmap=cmap, vmin=0, vmax=100)  # vmin and vmax for scaling 0% to 100%
        # Add text annotations (value of each cell) in white color
        for i in range(num_experiments):
            for j in range(num_experiments):
                plt.text(j, i, f'{cross_patients[i, j]}', ha='center', va='center', color='white')
        # Draw horizontal and vertical gridlines manually
        plt.hlines(np.arange(-0.5, num_experiments, 1), xmin=-0.5, xmax=num_experiments-0.5, colors='gray', linewidth=2, alpha=0.2)
        plt.vlines(np.arange(-0.5, num_experiments, 1), ymin=-0.5, ymax=num_experiments-0.5, colors='gray', linewidth=2, alpha=0.2)

        # Add row and column names (rotating the column names)
        ax.set_xticks(np.arange(num_experiments))
        ax.set_yticks(np.arange(num_experiments))
        ax.set_xticklabels(experiments_names, rotation=45, ha='right', rotation_mode='anchor', fontsize=10)
        ax.set_yticklabels(experiments_names, fontsize=10)
        # Add a colorbar to show the mapping from 0% to 100%
        cbar = fig.colorbar(cax, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label('Overlap Percentage (%)')
        # Add more space around the plot for better visualization of labels
        plt.subplots_adjust(left=0.3, bottom=0.3, top=0.95, right=0.95)
        # Show the plot
        plt.show()
