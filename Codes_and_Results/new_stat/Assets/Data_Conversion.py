import os
import pandas as pd

class Data_Conversion:
    def __init__(self, input_path, output_path, Log):
        self.iPath = input_path
        self.oPath = output_path
        self.Log = Log

    def __call__(
            self, file_name, groupBase_column,
            singelValue_columns, index_columns
    ):
        self.Log(f'\n..:: Start Data_Conversion Data for \"{file_name}\" ::..')
        df = pd.read_csv(f'{self.iPath}/{file_name}.csv')

        self.create_outPutFolder(self.oPath)

        groupBaseColumn_df = df.groupby(groupBase_column)
        result = pd.DataFrame({groupBase_column: list(groupBaseColumn_df.groups.keys())})
        
        unWanted_columns = index_columns+singelValue_columns 
        
        for col_name in singelValue_columns:
            temp = pd.DataFrame({col_name: list(groupBaseColumn_df[col_name].last(1))})
            result =  pd.concat([result, temp], axis=1)

        for col_name in df.columns:
            if col_name in singelValue_columns:
                temp1 = groupBaseColumn_df.apply(lambda x: x.iloc[:-1][col_name].agg(['mean', 'median', 'min', 'max']))
                temp1.columns = [f'{col_name}_mean', f'{col_name}_median', f'{col_name}_min', f'{col_name}_max']
                temp1 = temp1.reset_index(drop=True)
                result = pd.concat([result, temp1], axis=1)

            elif col_name not in unWanted_columns:
                temp1 = groupBaseColumn_df.apply(lambda x: x[col_name].agg(['mean', 'median', 'min', 'max']))
                temp1.columns = [f'{col_name}_mean', f'{col_name}_median', f'{col_name}_min', f'{col_name}_max']
                temp1 = temp1.reset_index(drop=True)
                result = pd.concat([result, temp1], axis=1)
        
        result.to_csv(f'{self.oPath}/{file_name}.csv', index=False) 
        self.Log(f'..:: End Data_Conversion Data for \"{file_name}\" ::..')

    def create_outPutFolder(self, tmp_path):
        if not os.path.isdir(tmp_path):
            os.makedirs(tmp_path)