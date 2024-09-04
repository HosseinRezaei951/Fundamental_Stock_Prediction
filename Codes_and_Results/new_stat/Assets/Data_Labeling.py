import os
import pandas as pd

class Data_Labeling:
    def __init__(self, input_path, output_path, Log):
        self.iPath = input_path
        self.oPath = output_path
        self.Log = Log

    def __call__(self, file_name, rules):
        self.Log(f'\n..:: Start Data_Labeling \"{file_name}\" by \"{rules}\" ::..')
        df = pd.read_csv(f'{self.iPath}/{file_name}.csv')

        self.create_outPutFolder(self.oPath)
        
        if("R2" in rules):
            df = self.calc_RCat2(df)
        if("B2" in rules):
            df = self.calc_BCat2(df)
        
        df.to_csv(f'{self.oPath}/{file_name}.csv', index=False) 
        self.Log(f'..:: End Data_Labeling \"{file_name}\" by \"{rules}\" ::..')

    def calc_RCat2(self, df):
        row_r_cat = []
        for _, row in df.iterrows():
            ret = row['Return']
            if(ret >33):
                row_r_cat.append(1)
            else:
                row_r_cat.append(-1)
        df['r_dicho'] = row_r_cat
        return df

    def calc_BCat2(self, df):
        row_b_cat = []
        for _, row in df.iterrows():
            beta = row['Beta']
            if(beta > 1 or beta < 0):
                row_b_cat.append(-1)
            elif(beta >= 0 and beta <= 1):
                row_b_cat.append(1)
        df['b_dicho'] = row_b_cat
        return df
    
    def create_outPutFolder(self, tmp_path):
        if not os.path.isdir(tmp_path):
            os.makedirs(tmp_path)
        
    