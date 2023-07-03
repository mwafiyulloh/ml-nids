import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from directories_training import *
from datetime import datetime, timedelta
from manual_arg import *

from additional_data import additional_data


class Data_Preprocessing:
    def read_dataset(self,dir): 
        print(f'\ntraining start at {datetime.now()}\n')
        print('\n==========================read dataset==========================\n')

        header_file = open(os.path.join(dir, 'header'))
        header_file.seek(os.SEEK_SET)
        header = header_file.read().split(',')

        df = pd.read_csv(os.path.join(dir, 'logs.csv'), names=header)
        
        if os.path.isfile(dir_volume + '/suggestion_data_training.txt'):
            print('found file for update training data')
            latest_file = os.path.join(dir, 'suggestion_data_training.txt')
            rows_file = open(latest_file, 'r')
            id_rows = []
            for line in rows_file:
                id_rows.append(line.strip())
            df.loc[df['_id'].isin(id_rows), 'timestamp'] = datetime.now() + timedelta(hours=time_interval_training)
            os.remove(dir + '/suggestion_data_training.txt')
            df.loc[df['_id'].isin(id_rows)].to_csv(dir + '/logs.csv', mode='a', header=False, index=False)
            
        return df
    
    def delete_NaN_val(self, df):
        print('\n==========================elimination NaN value==========================\n')
        print('total count of NaN values =', df.isnull().sum().sum())
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        return df
    
    def filtering_data(self, df):
        print('\n==========================filtering data==========================\n')
        current_time_packet = datetime.now()
        start_time = current_time_packet - timedelta(hours=time_interval_training)
        df[df.columns[-1]] = df[df.columns[-1]].astype(int)        
        df = df[df.iloc[:, 6].apply(lambda x: datetime.strptime(str(x)[:19], '%Y-%m-%d %H:%M:%S') < current_time_packet)]
        df = df[df.iloc[:, 6].apply(lambda x: datetime.strptime(str(x)[:19], '%Y-%m-%d %H:%M:%S') > start_time)]
        df = df.drop(df.columns[[0, 1, 2, 3, 5, 6]], axis=1) 

        df = additional_data(df)

        category = ['Normal', 'Malicious']
        print('Shape of current dataset =', df.shape)
        print(f'{category[0]} =', df[df.columns[-1]].value_counts()[0])
        print(f'{category[1]} =', df[df.columns[-1]].value_counts()[1])
        df.reset_index(inplace=True, drop=True)
        return df

    def split_to_train_and_test(self,df):
        print('\n==========================process splitting data==========================\n')
        train_x, test_x, train_y, test_y = train_test_split(df.iloc[:,:-1], df.iloc[:,-1], test_size = 0.2, random_state = 42)

        return train_x, test_x, train_y, test_y
    
    def data_standarization(self,train_x, test_x,dir):
        print('\n==========================process scaling data==========================\n')
        scaler = StandardScaler()
        scaled_train_x = scaler.fit_transform(train_x.values)
        scaled_test_x = scaler.transform(test_x.values)
        print('save scaler for model\n')

        pickle.dump(scaler, open(dir + '/scaler.pkl', 'wb'))

        return scaled_train_x, scaled_test_x
    

    