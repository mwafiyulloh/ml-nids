import pandas as pd
from directories_training import dir_dataset
import os

def additional_data(df):
    """
    additional_data function is used for add new samples from secondary dataset if
    current dataset doesn't have samples of another category
    """
    print('\n==========================add new data to minority class==========================\n')

    print('Shape of dataset before add new data','\n',df.shape)

    try:
        df[df.columns[-1]].value_counts()[1]
    except KeyError:
        print('low instance of malicious data')
        sec_df = pd.read_csv(os.path.join(dir_dataset, 'Research_CICIDS2017.csv'))
        sec_df.columns = df.columns.tolist()
        samples = (lambda x,y: y if x > y else x)(df.shape[0], sec_df.shape[0])
        sec_df = sec_df[sec_df.iloc[:, -1].apply(lambda x: int(x))==1].sample(n=samples, replace=False)
        new_df = pd.concat([df,sec_df])
        return new_df

    try:
       df[df.columns[-1]].value_counts()[0]
    except KeyError:
        print('low instance of normal data')
        sec_df = pd.read_csv(os.path.join(dir_dataset, 'Research_CICIDS2017.csv'))
        sec_df.columns = df.columns.tolist()
        samples = (lambda x,y: y if x > y else x)(df.shape[0], sec_df.shape[0])
        sec_df = sec_df[sec_df.iloc[:, -1].apply(lambda x: int(x))==0].sample(n=samples, replace = False)
        new_df = pd.concat([df,sec_df])
        return new_df
    print('There is no need to add new data')
    return df

