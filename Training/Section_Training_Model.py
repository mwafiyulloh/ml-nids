from model_structure import Model_IDS
from directories_training import *
import os

from data_preprocessing import Data_Preprocessing

class Section_Training_Model:
    def start_training(self):
        if os.path.isfile(dir_volume + '/time_for_training'):
            os.remove(dir_volume + '/time_for_training')
            return True

    def data_preprocessing(self):
        preprocess_data = Data_Preprocessing()
        df = preprocess_data.read_dataset(dir_volume)
        df = preprocess_data.delete_NaN_val(df)
        df = preprocess_data.filtering_data(df)
        train_x, test_x, train_y, test_y = preprocess_data.split_to_train_and_test(df)
        scaled_train_x, scaled_test_x = preprocess_data.data_standarization(train_x, test_x, dir_model)

        return [scaled_train_x, scaled_test_x, train_y, test_y]
    
    def design_model(self, train_x, test_x, train_y, test_y, dir_model):
        model = Model_IDS(train_x, test_x, train_y, test_y, train_x.shape[1])
        model.start_structuring()
        model.save_evaluation_model()
        model.save_models(dir_model)