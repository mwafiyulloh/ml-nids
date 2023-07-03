from Section_Training_Model import *

def main():
    print('\nwaiting for training time comes...')
    try:
        while(True):
            sec_train = Section_Training_Model()
            start_bool = sec_train.start_training()
            if start_bool:
                dataset = sec_train.data_preprocessing()
                if dataset == None:
                    continue
                train_x, test_x, train_y, test_y = dataset[0], dataset[1], dataset[2], dataset[3]
                sec_train.design_model(train_x, test_x, train_y, test_y, dir_model)
                print('\nwaiting for training time comes...')
    except Exception as e:
        print('Training Project Stop. Exception ' + str(e))
        return

if __name__ == "__main__":
    main()