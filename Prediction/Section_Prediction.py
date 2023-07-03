from directories_prediction import dir_volume, dir_model, dir_tmpfile
from manual_arg import *

import csv
import pickle
from datetime import datetime,timedelta
from mqtt_client import mqtt_client
import os 
import time
import json
import time
from tensorflow.keras.models import load_model

class Section_Prediction:
    def __init__(self) -> None:
        self.initialize_model()

        output = open(dir_volume + '/logs.csv', "a")
        self.csv_writer = csv.writer(output)
        self.current_time = datetime.now()
        self.end_time = self.current_time + timedelta(hours=time_interval_training)
        
    def initialize_model(self):
        self.scaler = pickle.load(open(dir_model + '/scaler.pkl', 'rb'))
        self.Encoder_1 = load_model(dir_model + "/Encoder1")
        self.Encoder_2 = load_model(dir_model + "/Encoder2")
        self.model = pickle.load(open(dir_model + '/model_ids.pkl', 'rb'))

    def get_event(self,dir,rownum) :
        time.sleep(1)
        try:
            latest_file = os.path.join(dir, f'{str(rownum)}_tmpfile')
            file = open(latest_file)
            file.seek(os.SEEK_SET)
            rowlist = file.read().split(',')
            original_log = rowlist.copy()
            mata_elang_file = os.path.join(dir, f'{str(rownum)}_tmp_mata_elang.json')
            mata_elang = json.load(open(mata_elang_file, 'r'))
            # [0,1,2,3,5,6]
            del rowlist[:4]
            del rowlist[1:3]
            os.remove(latest_file)
            os.remove(mata_elang_file)
            print('success retrieved the file')
            return rowlist, mata_elang, original_log
        except Exception as e:
            print('Unable to find the file')
            return None, None, None
    
    def predict(self, input_row, log:list):

        category = ['Normal', 'Malicious']

        row = [[float(el) for el in input_row]]
        # scaling
        new_data = self.scaler.transform(row)
        # dimensionality reduction
        new_data = self.Encoder_2.predict(self.Encoder_1.predict(new_data))
        # prediction
        preds = self.model.predict(new_data)

        log.append(int(preds[0]))
        print('flows identified as', category[int(preds[0])],'\n')

        return log
    
    def save_log_csv(self, log):
        if os.path.isfile(os.path.join(dir_tmpfile, 'header')):
            header_path = os.path.join(dir_tmpfile, 'header')
            header_file = open(header_path)
            header_file.seek(os.SEEK_SET)
            header = header_file.read().split(',')
            header.append('Label')

            write_header_file = open(os.path.join(dir_volume, 'header'), 'w')
            write_header_file.write(','.join(header))
            write_header_file.close()

        self.csv_writer.writerow(log)
           
    def save_to_dc(self, log, result, client):
        category = ['Normal', 'Malicious']
        log['label'] = category[result]

        client.send_log(log)

    def start_training_model(self):
        self.current_time = datetime.now()
        if self.current_time > self.end_time:
            print('\nTime for training model!\n')
            open(dir_volume + '/time_for_training', 'w').close()
            self.end_time = self.current_time + timedelta(hours=time_interval_training)

    def update_model(self):
        if os.path.isfile(dir_model + '/time_to_update_model'):
            print('Time to update model!')
            self.initialize_model()
            os.remove(dir_model + '/time_to_update_model')
        
    def run_prediction(self, rownum):      
        try:
            client = mqtt_client()
            client.on_connect_mqtt()
            while(True):
                rowlist, mata_elang_event, original_log = self.get_event(dir_tmpfile, rownum)
                if rowlist != None:
                    log = self.predict(input_row=rowlist, log=original_log )
                    self.save_log_csv(log)
                    # save to defence center 
                    self.save_to_dc(mata_elang_event, log[-1], client)
                    rownum += 1
                self.start_training_model()
                self.update_model()
        except KeyboardInterrupt:
            client.disconnect_mqtt()
            return