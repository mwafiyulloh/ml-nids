from directories_training import *
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score
from datetime import datetime
import numpy as np 
import matplotlib.pyplot as plt
import os
import PyPDF2

class DenseTranspose(tf.keras.layers.Layer):
    def __init__(self, dense, activation=None, **kwargs):
        self.dense = dense
        self.activation = tf.keras.activations.get(activation)
        super().__init__(**kwargs)
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'dense': self.dense,
            'activation': self.activation
        })
        return config
    def build(self, batch_input_shape):
        self.biases = self.add_weight(name="bias", initializer="zeros",shape=[self.dense.input_shape[-1]])
        self.W = tf.transpose(self.dense.weights[0]) 
        super().build(batch_input_shape)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dense.input_shape[-1])
    def call(self, inputs):
        z = tf.matmul(inputs, self.W)
        return self.activation(z + self.biases)

class Model_IDS:

    def __init__(self, train_x, test_x, train_y, test_y, input_shape) -> None:
       self.train_x = train_x
       self.test_x = test_x
       self.train_y = train_y
       self.test_y = test_y
       self.result_prediction = None
       self.num_hidden = (input_shape, 14, 28, 28)

       self.Encoder_1 = None
       self.Encoder_2 = None
       self.model_ids = None 
    
    def start_structuring(self):
        AE_1_encoded_train, AE_1_encoded_test = self.autoencoder_1()
        AE_2_encoded_train, AE_2_encoded_test = self.autoencoder_2(AE_1_encoded_train, AE_1_encoded_test)
        self.baseline_model(AE_2_encoded_train, AE_2_encoded_test)
        
    def autoencoder_1(self):
        print('\n==================Training 1st Autoencoder==================\n')

        Dense_11 = Dense(units=self.num_hidden[1], activation='sigmoid')
        Dense_12 = Dense(units=self.num_hidden[2], activation='sigmoid')
        Dense_13 = Dense(units=self.num_hidden[3], activation='sigmoid')

        inputs_1 = Input(shape=(self.num_hidden[0],))

        #Encoder
        encoded_11 = Dense_11(inputs_1)
        encoded_12 = Dense_12(encoded_11)
        encoded_13 = Dense_13(encoded_12)

        #Decoder
        decoded_11 = DenseTranspose(Dense_13, activation='sigmoid')(encoded_13)
        decoded_12 = DenseTranspose(Dense_12, activation='sigmoid')(decoded_11)
        outputs_1 = DenseTranspose(Dense_11, activation='sigmoid')(decoded_12)

        AE_1=Model(inputs_1, outputs_1)
        self.Encoder_1=Model(inputs_1, decoded_12)
        print(self.Encoder_1.summary())

        AE_1.compile(optimizer='rmsprop', loss= 'mse')
        AE_1.fit(self.train_x,self.train_x,epochs=10,batch_size=128,shuffle=True)
        print('Done Training\n')
        print('Encoding train data and test data')
        AE_1_encoded_train = self.Encoder_1.predict(self.train_x)
        AE_1_encoded_test = self.Encoder_1.predict(self.test_x)
        
        
        print('Done\n')

        return AE_1_encoded_train, AE_1_encoded_test

    def autoencoder_2(self, AE_1_encoded_train, AE_1_encoded_test):
        print('\n==================Training 2nd Autoencoder==================\n')

        Dense_21 = Dense(units=self.num_hidden[2], activation='sigmoid')
        Dense_22 = Dense(units=self.num_hidden[3], activation='sigmoid')

        inputs_2 = Input(shape=(self.num_hidden[1],))

        #Encoder
        encoded_21 = Dense_21(inputs_2)
        encoded_22 = Dense_22(encoded_21)

        #Decoder
        decoded_21 = DenseTranspose(Dense_22, activation='sigmoid')(encoded_22)
        outputs_2 = DenseTranspose(Dense_21, activation='sigmoid')(decoded_21)

        AE_2=Model(inputs_2, outputs_2)
        self.Encoder_2=Model(inputs_2, decoded_21)
        print(self.Encoder_2.summary())

        AE_2.compile(optimizer='rmsprop', loss= 'mse')
        print('Start Training\n')
        AE_2.fit(AE_1_encoded_train,AE_1_encoded_train,epochs=10,batch_size=128,shuffle=True)
        print('Done Training\n')
        print('Encoding train data and test data....')
        AE_2_encoded_train = self.Encoder_2.predict(AE_1_encoded_train)
        AE_2_encoded_test = self.Encoder_2.predict(AE_1_encoded_test)
        
        print('Done\n')

        return AE_2_encoded_train, AE_2_encoded_test

    def baseline_model(self, AE_2_encoded_train, AE_2_encoded_test):
        print('==================Training Baseline Model (Random Forest)==================')
        rfc = RandomForestClassifier(n_jobs=-1, n_estimators=10)
        print('Start Training\n')
        rfc.fit(AE_2_encoded_train, self.train_y)
        print('Done Training\n')
        self.result_prediction = rfc.predict(AE_2_encoded_test)
        self.model_ids = rfc
        print("The entire model training has been completed.")

        return

    def save_evaluation_model(self):
        print('\n==================Save Evaluation Model==================\n')

        current_time = datetime.now()
        current_time = datetime.strftime(current_time, '%Y-%m-%dT%H-%M-%S')
        path = os.path.join(dir_evaluation, f'eval_{str(current_time)}')
        os.mkdir(path)

        pr_curve = os.path.join(dir_evaluation,f'eval_{current_time}', 'Precision-Recall Curve.pdf')
        cls_report_bar = os.path.join(dir_evaluation,f'eval_{current_time}', 'Bar Graph of Classification Report.pdf')
        cls_report = os.path.join(dir_evaluation,f'eval_{current_time}', 'Classification Report.pdf')

        metrics = classification_report(self.test_y, self.result_prediction, output_dict= True, target_names=['Normal', 'Malicious'])

        precision = metrics['Normal']['precision'], metrics['Malicious']['precision']
        recall = metrics['Normal']['recall'],metrics['Malicious']['recall']
        f1 = metrics['Normal']['f1-score'],metrics['Malicious']['f1-score']
        support = metrics['Normal']['support'],metrics['Malicious']['support'] 
        accuracy = metrics['accuracy']
        macro_avg = metrics['macro avg']['precision'], metrics['macro avg']['recall'], metrics['macro avg']['f1-score'], metrics['macro avg']['support'] 
        weighted_avg = metrics['weighted avg']['precision'], metrics['weighted avg']['recall'], metrics['weighted avg']['f1-score'], metrics['weighted avg']['support'] 
        
        data = [[' ', 'Precision', 'Recall', 'F-1 Score', 'Support'],
                [' ', ' ', ' ', ' ', ' '],
                ['Normal', '{:.2f}'.format(precision[0]), '{:.2f}'.format(recall[0]), '{:.2f}'.format(f1[0]), '{:.2f}'.format(support[0])],
                ['Malicious', '{:.2f}'.format(precision[1]), '{:.2f}'.format(recall[1]), '{:.2f}'.format(f1[1]), '{:.2f}'.format(support[1])],
                [' ', ' ', ' ', ' ', ' '],
                ['Accuracy', ' ', ' ', ' ', '{:.2f}'.format(accuracy)],
                ['Macro Avg', '{:.2f}'.format(macro_avg[0]), '{:.2f}'.format(macro_avg[1]), '{:.2f}'.format(macro_avg[2]), '{:.2f}'.format(macro_avg[3])], 
                ['Weighted Avg', '{:.2f}'.format(weighted_avg[0]), '{:.2f}'.format(weighted_avg[1]), '{:.2f}'.format(weighted_avg[2]), '{:.2f}'.format(weighted_avg[3])]]
        fig, ax = plt.subplots()

        table = plt.table(cellText=data, loc='center', cellLoc='right')

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1) 

        for cell in table._cells:
            table._cells[cell].set_edgecolor('None')

        ax.axis('off')

        table_title = "Classification Report"
        plt.title(table_title, fontsize=12, fontweight='bold')
        plt.savefig(cls_report)
        plt.close()

        categories = ['Normal', 'Malicious']
        bar_width = 0.25
        r1 = [0, 1]
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]
        
        plt.figure(figsize=(10,5)) 
        plt.bar(r1, precision , color='b', width=bar_width, edgecolor='white', label='Precision')
        plt.bar(r2, recall, color='g', width=bar_width, edgecolor='white', label='Recall')
        plt.bar(r3, f1, color='r', width=bar_width, edgecolor='white', label='F1-Score')
        plt.xlabel('Categories')
        plt.ylabel('Values')
        plt.xticks([r + bar_width for r in range(len(categories))], categories)
        plt.legend()

        table_title = "Chart of Classification Report"
        plt.title(table_title, fontsize=12, fontweight='bold')
        plt.savefig(cls_report_bar)
        plt.close()

        precision, recall, _ = precision_recall_curve(self.test_y, self.result_prediction)
        auc_pr = average_precision_score(self.test_y, self.result_prediction)

        f1_scores = 2 * (precision * recall) / (precision + recall)
        optimal_idx = np.argmax(f1_scores)
        optimal_precision = precision[optimal_idx]
        optimal_recall = recall[optimal_idx]

        plt.figure(figsize=(10, 5))
        plt.plot(recall, precision, color='red', label='Precision-Recall Curve (AP = %0.2f)' % auc_pr)
        plt.scatter(optimal_recall, optimal_precision, color='blue', label='Optimal Operating Point', s=100)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()

        table_title = "Precision and Recall Curve"
        plt.title(table_title, fontsize=12, fontweight='bold')
        plt.savefig(pr_curve)
        plt.close()

        pdf_files = [cls_report, cls_report_bar, pr_curve]
        merged_pdf = PyPDF2.PdfMerger()
        for pdf_file in pdf_files:
            merged_pdf.append(pdf_file)
        output_file = os.path.join(dir_evaluation,f'eval_{current_time}', 'Evaluation of model.pdf')
        with open(output_file, 'wb') as file:
            merged_pdf.write(file)
            
        for pdf in pdf_files:
            os.remove(pdf)

    def save_models(self, dir):
        print('\n==================Save Trained Model==================\n')
        try:
            self.Encoder_1.save(dir + '/Encoder1')
            self.Encoder_2.save(dir + '/Encoder2')                
            pickle.dump(self.model_ids, open(dir + '/model_ids.pkl', 'wb'))
        except Exception as e: 
            print("Sorry, models not saved. Exception" + str(e))
            return
        time_update_model = open(dir + '/time_to_update_model', 'w')
        time_update_model.close()

            
        
