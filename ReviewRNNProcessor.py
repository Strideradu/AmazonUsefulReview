import json
import os
import pickle
import math
import csv
import time
from pathlib import Path
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adadelta
from keras.layers import LSTM, Embedding
from keras.utils import np_utils
from keras import backend as K
from keras import callbacks
from keras.models import load_model
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import re


class DataPreprocessor(object):
    def __init__(self):
        self.json_path = './json_raw/'
        self.chunk_path = './json_raw/Electronics_5_chunk_'
        self.csv_path = './input_sequence/text_int_seq_'
        self.review_text_list = []
        self.review_helpful_score = []
        self.V = None
        self.X_train = []
        self.Y_train = []
        self.X_test = []
        self.Y_test = []

    def file_chunking(self):
        """
            Divide the large file into smaller ones. 65536 lines per file.
            Require the 'Electronics_5.json' file existing.
        """
        i = 0  # line counter
        j = 0  # chunk counter

        with open('./Electronics_5.json', 'r') as json_raw:
            chunk_file = open(self.chunk_path + str(j) + '.json', 'w')
            while True:
                line_buffer = json_raw.readline()
                i += 1
                if i % 65536 != 0:
                    if not line_buffer:  # end of file
                        break
                    else:
                        chunk_file.write(line_buffer)
                else:
                    chunk_file.close()
                    j += 1
                    chunk_file = open(self.chunk_path + str(j) + '.json', 'w')
            chunk_file.close()
        json_raw.close()

    def load_json_raw_data(self, chunk_num):

        chunk_path = self.chunk_path + str(chunk_num) + '.json'
        with open(chunk_path, 'r') as json_chunk_file:
            for lines in json_chunk_file:
                # line_buf = json_chunk_file.readline()
                line_jsonify = json.loads(lines)  # build a dictionary
                self.review_text_list.append(str(line_jsonify['reviewText']))
                helpful_list = line_jsonify['helpful']
                # if hlpfl_list[1] == 0:
                #     hlpfl_score = 0.6
                # else:
                #     hlpfl_score = hlpfl_list[0] / hlpfl_list[1]  # smoothing
                if helpful_list[1] == 0 or helpful_list[0] == helpful_list[1] / 2:
                    helpful_score = 0  # Neutral
                elif helpful_list[0] > helpful_list[1] / 2:
                    helpful_score = 1  # helpful
                elif helpful_list[0] < helpful_list[1] / 2:
                    helpful_score = 2  # not helpful
                else:
                    helpful_score = 3  # abnormal cases
                self.review_helpful_score.append(helpful_score)
            # print(self.review_helpful_score)
            json_chunk_file.close()
        # print(self.review_text_list[6])
        # print(self.review_text_list[7])
        print('Chunk ' + str(chunk_num) + ' loaded. Label count: ')
        count = [0, 0, 0, 0]
        for item in self.review_helpful_score:
            if item == 0:
                count[0] += 1
            elif item == 1:
                count[1] += 1
            elif item == 2:
                count[2] += 1
            else:
                count[3] += 1
        print(count)

    def vocabulary_building(self):
        vocab_pkl = Path('./Review_Vocabulary.pkl')
        if vocab_pkl.is_file():
            with open('Review_Vocabulary.pkl', 'rb') as f_pkl:
                self.V = pickle.load(f_pkl)
                f_pkl.close()
        else:
            self.V = Vocabulary()
            with open('Review_Vocabulary.pkl', 'wb') as f_pkl:
                pickle.dump(self.V, f_pkl, protocol=pickle.HIGHEST_PROTOCOL)
                f_pkl.close()

        for review_text in self.review_text_list:
            # print(self.review_text_list[0][0])
            # use regular expression to split the sequence
            xx = re.findall(r"[\w']+|[.,!?;]", review_text)
            for word in xx:
                self.V.add_word(word)
        print(self.V.word_count)
        # print(self.vocabulary.vocabulary)
        with open('Review_Vocabulary.pkl', 'wb') as f_pkl:
            pickle.dump(self.V, f_pkl, protocol=pickle.HIGHEST_PROTOCOL)
            f_pkl.close()

    def matrixing_sequences(self, chunk_num):
        # Process only 1 chunk each time this function is called.
        vocab_pkl = Path('./Review_Vocabulary.pkl')
        if vocab_pkl.is_file():
            with open('Review_Vocabulary.pkl', 'rb') as f_pkl:
                self.V = pickle.load(f_pkl)
                f_pkl.close()
        else:
            print('Vocabulary pickling file does not exist. Quit.')

        with open(self.csv_path + str(chunk_num) + '.csv', 'w', newline='') as seq_csv_file:
            csv_writer = csv.writer(seq_csv_file, quoting=csv.QUOTE_MINIMAL)
            for i in range(len(self.review_text_list)):
                single_line = [self.review_helpful_score[i]]
                word_index_seq = []
                xx = re.findall(r"[\w']+|[.,!?;]", self.review_text_list[i])
                # print(xx)
                for word in xx:
                    word_index_seq.append(self.V.word_indexing(word))
                single_line += word_index_seq
                # print(single_line)
                csv_writer.writerow(single_line)
            seq_csv_file.close()

    def train_test_loading(self, training_chunk_num_list, test_chunk_num):
        for chk_num in training_chunk_num_list:
            with open(self.csv_path + str(chk_num) + '.csv', 'r', newline='') as csv_raw_f:
                csv_reader = csv.reader(csv_raw_f)
                total_list = list(csv_reader)
                csv_raw_f.close()
            for lines in total_list:
                self.X_train.append(lines[1:])
                self.Y_train.append(lines[0])  # truth label

        with open(self.csv_path + str(test_chunk_num) + '.csv', 'r', newline='') as csv_raw_f:
            csv_reader = csv.reader(csv_raw_f)
            total_list = list(csv_reader)
            csv_raw_f.close()
            for lines in total_list:
                self.X_test.append(lines[1:])
                self.Y_test.append(lines[0])  # truth label


class Vocabulary(object):

    def __init__(self):
        self.vocabulary = {}
        self.word_count = 0

    def add_word(self, word):
        if word not in self.vocabulary:
            self.vocabulary[word] = self.word_count
            self.word_count += 1

    def word_indexing(self, word):
        try:
            return self.vocabulary[word]
        except KeyError:
            print('Word not in vocabulary set, build the dictionary again.')


class ReviewRNNModel(object):
    def __init__(self, start_time, model_h5_name):
        self.model = None
        self.model_h5_name = model_h5_name
        self.hidden_units = 256
        self.batch_size = 64
        self.nb_epoch = 2
        self.nb_cls = 3
        self.max_len = 200  # remove the sequence after 500 words
        self.max_features = 700000  # vocabulary size
        self.start_time_stamp = time.strftime(
            '%Y%m%d_%H%M%S', time.localtime(start_time))
        K.set_image_dim_ordering('th')

    def network(self):

        model_h5_file = Path('./' + self.model_h5_name)
        if model_h5_file.is_file():
            self.model = load_model('./' + self.model_h5_name)
        else:
            self.model = Sequential()
            self.model.add(Embedding(self.max_features, self.hidden_units))
            self.model.add(LSTM(self.hidden_units,
                                dropout_W=0.2,
                                dropout_U=0.2
                                ))
            self.model.add(Dense(50))
            self.model.add(Activation('relu'))
            self.model.add(Dense(3))  # fully connected layer, 10 classes
            self.model.add(Activation('softmax'))
            self.model.compile(loss='categorical_crossentropy',
                               optimizer='adadelta',  # RMSprop
                               metrics=['accuracy'])
            self.model.save('./' + self.model_h5_name)

    def model_fitting(self, X_train, y_train):
        # CSV logger
        csv_path = './result_output/' + self.start_time_stamp + '_train_log.log'
        # csv_file_gen = open(csv_path, 'w')
        # csv_file_gen.close()
        csv_logger = callbacks.CSVLogger(csv_path, append=True)

        # tensorboard initialization
        tsb_path = './result_output/logs/'
        if not os.path.exists(tsb_path):
            os.makedirs(tsb_path)
        tsb = callbacks.TensorBoard(log_dir=tsb_path)

        # print(X_train[0])
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)
        print('x_train shape:', X_train.shape)
        # print(X_train[0])

        Y_train = np_utils.to_categorical(
            y_train, self.nb_cls)  # [0/1, 0/1, ..., 0/1]

        self.model.fit(X_train, Y_train,
                       batch_size=self.batch_size,
                       nb_epoch=self.nb_epoch,
                       verbose=1,
                       # validation_data=(X_test, Y_test),
                       callbacks=[csv_logger, tsb])
        self.save_model()

    def test_validate(self, X_test, y_test):
        model_h5_file = Path('./' + self.model_h5_name)
        if model_h5_file.is_file():
            self.model = load_model('./' + self.model_h5_name)

        X_test = sequence.pad_sequences(X_test, maxlen=self.max_len)
        print('x_test shape:', X_test.shape)
        Y_test = np_utils.to_categorical(y_test, self.nb_cls)

        score = self.model.evaluate(
            X_test, Y_test, verbose=0, batch_size=self.batch_size)

        print('Test score:', score[0])
        print('Test accuracy:', score[1])

    def save_model(self):
        print('Saving the model as .json and .h5 file')
        json_string = self.model.to_json()
        with open('./result_output/' + self.start_time_stamp + '_ReviewRNN_model.json', 'w') as f_json:
            f_json.write(json_string)
            f_json.close()

        self.model.save('./' + self.model_h5_name)
