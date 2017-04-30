from ReviewRNNProcessor import DataPreprocessor
from ReviewRNNProcessor import ReviewRNNModel
import time


def run(function_flag):
    start_time = time.time()
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
    print('Task begins. \nTime stamp: ' + current_time)

    # dpp0 = DataPreprocessor()
    # dpp0.load_data(0)
    train_chunk_list = [[0, 1, 2, 4],
                        [5, 6, 7, 9],
                        [10, 11, 12, 13],
                        [15, 16, 17, 18, 20],
                        [21, 22, 23, 25, 26]]
    test_chunk_list = [3, 8, 14, 19, 24]
    # for item in train_chunk_list[0]:

    if function_flag == 0:
        # divide the large json file into multiple small json files
        # require 'Electronics_5.json' file existing
        dpp0 = DataPreprocessor()
        dpp0.file_chunking()
    elif function_flag == 1:
        # build dictionary for indexing
        for i in range(26):
            dpp0 = DataPreprocessor()
            dpp0.load_json_raw_data(i)
            dpp0.vocabulary_building()
            del dpp0
    elif function_flag == 2:
        # sequence mapping, convert the sentences into indices matrix
        for i in range(26):
            dpp0 = DataPreprocessor()
            dpp0.load_json_raw_data(i)
            dpp0.matrixing_sequences(i)
            del dpp0
    elif function_flag == 3:
        # network fitting
        epoch = 5
        for ep in range(epoch):
            for i in range(5):
                dpp0 = DataPreprocessor()
                dpp0.train_test_loading(train_chunk_list[i], test_chunk_list[i])
                rrm0 = ReviewRNNModel(start_time, 'Review_RNN_weights.h5')
                rrm0.network()
                rrm0.model_fitting(dpp0.X_train, dpp0.Y_train)
                rrm0.test_validate(dpp0.X_test, dpp0.Y_test)

    end_time = time.time()
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))

    print(current_time)
    print('Total execution time: ' + '%.3f' % (end_time - start_time) + ' s')

if __name__ == '__main__':
    run(3)
