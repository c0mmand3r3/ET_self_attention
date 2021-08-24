import os

from ET_self_attention.utils.pickleutils import read_pickle_data, write_pickle_data
import numpy as np
import pandas as pd

if __name__ == '__main__':
    writer_main_path = os.path.join('data', 'vector')
    train_file_path = 'C:\\Users\\Anish\PycharmProjects\\twitter_covid19\\data\\fold_train_test_dataset_vectors' \
                      '\\set_1\\train'

    test_file_path = 'C:\\Users\\Anish\PycharmProjects\\twitter_covid19\\data\\fold_train_test_dataset_vectors\\set_1' \
                     '\\test'

    train_csv_path = 'C:\\Users\\Anish\PycharmProjects\\twitter_covid19\\data\\fold_train_test_dataset_vectors\\set_1' \
                     '\\train.csv'

    test_csv_path = 'C:\\Users\\Anish\PycharmProjects\\twitter_covid19\\data\\fold_train_test_dataset_vectors\\set_1' \
                    '\\test.csv'

    train_data = pd.read_csv(train_csv_path)
    test_data = pd.read_csv(test_csv_path)

    train_X = []
    train_Y = []

    test_X = []
    test_Y = []

    desire_dim = (10, 18)

    print(' -------- Initializing loading training data ---------')
    for train_index, t_label in enumerate(train_data['Label']):
        vector = read_pickle_data(train_file_path+'\\'+str(train_data['File_Key'][train_index]) + '.pkl')
        vector_shape = np.shape(vector)
        vector_main = np.zeros(desire_dim, dtype=float)
        if vector_shape[0] > desire_dim[0]:
            # if vector row is greater in axis
            vector_main[0:desire_dim[0], 0:vector_shape[1]] = vector[0:desire_dim[0], :]

        else:
            # vector row is smaller
            vector_main[0:vector_shape[0], 0:vector_shape[1]] = vector[:, :]
        train_X.append(vector_main)
        train_Y.append(float(t_label))
    print(' -------- Successfully Completed loading training data ---------')

    print(' -------- Initializing loading testing data ---------')
    for test_index, tt_label in enumerate(test_data['Label']):
        vector = read_pickle_data(test_file_path+'\\'+str(test_data['File_Key'][test_index]) + '.pkl')
        vector_shape = np.shape(vector)
        vector_main = np.zeros(desire_dim, dtype=float)
        if vector_shape[0] > desire_dim[0]:
            # if vector row is greater in axis
            vector_main[0:desire_dim[0], 0:vector_shape[1]] = vector[0:desire_dim[0], :]

        else:
            # vector row is smaller
            vector_main[0:vector_shape[0], 0:vector_shape[1]] = vector[:, :]
        test_X.append(vector_main)
        test_Y.append(float(tt_label))
    print(' -------- Successfully Completed loading testing data ---------')

    print(np.shape(train_X))
    print(np.shape(train_Y))
    print(np.shape(test_X))
    print(np.shape(test_Y))

    write_pickle_data(os.path.join(writer_main_path, 'train_x.pkl'), train_X)
    write_pickle_data(os.path.join(writer_main_path, 'train_y.pkl'), train_Y)
    write_pickle_data(os.path.join(writer_main_path, 'test_x.pkl'), test_X)
    write_pickle_data(os.path.join(writer_main_path, 'test_y.pkl'), test_Y)
    print('{} - Successfully created')