import os
import pickle

import numpy as np
import torch
from sklearn import preprocessing
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.saving.save import load_model
from tensorflow.python.keras.utils.np_utils import to_categorical
from torch.utils.data import TensorDataset, DataLoader


def read_pickle_data(path=None):
    if path is None or not os.path.isfile(path):
        return None
    with open(path, 'rb') as fid:
        data = pickle.load(fid)
        fid.close()
        return data

import os
from functools import partial
from ET_self_attention.dataset import ReverseDataset
from ET_self_attention.predictor import ReversePredictor
import torch
import torch.utils.data as data
try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
    import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint




SETS = 10

if __name__ == '__main__':
    read_main_path = os.path.join('data', 'fold_train_test_dataset_overall_vectors')


    for n_set in range(SETS):
        read_joiner_path = os.path.join(read_main_path, 'set_' + str(n_set + 1))

        train_x = read_pickle_data(os.path.join(read_joiner_path, 'train_x.pkl'))
        train_y = read_pickle_data(os.path.join(read_joiner_path, 'train_y.pkl'))
        test_x = read_pickle_data(os.path.join(read_joiner_path, 'test_x.pkl'))
        test_y = read_pickle_data(os.path.join(read_joiner_path, 'test_y.pkl'))
        print("---------------- Loading Set {} completed -------------------".format(n_set + 1))

        print(np.shape(train_x), np.shape(train_y))
        print(np.shape(test_x), np.shape(test_y))

        scale_model = StandardScaler()
        scale_model.fit(train_x)

        train_x = scale_model.transform(train_x)
        test_x = scale_model.transform(test_x)

        print(np.shape(train_x), np.shape(train_y))
        print(np.shape(test_x), np.shape(test_y))

        le = preprocessing.LabelEncoder()
        le.fit(train_y)

        train_y_ = le.transform(train_y)
        test_y_ = le.transform(test_y)

        train_y = to_categorical(train_y_)
        test_y = to_categorical(test_y_)

        train_x = np.array(train_x).reshape((np.shape(train_x)[0], np.shape(train_x)[1], 1))
        test_x = np.array(test_x).reshape((np.shape(test_x)[0], np.shape(test_x)[1], 1))

        print(np.shape(train_x), np.shape(train_y))
        print(np.shape(test_x), np.shape(test_y))

        train_x = torch.Tensor(train_x)
        train_y = torch.Tensor(train_y)
        train_loader = TensorDataset(train_x, train_y)
        train_dataloader = DataLoader(train_loader)

        test_x = torch.Tensor(test_x)
        test_y = torch.Tensor(test_y)
        test_loader = TensorDataset(test_x, test_y)
        test_dataloader = DataLoader(test_loader)

        model = ReversePredictor(max_iters=10, input_dim=17, model_dim=1, num_classes=3, num_heads=1, num_layers=10, lr=0.001, warmup=1)
        trainer = pl.Trainer()
        trainer.fit(model, train_dataloader, test_dataloader)
        print()
        exit(0)

