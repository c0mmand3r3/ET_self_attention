import os

import numpy as np
import torch
import torch.utils.data as data
from sklearn import preprocessing
from tensorflow.python.keras.utils.np_utils import to_categorical

from ET_self_attention.predictor import ReversePredictor_secondary
from ET_self_attention.utils.pickleutils import read_pickle_data

try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
    import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
CHECKPOINT_PATH = "data/saved_model"

def train_reverse(**kwargs):
    root_dir = os.path.join(CHECKPOINT_PATH, "ReverseTask")
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=1,
                         gradient_clip_val=5,
                         progress_bar_refresh_rate=1)
    trainer.logger._default_hp_metric = None

    pretrained_filename = os.path.join(CHECKPOINT_PATH, "ReverseTask_secondary.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = ReversePredictor_secondary.load_from_checkpoint(pretrained_filename)
    else:
        model = ReversePredictor_secondary(max_iters=trainer.max_epochs * len(train_loader), **kwargs)
        trainer.fit(model, train_loader, val_loader)
        print("Sucessfully fitted")
        trainer.save_checkpoint(pretrained_filename)

    val_result = trainer.test(model, test_dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
    result = {"test_acc": test_result[0]["test_acc"], "val_acc": val_result[0]["test_acc"]}

    model = model.to(device)
    return model, result

class ReverseDataset(data.Dataset):

    def __init__(self, data, labels, num_categories, seq_len, size):
        super().__init__()
        self.num_categories = num_categories
        self.seq_len = seq_len
        self.size = size

        self.data = torch.from_numpy(np.asarray(data))
        self.data.requires_grad=True

        self.labels = torch.from_numpy(np.asarray(labels))
        self.labels.requires_grad=True


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

if __name__ == '__main__':
    DATASET_PATH = "data"
    pl.seed_everything(42)

    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)
    train_x_path = os.path.join('data', 'vector', 'train_x.pkl')
    train_y_path = os.path.join('data', 'vector', 'train_y.pkl')
    test_x_path = os.path.join('data', 'vector', 'test_x.pkl')
    test_y_path = os.path.join('data', 'vector', 'test_y.pkl')

    train_x = read_pickle_data(train_x_path)
    train_y = read_pickle_data(train_y_path)
    test_x = read_pickle_data(os.path.join(test_x_path))
    test_y = read_pickle_data(os.path.join(test_y_path))
    print("---------------- Loading data completed -------------------")

    le = preprocessing.LabelEncoder()
    le.fit(train_y)

    train_y = le.transform(train_y)
    test_y = le.transform(test_y)

    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)

    train_dataset = ReverseDataset(train_x, train_y, 3, 18, len(train_x))
    train_loader = data.DataLoader(train_dataset, batch_size=18, shuffle=True, drop_last=True, pin_memory=True)
    val_dataset = ReverseDataset(test_x, test_y, 3, 18, len(test_x))

    val_loader = data.DataLoader(val_dataset, batch_size=18, shuffle=True, drop_last=True, pin_memory=True)
    test_loader = data.DataLoader(val_dataset, batch_size=18, shuffle=True, drop_last=True, pin_memory=True)

    inp_data, labels = train_loader.dataset[0]
    print("Input data:", inp_data)
    print("Labels:    ", labels)
    reverse_model, reverse_result = train_reverse(input_dim=18,
                                                  model_dim=18,
                                                  num_heads=1,
                                                  num_classes=3,
                                                  num_layers=2,
                                                  dropout=0.0,
                                                  lr=0.01,
                                                  warmup=50)


    print("Val accuracy:  %4.2f%%" % (100.0 * reverse_result["val_acc"]))
    print("Test accuracy: %4.2f%%" % (100.0 * reverse_result["test_acc"]))
