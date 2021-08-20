import os
from functools import partial
from ET_self_attention.dataset import ReverseDataset
from ET_self_attention.predictor import ReversePredictor_secondary
import torch
import torch.utils.data as data
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

    pretrained_filename = os.path.join(CHECKPOINT_PATH, "ReverseTask.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = ReversePredictor_secondary.load_from_checkpoint(pretrained_filename)
    else:
        model = ReversePredictor_secondary(max_iters=trainer.max_epochs * len(train_loader), **kwargs)
        trainer.fit(model, train_loader, val_loader)

    val_result = trainer.test(model, test_dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
    result = {"test_acc": test_result[0]["test_acc"], "val_acc": val_result[0]["test_acc"]}

    model = model.to(device)
    return model, result


if __name__ == '__main__':
    DATASET_PATH = "data"
    pl.seed_everything(42)

    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)

    # preparing dataset
    dataset = partial(ReverseDataset, 10, 16)
    train_loader = data.DataLoader(dataset(50000), batch_size=128, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = data.DataLoader(dataset(1000), batch_size=128)
    test_loader = data.DataLoader(dataset(10000), batch_size=128)
    inp_data, labels = train_loader.dataset[0]
    print("Input data:", inp_data)
    print("Labels:    ", labels)

    # model generation and evaluation
    reverse_model, reverse_result = train_reverse(input_dim=train_loader.dataset.num_categories,
                                                  model_dim=32,
                                                  num_heads=1,
                                                  num_classes=train_loader.dataset.num_categories,
                                                  num_layers=1,
                                                  dropout=0.0,
                                                  lr=5e-4,
                                                  warmup=50)

    print("Val accuracy:  %4.2f%%" % (100.0 * reverse_result["val_acc"]))
    print("Test accuracy: %4.2f%%" % (100.0 * reverse_result["test_acc"]))
