import torch
import numpy as np

if __name__ == '__main__':
    values = torch.tensor([[[1, 0, 0],
                           [1, 0, 0],
                           [0, 1, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                           [[1, 0, 0],
                            [0, 0, 0],
                            [0, 1, 0],
                            [0, 1, 0],
                            [0, 0, 1]]
                           ], dtype=torch.float32)

    preds = torch.mean(values, 1)
    tmax = preds.max(-1, keepdim=True)[0]

    predict_ = preds.ge(tmax).int()
    predict_ = predict_.to(torch.float32)
    print()
