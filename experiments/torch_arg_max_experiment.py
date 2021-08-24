import torch
import numpy as np

if __name__ == '__main__':
    values = torch.tensor([[[0.2, 0.3, 0.44],
                           [0.15, 0.01, 0.09],
                           [0.32, 0.66, 0.11]]], dtype=torch.float32)

    tmax = values.max(-1, keepdim=True)[0]

    mask = values.ge(tmax).int()
    mask = mask.to(torch.float32)
    # mask.dtype = torch.float32
    print(mask)
    exit(0)
    print(values)
    argMax= torch.zeros_like(values, dtype=torch.float32)
    for value in values:
        print(torch.max)
    print(argMax)
