import torch


def test(a, b, add):
    return 2 + a * b if add else 0

a = torch.tensor(1)
b = a
print(type(b))