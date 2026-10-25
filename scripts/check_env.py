import recbole
import torch
import numpy
import pandas

print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
print("recbole", recbole.__version__)
print("numpy", numpy.__version__)
