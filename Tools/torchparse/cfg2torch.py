import torch
import torch.nn as nn
import torch.nn.functional as F
from .parser import parse_cfg

with open('setup.cfg', 'r') as f:
    cfg_model = f.read()

# 这和前面的try ... finally是一样的，但是代码更佳简洁，并且不必调用f.close()方法

a = parse_cfg(cfg_model)
print(a)
