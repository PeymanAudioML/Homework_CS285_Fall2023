import sys
print(sys.path)

import sys
print(sys.executable)

import numpy as np
print(np.__version__)

import torch
import numpy as np

np_array = np.ones(5)
torch_tensor = torch.from_numpy(np_array)
print(torch_tensor)


