import numpy as np
import os
from gym_webots import Nao
nao = Nao(action_dim=6, obs_dim=22)
file_name = 'temp_data/k.npy'
if os.path.exists(file_name):
    k = np.load(file_name)
else:
    k = 0
print(k)
nao.run()
if k < 5:
    np.save(file_name, k+1)
    nao.reset()
else:
    os.remove(file_name)