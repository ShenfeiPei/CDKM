import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from alias_copyi_module_CDKM import CDKM
from alias_copyi_module_CDKM.Public import Ifuns, Mfuns

name = "Iris"

current_dir = os.path.dirname(__file__)
data_full_name = os.path.join(current_dir, f"data/{name}.mat")
X, y_true, N, dim, c_true = Ifuns.load_mat(data_full_name)
X = X.astype(np.float64)


MM = MinMaxScaler(feature_range=(0, 1))
X = MM.fit_transform(X)

init_Y = Ifuns.initial_Y(X, c_true, rep=10, way="random")

mod = CDKM(X, c_true, debug=0)
mod.opt(init_Y, ITER=200)
Y = mod.y_pre
n_iter = mod.n_iter_

obj = Mfuns.multi_kmeans_obj(X, Y)
print(f"cdkm: mean = {np.mean(obj)}, std = {np.std(obj)}, min = {np.min(obj)}, iter = {np.mean(n_iter)}")

# paper
# Iris: random, mean = 6.9981, std = 0, min = 6.9981, iter = 3.52
