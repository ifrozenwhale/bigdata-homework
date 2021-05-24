import numpy as np
import pandas as pd

P = np.array([[1, 2, 3], [2, 3, 4]])
Q = np.array([[0, 2, 3], [2, 3, 4]])
P, Q = -P, P
print(P, Q)
