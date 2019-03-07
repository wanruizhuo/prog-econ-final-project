import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
from src.model_code.panel_model import gen_xy
from bld.project_paths import project_paths_join as ppj

data = pd.read_csv(ppj("OUT_DATA", "productivity_clean.csv"))

X,y = gen_xy()

#i = []
#X[i] = X[:, i]
X1 = X[:, 0]
X2 = X[:, 1]
X3 = X[:, 2]
X4 = X[:, 3]

#def X(i):
#    for i in len(X[i]):
#        X() = np.mean(X[i]) 
#    return X
#X_1 = np.mean(X1)
#X_2 = np.mean(X2)
#X_3 = np.mean(X3)
#X_4 = np.mean(X4)
#y_0 = np.mean(y)

fig, ax = plt.subplots()
ax.plot(X1, y, color="blue", label="LNPUB_CAP")
ax.plot(X2, y, color="red", label="LNPC")
ax.plot(X3, y, color="green", label="LNEMP")
ax.plot(X4,y, color="yellow", label="UNEMP")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()

plt.savefig(ppj("OUT_FIGURES", "Independent_Dependent_Variables.png"))