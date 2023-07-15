import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./data/caco2_R2(2).csv')

df.plot.box(title="RMSD")

plt.grid(linestyle="--", alpha=0.3,)

plt.show()


print(df.describe())

import pandas

