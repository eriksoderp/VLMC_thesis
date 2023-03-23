import pandas as pd
import numpy as np
import seaborn as sns
import sys
import matplotlib.pyplot as plt

df = pd.read_csv("vlmc_distances.csv")

print(df)

sns.set_theme()

sns.relplot(
    data=df,
    x='VLMC dist', y='Evolutionary dist', hue=df[['threshold', 'min_count', 'max_depth']].apply(tuple, axis=1)
)

plt.show()
