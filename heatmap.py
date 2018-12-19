import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("confusion_matrix.txt", index_col=0)
absolute = df.values
relative = absolute / absolute.sum(axis=1)[:, None]
plt.figure(figsize=(9, 6.25))
ax = plt.imshow(relative, cmap="Wistia", aspect="auto")
plt.yticks(range(len(df.index)), df.index)
plt.xticks(range(len(df.columns)), df.columns)
plt.gca().xaxis.set_ticks_position("top")
for i in range(len(df.index)):
    for j in range(len(df.columns)):
        value = absolute[i, j]
        if value:
            plt.text(j, i, value, ha="center", va="center")
plt.colorbar()
plt.savefig("confusion_matrix.png", transparent=True)
# plt.show()
