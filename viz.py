import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv(r"C:\Users\chr_1\Downloads\csv (1).csv")
y=data["Value"].array
plt.plot(np.arange(300),y)
plt.xlabel("epoch")
plt.ylabel("loss L_e")