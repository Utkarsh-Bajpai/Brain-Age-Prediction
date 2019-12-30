#! /bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ytrain = pd.read_csv('../y_train.csv', index_col='id', dtype={'id':np.int32})

plt.hist(ytrain, bins=200)
plt.title('Age Histogram')
plt.savefig('train_age_hist.png')
plt.show()
