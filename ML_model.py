### create a classification model for count languages ###

# import libraries
import pandas as pd
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# load MD data
lan_5 = sio.loadmat('means.mat')
lan_5 = lan_5['MEAN_nMD_pre']
Y = np.ones((lan_5.shape[1],1))*5

lan_4 = sio.loadmat('means.mat')
lan_4 = lan_4['MEAN_nMD_post']
Y = np.vstack((Y, np.ones((lan_4.shape[1],1))*4))

# load MD data
lan_2 = sio.loadmat('means.mat')
lan_2 = lan_2['MEAN_nMD_pre']*-0.2
Y = np.vstack((Y, np.ones((lan_2.shape[1],1))*2))

lan_1 = sio.loadmat('means.mat')
lan_1 = lan_1['MEAN_nMD_post']*0.2
Y = np.vstack((Y, np.ones((lan_1.shape[1],1))*1))

# Train/Test Split
Y = Y.squeeze()
X = np.hstack((lan_5, lan_4, lan_2, lan_1)).T
X = X[:,np.where(np.isnan(X).sum(axis=0) == 0)[0]]

# Model Creation
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
model = SVC()
_ = model.fit(x_train, y_train)


# Model Application¶
y_pred = model.predict(x_test)

# Model Evaluation
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
mean_squared_error(y_test, y_pred)
confusion_matrix(y_test, y_pred.round())

# plot the results and confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(confusion_matrix(y_test, y_pred.round()), annot=True, cmap='Greens', cbar=False)
plt.show()