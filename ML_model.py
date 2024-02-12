### create a classification model for count languages ###

# import libraries
import pandas as pd
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

# load MD data
lan_5 = sio.loadmat('MD_vectors.mat')
lan_5 = lan_5['MD_vectors_all'][:,:33]
Y = np.ones((lan_5.shape[1],1))*5

lan_4 = sio.loadmat('MD_vectors.mat')
lan_4 = lan_4['MD_vectors_all'][:,34:]
Y = np.vstack((Y, np.ones((lan_4.shape[1],1))*4))

# Train/Test Split
Y = Y.squeeze()
X = np.hstack((lan_5[0], lan_4)).T
X = pd.DataFrame(X)

# Model Creation
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = LinearRegression()
_ = model.fit(x_train, y_train)


# Model Application¶
y_pred = model.predict(x_test)





