### create a classification model for count languages ###

# import libraries
import pandas as pd
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVC



# load MD data
lan_5 = sio.loadmat('vectors/MD_vectors_5_lang.mat')
lan_5 = lan_5['MEAN_nMD_vectors'][:245,:]
Y = np.ones((lan_5.shape[1],1))*5

lan_4 = sio.loadmat('vectors/MD_vectors_4_lang.mat')
lan_4 = lan_4['MEAN_nMD_vectors'][:245,:]
Y = np.vstack((Y, np.ones((lan_4.shape[1],1))*4))

# load MD data
lan_2 = sio.loadmat('vectors/MD_vectors_2_lang_temp.mat')
lan_2 = lan_2['MEAN_nMD_vectors'][:245,:]
Y = np.vstack((Y, np.ones((lan_2.shape[1],1))*2))

lan_1 = sio.loadmat('vectors/MD_vectors_1_lang.mat')
lan_1 = lan_1['MEAN_nMD_vectors'][:245,:]
Y = np.vstack((Y, np.ones((lan_1.shape[1],1))*1))

# Train/Test Split
Y = Y.squeeze()
X = np.hstack((lan_5, lan_4, lan_2, lan_1)).T
X[np.isnan(X)] = 0

# Model Creation
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = LinearRegression()
_ = model.fit(x_train, y_train)


# Model Application¶
y_pred = model.predict(x_test)

# Model Evaluation
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
mean_squared_error(y_test, y_pred)
confusion_matrix(y_test, y_pred.round())

# plot the results and confusion matrix
labels = np.unique([y_pred.round(),y_test])
labels.sort()
plt.figure(figsize=(10,5))
sns.heatmap(confusion_matrix(y_test, y_pred.round()), annot=True, cmap='Greens', cbar=False, xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')


# PCA components
N_COMPONENTS = 50
ROIs_pca = PCA(n_components=N_COMPONENTS)
_=ROIs_pca.fit(X)
N_threshold = 0.95
N_components = np.where(np.cumsum(ROIs_pca.explained_variance_ratio_)>N_threshold)[0][0]

# transform the data
X_pca = ROIs_pca.transform(X)
X_pca_threshold = X_pca[:,:N_components]
x_train, x_test, y_train, y_test = train_test_split(X_pca_threshold, Y, test_size=0.2, random_state=42)
model = LinearRegression()
_ = model.fit(x_train, y_train)

# Model Application
y_pred = model.predict(x_test)

# Model Evaluation
mean_squared_error(y_test, y_pred)
confusion_matrix(y_test, y_pred.round())

# show the first components scores
plt.figure(figsize=(10,5))
plt.plot(ROIs_pca.components_[:5].T)
plt.xlabel('ROIs')
plt.ylabel('Scores')
plt.title('PCA Components Scores')


# plot the results and confusion matrix
labels = np.unique([y_pred.round(),y_test])
labels.sort()
plt.figure(figsize=(10,5))
sns.heatmap(confusion_matrix(y_test, y_pred.round()), annot=True, cmap='Greens', cbar=False, xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()