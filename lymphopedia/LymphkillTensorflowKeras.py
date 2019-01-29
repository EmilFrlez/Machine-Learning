#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

# load dataset

dataframe = pd.read_csv("/content/drive/My Drive/cancer_data.csv", delim_whitespace=True, header=None) 
dataset = dataframe.values


# In[ ]:





# In[ ]:





# In[ ]:





# # Regresion of Lymphocyte Count  Using Keras and TensorFlow
# 
# We will create a neural network model with Keras. We will also use scikit-learn to evaluate models using cross-validation. Finally, we will tune the network topology of models with Keras. My backend will be TensorFlow.

# ## Visualizing Dataset

# ### Attributes
# 
#  0. AOMAX: Aorta maximum dose (Gy)
#  1. AOMEA: Aorta mean dose (Gy)
#  2. AOVOL: Aorta total volume (cm^3)
#  3. AOINT: Aorta integral dose (Gy)
#  4. AOV5:  Aorta V5 (cm^3)
#  5. AOV10: Aorta V10 (cm^3)
#  6. AOV15: Aorta V15 (cm^3)
#  7. AOV20: Aorta V20 (cm^3)
#  8. VO: Partial tumor volume (cm^3) 
#  9. LA0:  Pre-treatment, initial LYA count (cells/l/10^9)
#  10. DAYS:  Days elapsed since start of the treatment.
#  11. BEAM:  Beam energy index
#  12. AGE:   Patient age (years)
#  13. LYA: Final LYA count (cells/l/10^9)
# 

# In[ ]:


dataframe.head()


# ### As you can see, with higher aorta integral dose, lymphocyte loss increases. What else could we predict with Neural Networks using Keras and TensorFlow?

# In[ ]:


import seaborn as sns

new_df = pd.concat([dataframe[3], dataframe[9]-dataframe[13]], axis=1)
new_df.columns = ['Aorta Integral Dose (Gy)', 'Lymphocyte Decrease/10^9/l']
sns.jointplot(x='Lymphocyte Decrease/10^9/l', y='Aorta Integral Dose (Gy)', data=new_df)


# ## Develop a Baseline Neural Network Model

# In[ ]:


# Regression Example With Lymphocyte Dataset: Baseline
import numpy
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# load dataset
dataframe = read_csv("/content/drive/My Drive/cancer_data.csv", delim_whitespace=True, header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:13]
Y = dataset[:,13]
# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# evaluate model
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))

#import seaborn as sns
#Z=baseline_model().predict(X)
#df1 = pd.concat( [Z, Y], axis=1)
#df1.columns = ['Measured', 'Predicted']
#sns.joinplot(x='Predicted', y='Measured', data=df1)


# In[ ]:





# In[ ]:





# In[ ]:





# ## Lift Performance By Standardizing the Dataset
# 
# Knowing that each attribute in the Lymphocyte dataset measures in different scales, we could standardize everything to get better results. We are going to use scikit-learn's Pipeline feature within each fold of the cross-validation.

# In[ ]:


# Regression Example With Lymphocyte Dataset: Standardized
import numpy
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# load dataset
dataframe = read_csv("/content/drive/My Drive/cancer_data.csv", delim_whitespace=True, header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:13]
Y = dataset[:,13]
# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# ## Tuning the Neural Network Topology
# Now that we have utilized k-folds for cross-validiation and standardization of datasets, let's work on expanding our Neural Network.
# 
# ### Making our Neural Network even Deeper
# Here we add more layers and neurons in each layer of our neural network.

# In[ ]:


# Regression Example With Lymphocyte Dataset: Standardized and Larger
import numpy
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# load dataset
dataframe = read_csv("/content/drive/My Drive/cancer_data.csv", delim_whitespace=True, header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:13]
Y = dataset[:,13]
# define the model
def larger_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# ### Making our Neural Network even Wider
# Now, we roughly double our neuron count from 13 to 20 inputs.

# In[ ]:


# Regression Example With Lymphocyte Dataset: Standardized and Wider
import numpy
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# load dataset
dataframe = read_csv("/content/drive/My Drive/cancer_data.csv", delim_whitespace=True, header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:13]
Y = dataset[:,13]
# define wider model
def wider_model():
	# create model
	model = Sequential()
	model.add(Dense(20, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# evaluate model with standardized dataset
#estimators = []
#estimators.append(('standardize', StandardScaler()))
#estimators.append(('mlp', KerasRegressor(build_fn=wider_model, epochs=100, batch_size=5, verbose=0)))
#pipeline = Pipeline(estimators)
#kfold = KFold(n_splits=10, random_state=seed)
#results = cross_val_score(pipeline, X, Y, cv=kfold)
#print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# extract predictions
estimator = KerasRegressor(build_fn=wider_model, epochs=100, batch_size=5, verbose=0)
estimator.fit(X, Y)
predictions = estimator.predict(X)

train_error =  numpy.abs(Y - predictions)
mean_error = numpy.mean(train_error)
min_error = numpy.min(train_error)
max_error = numpy.max(train_error)
std_error = numpy.std(train_error)

import matplotlib.pyplot as plt

plt.scatter(Y, predictions)
plt.xlabel('True LYA Decrease [cells/]/1E9')
plt.ylabel('Predicted LYA Decrease [cells/l/1E9]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-2, 2], [-2, 2])

# load new patient, change age 77->17 years
dataframeNew = read_csv("/content/drive/My Drive/new-patient.csv", delim_whitespace=True, header=None)
dataset = dataframeNew.values
XNew = dataset[:,0:13]
print(estimator.predict(XNew))


# ## Conclusion
# 
# Our baseline model using Keras, TensorFlow, and SciKit-Learn scored: -0.41  (0.21) MSE. However, with standardizing our dataset, tuning our neural network by making our neural network deeper and wider, we were able to score a final  -0.02 (0.02) MSE. That's over a $10,000 improvement in mean-squared-error!

# 
