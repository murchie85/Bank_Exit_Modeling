"""

# Building a Neural Network with Keras and Tensorflow 


OVERVIEW
--------
- Using Keras, it is a wrapper around tensor flow and is good for error description
- Keras network is defined as:
- Sequential Neural network
- Using Dense Function (classifier.add(dense..)
- ADAM reduces cost function
- `Binary_crossentropy` because we only have two output values
- For more than 2 output values use `categorical_crossentropy`

NETWORK OVERVIEW
----------------

1. 11 `inputs`  
2. 6 neurons `L1`  
3. 6 neurons `L2`   
4. 1 `ouptut`  


NOTES
------

`pd.get_dummies` converts a textual field into numbers. i.e: 

> `Spain`   = 1 0 0   
> `Germany` = 0 1 0  
> `France`  = 0 0 1

`drop_first=True` means the first  field i.e. Spain is removed. As it can be inferred by the other data i.e. when they are all zeros.

"""


#----------------------------------------------------------------------------------------------
#                                               IMPORTS
#----------------------------------------------------------------------------------------------

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
# Feature Scaling 
from sklearn.preprocessing import StandardScaler

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score

#----------------------------------------------------------------------------------------------
#                                               DATA PRE PROCESSING
#----------------------------------------------------------------------------------------------
# Importing the dataset
print('importing data...')
dataset = pd.read_csv('BankCustomers.csv')
X = dataset.iloc[:, 3:13] # FULL TABLE 
y = dataset.iloc[:, 13]   # ONLY OUTPUT EXIT VALUES
print('')

print('Printing table preview')
print(X.head())
print('')
print('Number of customers are:')
print(len(y))
print('')

# convert categorical feature into dummy variables
states=pd.get_dummies(X['Geography'],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)

#concatenate the remaining dummies columns
# NOTE EACH COLUMN IS A VALUE NOW, I.E. GERMANY, SPAIN etc
X=pd.concat([X,states,gender],axis=1)

#drop the columns as it is no longer required
X=X.drop(['Geography','Gender'],axis=1)


#dataset.iloc[:, 3:13].values
#print(states.head())
#print('note France is deleted')


#----------------------------------------------------------------------------------------------
#                                               TRAINING/TEST DATA SPLIT
#----------------------------------------------------------------------------------------------

# X = FULL TABLE 
# y = EXIT Values
feature_train, feature_test, label_train, label_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling STANDARD NORMAL DISTRIBUTION TO MAKE SURE ALL VALUES RANGE FROM -1 TO 1
sc = StandardScaler()
feature_train = sc.fit_transform(feature_train)
feature_test = sc.transform(feature_test)



#----------------------------------------------------------------------------------------------
#                                    DEFINE AND TRAIN NEURAL NET
#----------------------------------------------------------------------------------------------

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))

# Adding the second hidden layer
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

# Adding the output layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(feature_train, label_train, batch_size = 10, nb_epoch = 100)


#----------------------------------------------------------------------------------------------
#                                    ACCURACY AND CONFUSION MATRIX
#----------------------------------------------------------------------------------------------

# Predicting the Test set results
label_pred = classifier.predict(feature_test)
label_pred = (label_pred > 0.5) # FALSE/TRUE depending on above or below 50%

cm = confusion_matrix(label_test, label_pred)  
accuracy=accuracy_score(label_test,label_pred)

print('printing model summary')
print(classifier.summary())
print('')
print('Printing Confusion Matrix')
print(cm)
print('')
print('values should add up to :')
print(len(label_test))
print('')
print('Printing Accuracy')
print(accuracy)