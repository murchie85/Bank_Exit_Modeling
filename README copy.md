# 客户银行出口建模

![Bank Customers](https://gss0.baidu.com/94o3dSag_xI4khGko9WTAnF6hhy/zhidao/pic/item/c8ea15ce36d3d539ee64cf8d3787e950352ab027.jpg "Bank Customers")

## 使用Keras和Tensorflow构建神经网络

## 概观 

- Keras是Tensorflow的扩展程序，它提供了良好的错误输出消息。
- 该网络定义如下:
- 顺序神经网络
- 该代码库使用“密集”功能（classifier.add（dense ..）
- ADAM超参数 (Hyper Parameters) 可降低损失函数
- 因为我们才有两个输出值所以我们可以使用 `Binary_crossentropy` 
- 对于两个或多个值，请使用 `categorical_crossentropy`

## 网络汇总

1. 11 `输入参数`  
2. 6 神经元 `L1`  
3. 6 神经元 `L2`   
4. 1 `输出`  



`pd.get_dummies` 将文本字段转换为数字。例如

> `Spain`   = 1 0 0   
> `Germany` = 0 1 0  
> `France`  = 0 0 1

`drop_first=True` 表示删除了第一个字段，即SPAIN. .剩余的数据可以推断出该删除的字段



This repo is forked from `krishnaik06` and builds upon the code base, to help predict customer churn. 


```
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

# Feature Scaling STANDARD NORMAL DISTRIBUTION TO MAKE SURE ALL VALUES RANGE FROM -1 TO 1
from sklearn.preprocessing import StandardScaler

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score

```
  

```
#----------------------------------------------------------------------------------------------
#                                               DATA PRE PROCESSING
#----------------------------------------------------------------------------------------------
# Importing the dataset
dataset = pd.read_csv('BankCustomers.csv')
X = dataset.iloc[:, 3:13] # FULL TABLE 
y = dataset.iloc[:, 13]   # ONLY OUTPUT EXIT VALUES

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
print(states.head())
print('note France is deleted')
```
 
## OUTPUT 

```
Printing table preview
   CreditScore Geography  Gender  Age  Tenure    Balance  NumOfProducts  \
0          619    France  Female   42       2       0.00              1   
1          608     Spain  Female   41       1   83807.86              1   
2          502    France  Female   42       8  159660.80              3   
3          699    France  Female   39       1       0.00              2   
4          850     Spain  Female   43       2  125510.82              1   

   HasCrCard  IsActiveMember  EstimatedSalary  
0          1               1        101348.88  
1          0               1        112542.58  
2          1               0        113931.57  
3          0               0         93826.63  
4          1               1         79084.10  

Number of customers are:
10000
```


```
# Splitting the dataset into the Training set and Test set
# X = FULL TABLE 
# y = EXIT Values
feature_train, feature_test, label_train, label_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling STANDARD NORMAL DISTRIBUTION TO MAKE SURE ALL VALUES RANGE FROM -1 TO 1
sc = StandardScaler()
feature_train = sc.fit_transform(feature_train)
feature_test = sc.transform(feature_test)
```
  
```
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
```

```
#----------------------------------------------------------------------------------------------
#                                    ACCURACY AND CONFUSION MATRIX
#----------------------------------------------------------------------------------------------

# Predicting the Test set results
label_pred = classifier.predict(feature_test)
label_pred = (label_pred > 0.5) # FALSE/TRUE depending on above or below 50%

cm = confusion_matrix(label_test, label_pred)  
accuracy=accuracy_score(label_test,label_pred)
```

```
print(classifier.summary())
```

## OUTPUT 

```
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_4 (Dense)              (None, 6)                 72        
_________________________________________________________________
dense_5 (Dense)              (None, 6)                 42        
_________________________________________________________________
dense_6 (Dense)              (None, 1)                 7         
=================================================================
Total params: 121
Trainable params: 121
Non-trainable params: 0
_________________________________________________________________
None
```

# Confusion Matrix and Accuracy

```
print(cm)
print(accuracy)
```


## ACKNOWLEDGEMENTS 

This repo is forked from `krishnaik06` and builds upon the code base, to help predict customer churn. Krishnaik is a fantastic DataScientist and educator, subscribe to his youtube channel [here](https://www.youtube.com/channel/UCNU_lfiiWBdtULKOw6X0Dig) for a full breakdown on creating this model.

**note** my repo will have deviated quite a bit from original source code, and end goal is to produce full matrix of customers who actually are predicted to leave. Then perform post validation using validation to keep inference decoupled. By time a new dataset is used, will migrate code to standalone repo.


