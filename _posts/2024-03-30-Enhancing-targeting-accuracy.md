---
layout: post
title: Credit Card Fraud Detection Using ML
image: "/posts/fraud-EMV-chip-credit-card.png"
tags: [Fraud Detection, Machine Learning, Classification, Python]
---

# Project Overview  <a name="overview-main"></a>

<br>
### Context <a name="overview-context"></a>

Welcome to the Credit Card Fraud Detection Project! 

Credit card fraud poses a major threat to both financial institutions and consumers. As online transactions become more prevalent, identifying fraudulent activities has become more complex. 

Here, I created a fraud detection system using Python and widely-used machine learning libraries like scikit-learn. The primary goal was to build a robust fraud detection system using Random Forest, Logistic Regression, and Decision Tree classifiers. This project also addresses the class imbalance problem inherent in credit card fraud datasets.

<br>
### Project Structure <a name="Project-Structure"></a>

creditcard.csv: The dataset containing transaction details and fraud labels (The dataset used in this project is available on Kaggle).

Credit Card Fraud Detection.py: The main script containing data preprocessing, model training, evaluation, and model saving.

credit_card_model.p: The serialized Random Forest model saved using pickle.

<br>
### Actions <a name="overview-actions"></a>

I first needed to compile the necessary data from tables in the database.

As I was predicting a binary output, I tested three classification modeling approaches, namely:

* Logistic Regression
* Decision Tree
* Random Forest

I imported the data, defined models using a dictionary, trained & tested each model, and then measured this predictive performance based on several metrics to give a well-rounded overview of which was the best.
I also used undersampling and oversampling techniques to address the class imbalance problem inherent in credit card fraud datasets.

<br>
### Results <a name="overview-results"></a>

Here, I aimed to build a model that would accurately predict the fraud transaction.
Based upon these, the chosen model is the Random Forest as it was the most consistently performant on the test set across classification accuracy, precision, recall, and f1-score. 

<br>
**Metric 1: Classification Accuracy**

* Random Forest = 0.99
* Decision Tree = 0.99
* Logistic Regression = 0.99
  
<br>
**Metric 2: Precision**

* Random Forest = 0.90
* Logistic Regression = 0.89
* Decision Tree = 0.66

<br>
**Metric 3: Recall**

* Random Forest = 0.76
* Decision Tree = 0.75
* Logistic Regression = 0.6

<br>
**Metric 4: F1 Score**

* Random Forest = 0.82
* Logistic Regression = 0.72
* Decision Tree = 0.7

<br>
### Handling Class Imbalance

The project addresses class imbalance using:

* Undersampling: Reducing the majority class to balance the dataset.
* Oversampling: Using SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic samples for the minority class.

I implemented both undersampling and oversampling techniques, but oversampling appears to be more favorable as it ensures no data is lost.


___

<br>
### Modelling Overview  <a name="modelling-overview"></a>

I built a model that looked to accurately predict fraud transaction.

As I was predicting a binary output, I tested three classification modeling approaches, namely:

* Logistic Regression
* Decision Tree
* Random Forest

<br>
### Import Required Packagess

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
```

<br>
### Get the Data

Read the 'Creditcard.csv file into a dataframe.


```python
data = pd.read_csv("creditcard.csv")
```

<br>
### Data Preprocessing


```python
# Check data head
data.head()

# Let's see all the columns (instead of seeing ... for a couple of them)
pd.options.display.max_columns = None
data.head()

# Check data tail
data.tail()

# Check data shape
data.shape
print(f'Number of rows: {data.shape[0]}')
print(f'Number of columns: {data.shape[1]}')

# Check data type and missing values
data.info()
data.isna().sum()

# Standardisation
scale_standard = StandardScaler()

# Convert the Result from Array to DataFrame
data['Amount'] = scale_standard.fit_transform(pd.DataFrame(data['Amount']))         

# Drop Time column
data.drop('Time', axis = 1, inplace = True)

# Find duplicated data
data.duplicated().any()

# Drop duplicated data
data = data.drop_duplicates()
data.shape
```

<br>
I also investigated the class balance of the dependent variable - which is important when assessing classification accuracy.


```python
# Class balance
data['Class'].value_counts()   
```

<br>
From the last step in the above code, I saw that **0.2% of the data were in class 1 and the rest of them were in class 0**. This showed me a clear class imbalance. I made sure to not rely on classification accuracy alone when assessing results - also analyzing Precision, Recall, and F1-Score.

<br>
### Create Input and Output Variables


```python
X = data.drop(['Class'], axis = 1)
y = data['Class']
```

<br>
### Train Test Split

Train_test_split was used to split our data into a training set and a testing set.


```python
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.2, random_state = 42)
```

<br>
### Model Training and Assessment

Three classifiers were trained and evaluated:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

Each model's performance was evaluated using the following metrics:

* Accuracy
* Precision
* Recall
* F1-Score

```python
classifier = {
       'LogisticRegression': LogisticRegression(random_state = 42),
       'DecisionTreeClassifier': DecisionTreeClassifier(random_state = 42),
       'RandomForestClassifier': RandomForestClassifier(random_state = 42)}

for name, clf in classifier.items():
    
    print(f'\n==========={name}==========')
    
    clf.fit(X_train, y_train)
    y_pred_class = clf.predict(X_test)

    # Accuracy (the number of correct classification out of all attempted classifications)
    accuracy = accuracy_score(y_test, y_pred_class) 
    print(f'\n Accuracy: {accuracy}')
    
    # Precision (of all observations that were predicted as positive, how many were actually positive)
    precision = precision_score(y_test, y_pred_class)
    print(f'\n Precision: {precision}')

    # Recall (of all positive observations, how many did we predict as positive)
    recall = recall_score(y_test, y_pred_class) 
    print(f'\n Recall: {recall}')

    # F1-Score (the harmonic mean of precision and recall)
    f1 = f1_score(y_test, y_pred_class)
    print(f'\n F1_score: {f1}')
```

<br>
### Handling Class Imbalance

The dataset includes two classes of data: class 1 represents fraudulent transactions, while class 0 represents normal transactions:

```python
normal = data[data['Class']==0]
fraud = data[data['Class']==1]

normal.shape
fraud.shape
```

<br>
#### Undersampling

```python
normal.sample = normal.sample(n=473)

# Create new undersampled dataset
new_data  = pd.concat([normal.sample, fraud], ignore_index=True)
new_data.head()
new_data['Class'].value_counts()

X = new_data.drop(['Class'], axis = 1)
y = new_data['Class']

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.2, random_state = 42)

classifier = {
       'LogisticRegression': LogisticRegression(random_state = 42),
       'DecisionTreeClassifier': DecisionTreeClassifier(random_state = 42),
       'RandomForestClassifier': RandomForestClassifier(random_state = 42)}               

for name, clf in classifier.items():
    print(f'\n==========={name}==========')
    clf.fit(X_train, y_train)
    y_pred_class = clf.predict(X_test)

    # Accuracy (the number of correct classification out of all attempted classifications)
    accuracy = accuracy_score(y_test, y_pred_class) 
    print(f'\n Accuracy: {accuracy}')
    
    # Precision (of all observations that were predicted as positive, how many were actually positive)
    precision = precision_score(y_test, y_pred_class)
    print(f'\n Precision: {precision}')

    # Recall (of all positive observations, how many did we predict as positive)
    recall = recall_score(y_test, y_pred_class) 
    print(f'\n Recall: {recall}')

    # F1-Score (the harmonic mean of precision and recall)
    f1 = f1_score(y_test, y_pred_class)
    print(f'\n F1_score: {f1}')
```

<br>
#### Oversampling

```python
X = data.drop(['Class'], axis = 1)
y = data['Class']

X.shape
y.shape

X_res, y_res = SMOTE().fit_resample(X,y)

y_res.value_counts()

X_train, X_test, y_train, y_test = train_test_split (X_res, y_res, test_size = 0.2, random_state = 42)

classifier = {
       'LogisticRegression': LogisticRegression(random_state = 42),
       'DecisionTreeClassifier': DecisionTreeClassifier(random_state = 42),
       'RandomForestClassifier': RandomForestClassifier(random_state = 42)}              

for name, clf in classifier.items():
    print(f'\n==========={name}==========')
    clf.fit(X_train, y_train)
    y_pred_class = clf.predict(X_test)

    # Accuracy (the number of correct classification out of all attempted classifications)
    accuracy = accuracy_score(y_test, y_pred_class) 
    print(f'\n Accuracy: {accuracy}')
    
    # Precision (of all observations that were predicted as positive, how many were actually positive)
    precision = precision_score(y_test, y_pred_class)
    print(f'\n Precision: {precision}')

    # Recall (of all positive observations, how many did we predict as positive)
    recall = recall_score(y_test, y_pred_class) 
    print(f'\n Recall: {recall}')

    # F1-Score (the harmonic mean of precision and recall)
    f1 = f1_score(y_test, y_pred_class)
    print(f'\n F1_score: {f1}')
```
Running this code resulted in:

<br>
**Metric 1: Classification Accuracy**

* Random Forest = 0.99
* Decision Tree = 0.99
* Logistic Regression = 0.95

<br>
**Metric 2: Precision**

* Random Forest = 0.99
* Decision Tree = 0.99
* Logistic Regression = 0.97


<br>
**Metric 3: Recall**

* Random Forest = 1
* Decision Tree = 0.99
* Logistic Regression = 0.92

<br>
**Metric 4: F1 Score**

* Random Forest = 0.99
* Decision Tree = 0.99
* Logistic Regression = 0.94

<br>

These are all higher than what we saw without resampling! Random forest and decision tree offers very smilar metrics.

<br>
### Save Model

```python
rfc = RandomForestClassifier()
rfc.fit(X_res, y_res)

pickle.dump(rfc, open('Credit Card Fraud Detection/credit_card_model.p', 'wb'))   
```

<br>
### Fraud Detection (Prediction)

An example prediction is provided using the saved model to classify a new transaction.

```python
model = pickle.load(open('Credit Card Fraud Detection/credit_card_model.p', 'rb'))         
                 
prediction = model.predict([[-1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,-0.338320769942518,0.462387777762292,0.239598554061257,0.0986979012610507,0.363786969611213,0.0907941719789316,-0.551599533260813,-0.617800855762348,-0.991389847235408,-0.311169353699879,1.46817697209427,-0.470400525259478,0.207971241929242,0.0257905801985591,0.403992960255733,0.251412098239705,-0.018306777944153,0.277837575558899,-0.110473910188767,0.0669280749146731,0.128539358273528,-0.189114843888824,0.133558376740387,-0.0210530534538215,149.62]])                                                                     
prediction[0]     

if prediction[0]==0:
    print ('Normal Transaction')
else:
    print ('Fraud Transaction')
```
___

<br>
# Growth & Next Steps <a name="growth-next-steps"></a>

We could look to tune the hyperparameters of the Random Forest, notably regularisation parameters such as tree depth, as well as potentially training on a higher number of Decision Trees in the Random Forest.

