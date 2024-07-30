---
layout: post
title: Predicting Life Insurance Premiums for Clients
image: "/posts/premium.png"
tags: [Machine Learning, Regression, Python]
---

In the rapidly evolving financial services industry, companies strive to provide accurate and personalized offerings to their clients. An insurance agency working with a leading financial services provider, embarked on a mission to enhance their life insurance offerings by leveraging the power of machine learning. 
This initiative aimed to design a platform that would give clients access to predict premiums for a combined life insurance and investment package with greater accuracy, ensuring fair pricing and personalized plans. 

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

The agency serves a broad spectrum of clients, each with unique financial needs and health profiles. Traditionally, calculating life insurance premiums involved a complex evaluation of multiple factors, often leading to discrepancies and inefficiencies. While financial advisors have access to software for evaluating different plans and determining premiums, the agency wants to empower clients with a platform that allows them to get a rough estimate of their premiums based on their specific budget and status, as well as the potential savings.

### Actions <a name="overview-actions"></a>

To tackle this challenge, a comprehensive data-driven approach was adopted. The journey began with the collection of extensive client data, including demographic information, health metrics, and lifestyle factors. This data was then meticulously preprocessed to ensure its accuracy and completeness.

I built a predictive model to find relationships between client metrics and *life insurance premium* for previous clients, and used this to predict premiums for potential new clients.

As I was predicting a numeric output, I tested three regression modeling approaches, namely:

* Linear Regression
* Decision Tree
* Random Forest
<br>
<br>

### Results <a name="overview-results"></a>

The Random Forest had the highest predictive accuracy.

<br>
**Metric 1: R-Squared (Test Set)**

* Random Forest = 0.918
* Decision Tree = 0.908
* Linear Regression = 0.766
  
<br>
**Metric 2: Adjusted R-Squared (Test Set)**

* Random Forest = 0.915
* Decision Tree = 0.904
* Linear Regression = 0.755

<br>
**Metric 3: Cross Validated R-Squared (K-Fold Cross Validation, k = 4)**

* Random Forest = 0.881
* Decision Tree = 0.865
* Linear Regression = 0.755

As the most important outcome for this project was predictive accuracy, rather than explicitly understanding weighted drivers of prediction, I chose the Random Forest as the model to use for making predictions on the life insurance premiums for future clients.
<br>
<br>

### Key Definition  <a name="overview-definition"></a>

age: client's age
sex: client's gender
bmi: Body mass index is a value derived from the mass and height of a person (the body mass (kg) divided by the square of the body height (m^2))
children: number of client's children
smoker: client's smoking status	
region: client's place of residence
CI: includes critical illness insurance
rated: increased premium due to health problems
UL permanent: combined investment and life insurance
disability: includes disability insurance
premium: client's monthly payment

___

# Data Overview  <a name="data-overview"></a>

The initial dataset included various attributes such as age, gender, BMI, number of children, smoking status, region, and several insurance-related features. To prepare the data for modeling, several key steps were undertaken:

Handling Missing Values: Any missing values in the dataset were identified and appropriately addressed.

Dealing with Outliers: The dataset was examined for outliers to ensure the integrity of the data.

Encoding Categorical Variables: Categorical variables like gender, smoker status, and region were encoded using one-hot encoding to make them suitable for machine learning models.

Feature Scaling: Numerical features were standardized to ensure they were on a comparable scale, enhancing the model's performance.

# Model Training and Evaluation  <a name="Model Training and Evaluation"></a>

With the data prepared, the next step was to train a machine learning model capable of accurately predicting life insurance premiums.

I tested three regression modeling approaches, namely:

* Linear Regression
* Decision Tree
* Random Forest

For each model, I imported the data in the same way but needed to pre-process the data based on the requirements of each particular algorithm. I trained & tested each model, refined each to provide optimal performance, and then measured this predictive performance based on several metrics to give a well-rounded overview of which is best.

The dataset was split into training and testing sets, ensuring that the model could be evaluated on unseen data. The model was trained on the training set, and its performance was evaluated using the testing set. Key metrics of R-squared and adjusted R-squared were calculated to assess the model's accuracy.

To further refine the model, cross-validation was performed using KFold, providing a more robust evaluation by splitting the data into multiple folds and ensuring the model's consistency across different subsets of the data.

___
<br>
# Linear Regression <a name="linreg-title"></a>

I utilized the scikit-learn library within Python to model the data using Linear Regression.
```python
# Import Required Packages

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score

# Import Sample Data

data_for_model = pd.read_excel('Life insurance.xlsx', sheet_name='Life insurance')  

# Shuffle data
data_for_model = shuffle(data_for_model, random_state = 42)               

# Get Data Information

data_for_model.info()
data_for_model['region'].value_counts()

# Deal with Missing Values

data_for_model.isna().sum()     
```
##### Outliers

The ability of a Linear Regression model to generalize well across *all* data could be hampered if there were outliers present. There was no right or wrong way to deal with outliers, but it was always something worth very careful consideration - just because a value was high or low, did not necessarily mean it should not be there!
In this code section, I used **.describe()** from Pandas to investigate the spread of values for each of the predictors. 
```python
# Deal with Outliers

outlier_investigation = data_for_model.describe()
```

The results of this can be seen in the table below.

| **metric** | **age** | **bmi** | **children** | **premium** | 
|---|---|---|---|---|
| count | 1338 | 1338 | 1338 | 1338 |
| mean | 39.2 | 30.67 | 1.09 | 1104.10 |
| std | 14.05 | 6.10 | 1.21 | 1010.11 | 
| min | 18 | 16 | 0 | 93.49 | 
| 25% | 27 | 26.3 | 0 | 391.28 | 
| 50% | 39 | 30.4 | 1 | 775.27 | 
| 75% | 51 | 34.7 | 2 | 1386.66 | 
| max | 64 | 53.1 | 5 | 5314.20 | 
<br>

Based on this investigation, I saw some *max* column values for bmi was much higher than the *median* value.

Because of this, I applied some outlier removal to facilitate generalization across the full dataset.

I did this using the "boxplot approach" where I removed any rows where the values within those columns were outside of the interquartile range multiplied by 2.
```python
outlier_columns = ["bmi"]

# boxplot approach
for column in outlier_columns:
    
    lower_quartile = data_for_model[column].quantile(0.25)
    upper_quartile = data_for_model[column].quantile(0.75)
    iqr = upper_quartile - lower_quartile
    iqr_extended = iqr * 2
    min_border = lower_quartile - iqr_extended
    max_border = upper_quartile + iqr_extended
    
    outliers = data_for_model[(data_for_model[column] < min_border) | (data_for_model[column] > max_border)].index
    print(f"{len(outliers)} outliers detected in column {column}")
    
    data_for_model.drop(outliers, inplace = True)
```
<br>
##### Split Out Data For Modelling
```python
# Split Input Variables and Output Variables

X = data_for_model.drop(['premium'], axis = 1)
y = data_for_model['premium']

# Split out Training and Test Sets

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.2, random_state = 42)
```
<br>
##### Categorical Predictor Variables

In my dataset, I had seven categorical variables. As the Linear Regression algorithm could only deal with numerical data, I had to encode my categorical variables.

One Hot Encoding was used to represent categorical variables as binary vectors. 

For ease, after I applied One Hot Encoding, I turned my training and test objects back into Pandas Dataframes, with the column names applied.
```python
# Deal with Categorical Variables

# Create a list of categorical variables                    
categorical_vars = ['sex', 'smoker', 'region', 'CI', 'rated', 'UL permanent', 'disability']   

# Create and apply OneHotEncoder while removing the dummy variable
one_hot_encoder = OneHotEncoder(sparse = False, drop = 'first')               

# Apply fit_transform on training data
X_train_encoded =  one_hot_encoder.fit_transform(X_train[categorical_vars])

# Apply transform on test data
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])            

# Get feature names to see what each column in the 'encoder_vars_array' presents
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

# Convert our result from an array to a DataFrame
X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)

# Concatenate (Link together in a series or chain) new DataFrame to our original DataFrame 
X_train = pd.concat([X_train.reset_index(drop = True), X_train_encoded.reset_index(drop = True)], axis = 1)    
X_test = pd.concat([X_test.reset_index(drop = True), X_test_encoded.reset_index(drop = True)], axis = 1)    
 
# Drop the original categorical variable columns
X_train.drop(categorical_vars, axis = 1, inplace = True)           
X_test.drop(categorical_vars, axis = 1, inplace = True) 
```
<br>
##### Data Visualization
```python
# Data Distribution
data_for_model.hist(figsize=(10,8))  
```
This created the below plot, which showed where the majority of data points lied.
![alt text](/img/posts/hist.png)

```python
# Pairplot
sns.pairplot(data_for_model)
```

Following pairplot showed the correlation between bmi and age with premium.

* A positive correlation between bmi and premium suggests that clients with higher BMI might have higher premiums. This could be due to the increased health risks associated with higher bmi.

* A positive correlation between age and premium indicates that older clients tend to have higher premiums. This is expected as life insurance premiums generally increase with age due to higher risk.
![alt text](/img/posts/Pairplotp.png)

Feature selection and feature scaling did not impact the modeling process or the accuracy of the predictions. Therefore, they were excluded.
<br>
### Model Training <a name="linreg-model-training"></a>

Instantiating and training the Linear Regression model was done using the below code:

```python
# instantiate my model object
regressor = LinearRegression()

# fit my model using our training & test sets
regressor.fit(X_train, y_train)
```
<br>
### Model Performance Assessment <a name="linreg-model-assessment"></a>

##### Predict On The Test Set

To assess how well my model was predicting new data - I used the trained model object (here called *regressor*) and asked it to predict the *insurance premium* variable for the test set.

```python
# predict on the test set
y_pred = regressor.predict(X_test)
```

<br>
##### Calculate R-Squared
R-Squared is a metric that shows the percentage of variance in the output variable *y* that is being explained by the input variable(s) *x*.  It is a value that ranges between 0 and 1, with a higher value showing a higher level of explained variance.
```python
r_squared = r2_score(y_test, y_pred)
print(r_squared)
```
The resulting r-squared score from this was **0.766**.
<br>
##### Calculate adjusted R-squared
```python
num_data_points, num_input_vars = X_test.shape                           
adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars - 1)
print(adjusted_r_squared)
```
The resulting r-squared score from this was **0.755**.
<br>
##### Calculate Cross Validated R-Squared (Cross validation (KFold: including both shuffling and the random state))

An even more powerful and reliable way to assess model performance is to utilize Cross Validation.

Instead of simply dividing our data into a single training set, and a single test set, with Cross Validation we can break our data into several "chunks" and then iteratively train the model on all but one of the "chunks", test the model on the remaining "chunk" until each has had a chance to be the test set.

The result of this is that we are provided a number of test set validation results - and we can take the average of these to give a much more robust & reliable view of how our model will perform on new, unseen data!

In the code below, I put this into place. I first specified that I wanted 4 "chunks" and then I passed in my regressor object, training set, and test set. I also specified the metric I wanted to assess with, in this case, I sticked to r-squared.

Finally, I took a mean of all four test set results.
```python
cv = KFold(n_splits = 4, shuffle = True, random_state = 42)    
cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv, scoring = 'r2')     # returns r2 for each chunk of data (each cv)
cv_scores.mean()
print(cv_scores.mean())
```
The resulting r-squared score from this was **0.755**.

##### Extract model coefficients and intercept

Here, I extracted the coefficients of the linear model which showed a high positive correlation between age, bmi, smoking status and gender with the resulted premium.
```python
coefficients = pd.DataFrame(regressor.coef_)
input_variable_names = pd.DataFrame(X_train.columns)
summary_stats = pd.concat([input_variable_names, coefficients], axis = 1)
summary_stats.columns = ['input_variable', 'coefficient']

# Extract model intercept
regressor.intercept_
```









Optimal Model Selection
Determining the optimal complexity of the decision tree was crucial. By experimenting with different maximum depths for the tree, the optimal depth was identified based on the highest accuracy score. This step ensured that the model was neither too simple to capture essential patterns nor too complex to overfit the training data.

Results and Insights
The final decision tree model provided valuable insights into the factors influencing life insurance premiums. Feature importance analysis highlighted the key variables impacting premium calculations, offering transparency and interpretability to WFG's underwriting process.

Visualization and Predictions
Visualizations such as histograms, pair plots, and tree plots were created to understand the data distribution and model structure better. Additionally, the model was used to predict premiums for new clients, showcasing its practical applicability in real-world scenarios.

Impact and Future Directions
The implementation of this machine learning solution marked a significant milestone for WFG. By accurately predicting life insurance premiums, WFG was able to offer fairer and more personalized insurance plans to their clients, enhancing customer satisfaction and trust.

Looking ahead, WFG plans to continuously refine and expand this model by incorporating additional data sources and exploring more advanced machine learning techniques. This initiative represents a commitment to innovation and excellence, ensuring that WFG remains at the forefront of the financial services industry.

Through this project, WFG has demonstrated the transformative power of data and machine learning in revolutionizing traditional financial processes, paving the way for a more efficient and customer-centric future.



Growth:
It would also allow clients to play with premiums and see how different premiums could allocate money for their retirements.
