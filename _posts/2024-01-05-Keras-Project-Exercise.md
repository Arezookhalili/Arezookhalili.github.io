---
layout: post
title: Predicting Loan Repayment- Analyzing LendingClub Data for Credit Risk Assessment
image: "/posts/loan.png"
tags: [ANN, Machine Learning, Python]
---

LendingClub is a US peer-to-peer lending company, headquartered in San Francisco, California[3]. It was the first peer-to-peer lender to register its offerings as securities with the Securities and Exchange Commission (SEC), and to offer loan trading on a secondary market. LendingClub is the world's largest peer-to-peer lending platform.

I used a subset of the LendingClub DataSet obtained from Kaggle: https://www.kaggle.com/wordsforthewise/lending-club

Given historical data on loans given out with information on whether or not the borrower defaulted (charge-off), I built a model to predict whether or not a borrower would pay back their loan? This way in the future when we get a new potential customer we can assess whether or not they are likely to pay back the loan. 

The "loan_status" column contains the label.

### Data Overview

Here is the information on this particular data set:

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LoanStatNew</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>loan_amnt</td>
      <td>The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>term</td>
      <td>The number of payments on the loan. Values are in months and can be either 36 or 60.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>int_rate</td>
      <td>Interest Rate on the loan</td>
    </tr>
    <tr>
      <th>3</th>
      <td>installment</td>
      <td>The monthly payment owed by the borrower if the loan originates.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>grade</td>
      <td>LC assigned loan grade</td>
    </tr>
    <tr>
      <th>5</th>
      <td>sub_grade</td>
      <td>LC assigned loan subgrade</td>
    </tr>
    <tr>
      <th>6</th>
      <td>emp_title</td>
      <td>The job title supplied by the Borrower when applying for the loan.*</td>
    </tr>
    <tr>
      <th>7</th>
      <td>emp_length</td>
      <td>Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>home_ownership</td>
      <td>The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER</td>
    </tr>
    <tr>
      <th>9</th>
      <td>annual_inc</td>
      <td>The self-reported annual income provided by the borrower during registration.</td>
    </tr>
    <tr>
      <th>10</th>
      <td>verification_status</td>
      <td>Indicates if income was verified by LC, not verified, or if the income source was verified</td>
    </tr>
    <tr>
      <th>11</th>
      <td>issue_d</td>
      <td>The month which the loan was funded</td>
    </tr>
    <tr>
      <th>12</th>
      <td>loan_status</td>
      <td>Current status of the loan</td>
    </tr>
    <tr>
      <th>13</th>
      <td>purpose</td>
      <td>A category provided by the borrower for the loan request.</td>
    </tr>
    <tr>
      <th>14</th>
      <td>title</td>
      <td>The loan title provided by the borrower</td>
    </tr>
    <tr>
      <th>15</th>
      <td>zip_code</td>
      <td>The first 3 numbers of the zip code provided by the borrower in the loan application.</td>
    </tr>
    <tr>
      <th>16</th>
      <td>addr_state</td>
      <td>The state provided by the borrower in the loan application</td>
    </tr>
    <tr>
      <th>17</th>
      <td>dti</td>
      <td>A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.</td>
    </tr>
    <tr>
      <th>18</th>
      <td>earliest_cr_line</td>
      <td>The month the borrower's earliest reported credit line was opened</td>
    </tr>
    <tr>
      <th>19</th>
      <td>open_acc</td>
      <td>The number of open credit lines in the borrower's credit file.</td>
    </tr>
    <tr>
      <th>20</th>
      <td>pub_rec</td>
      <td>Number of derogatory public records</td>
    </tr>
    <tr>
      <th>21</th>
      <td>revol_bal</td>
      <td>Total credit revolving balance</td>
    </tr>
    <tr>
      <th>22</th>
      <td>revol_util</td>
      <td>Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.</td>
    </tr>
    <tr>
      <th>23</th>
      <td>total_acc</td>
      <td>The total number of credit lines currently in the borrower's credit file</td>
    </tr>
    <tr>
      <th>24</th>
      <td>initial_list_status</td>
      <td>The initial listing status of the loan. Possible values are – W, F</td>
    </tr>
    <tr>
      <th>25</th>
      <td>application_type</td>
      <td>Indicates whether the loan is an individual application or a joint application with two co-borrowers</td>
    </tr>
    <tr>
      <th>26</th>
      <td>mort_acc</td>
      <td>Number of mortgage accounts.</td>
    </tr>
    <tr>
      <th>27</th>
      <td>pub_rec_bankruptcies</td>
      <td>Number of public record bankruptcies</td>
    </tr>
  </tbody>
</table>

## Import Required Packages and Data


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
df = pd.read_csv('../DATA/lending_club_loan_two.csv')
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 396030 entries, 0 to 396029
    Data columns (total 27 columns):
     #   Column                Non-Null Count   Dtype  
    ---  ------                --------------   -----  
     0   loan_amnt             396030 non-null  float64
     1   term                  396030 non-null  object 
     2   int_rate              396030 non-null  float64
     3   installment           396030 non-null  float64
     4   grade                 396030 non-null  object 
     5   sub_grade             396030 non-null  object 
     6   emp_title             373103 non-null  object 
     7   emp_length            377729 non-null  object 
     8   home_ownership        396030 non-null  object 
     9   annual_inc            396030 non-null  float64
     10  verification_status   396030 non-null  object 
     11  issue_d               396030 non-null  object 
     12  loan_status           396030 non-null  object 
     13  purpose               396030 non-null  object 
     14  title                 394275 non-null  object 
     15  dti                   396030 non-null  float64
     16  earliest_cr_line      396030 non-null  object 
     17  open_acc              396030 non-null  float64
     18  pub_rec               396030 non-null  float64
     19  revol_bal             396030 non-null  float64
     20  revol_util            395754 non-null  float64
     21  total_acc             396030 non-null  float64
     22  initial_list_status   396030 non-null  object 
     23  application_type      396030 non-null  object 
     24  mort_acc              358235 non-null  float64
     25  pub_rec_bankruptcies  395495 non-null  float64
     26  address               396030 non-null  object 
    dtypes: float64(12), object(15)
    memory usage: 81.6+ MB
    


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_amnt</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>emp_title</th>
      <th>emp_length</th>
      <th>home_ownership</th>
      <th>annual_inc</th>
      <th>...</th>
      <th>open_acc</th>
      <th>pub_rec</th>
      <th>revol_bal</th>
      <th>revol_util</th>
      <th>total_acc</th>
      <th>initial_list_status</th>
      <th>application_type</th>
      <th>mort_acc</th>
      <th>pub_rec_bankruptcies</th>
      <th>address</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10000.0</td>
      <td>36 months</td>
      <td>11.44</td>
      <td>329.48</td>
      <td>B</td>
      <td>B4</td>
      <td>Marketing</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>117000.0</td>
      <td>...</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>36369.0</td>
      <td>41.8</td>
      <td>25.0</td>
      <td>w</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0174 Michelle Gateway\nMendozaberg, OK 22690</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8000.0</td>
      <td>36 months</td>
      <td>11.99</td>
      <td>265.68</td>
      <td>B</td>
      <td>B5</td>
      <td>Credit analyst</td>
      <td>4 years</td>
      <td>MORTGAGE</td>
      <td>65000.0</td>
      <td>...</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>20131.0</td>
      <td>53.3</td>
      <td>27.0</td>
      <td>f</td>
      <td>INDIVIDUAL</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1076 Carney Fort Apt. 347\nLoganmouth, SD 05113</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15600.0</td>
      <td>36 months</td>
      <td>10.49</td>
      <td>506.97</td>
      <td>B</td>
      <td>B3</td>
      <td>Statistician</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>43057.0</td>
      <td>...</td>
      <td>13.0</td>
      <td>0.0</td>
      <td>11987.0</td>
      <td>92.2</td>
      <td>26.0</td>
      <td>f</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>87025 Mark Dale Apt. 269\nNew Sabrina, WV 05113</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7200.0</td>
      <td>36 months</td>
      <td>6.49</td>
      <td>220.65</td>
      <td>A</td>
      <td>A2</td>
      <td>Client Advocate</td>
      <td>6 years</td>
      <td>RENT</td>
      <td>54000.0</td>
      <td>...</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>5472.0</td>
      <td>21.5</td>
      <td>13.0</td>
      <td>f</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>823 Reid Ford\nDelacruzside, MA 00813</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24375.0</td>
      <td>60 months</td>
      <td>17.27</td>
      <td>609.33</td>
      <td>C</td>
      <td>C5</td>
      <td>Destiny Management Inc.</td>
      <td>9 years</td>
      <td>MORTGAGE</td>
      <td>55000.0</td>
      <td>...</td>
      <td>13.0</td>
      <td>0.0</td>
      <td>24584.0</td>
      <td>69.8</td>
      <td>43.0</td>
      <td>f</td>
      <td>INDIVIDUAL</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>679 Luna Roads\nGreggshire, VA 11650</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>



# Part1: Exploratory Data Analysis

Since I wanted to predict loan_status, I created a countplot as shown below to see how balanced the labels were:


```python
sns.countplot(x='loan_status', data=df)
```




    <Axes: xlabel='loan_status', ylabel='count'>





![alt text](/img/posts/output_9_1.png)

    


I had an imbalanced dataset. I expected to do very well in terms of accuracy but I had to use recall and precision to evaluate my data.

I created a histogram of the loan_amnt column.


```python
plt.figure(figsize=(9,4))
sns.histplot(data=df,x='loan_amnt', bins=40)
```




    <Axes: xlabel='loan_amnt', ylabel='Count'>





![alt text](/img/posts/output_12_1.png)
    


#### It shows that the vast majority of loans are between 5000 and 25000$.

### Let's explore correlation between the continuous feature variables and visualize it using heatmap.


```python
df.corr()
```

    C:\Users\arezo\AppData\Local\Temp\ipykernel_25184\1134722465.py:1: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
      df.corr()
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_amnt</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>annual_inc</th>
      <th>dti</th>
      <th>open_acc</th>
      <th>pub_rec</th>
      <th>revol_bal</th>
      <th>revol_util</th>
      <th>total_acc</th>
      <th>mort_acc</th>
      <th>pub_rec_bankruptcies</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>loan_amnt</th>
      <td>1.000000</td>
      <td>0.168921</td>
      <td>0.953929</td>
      <td>0.336887</td>
      <td>0.016636</td>
      <td>0.198556</td>
      <td>-0.077779</td>
      <td>0.328320</td>
      <td>0.099911</td>
      <td>0.223886</td>
      <td>0.222315</td>
      <td>-0.106539</td>
    </tr>
    <tr>
      <th>int_rate</th>
      <td>0.168921</td>
      <td>1.000000</td>
      <td>0.162758</td>
      <td>-0.056771</td>
      <td>0.079038</td>
      <td>0.011649</td>
      <td>0.060986</td>
      <td>-0.011280</td>
      <td>0.293659</td>
      <td>-0.036404</td>
      <td>-0.082583</td>
      <td>0.057450</td>
    </tr>
    <tr>
      <th>installment</th>
      <td>0.953929</td>
      <td>0.162758</td>
      <td>1.000000</td>
      <td>0.330381</td>
      <td>0.015786</td>
      <td>0.188973</td>
      <td>-0.067892</td>
      <td>0.316455</td>
      <td>0.123915</td>
      <td>0.202430</td>
      <td>0.193694</td>
      <td>-0.098628</td>
    </tr>
    <tr>
      <th>annual_inc</th>
      <td>0.336887</td>
      <td>-0.056771</td>
      <td>0.330381</td>
      <td>1.000000</td>
      <td>-0.081685</td>
      <td>0.136150</td>
      <td>-0.013720</td>
      <td>0.299773</td>
      <td>0.027871</td>
      <td>0.193023</td>
      <td>0.236320</td>
      <td>-0.050162</td>
    </tr>
    <tr>
      <th>dti</th>
      <td>0.016636</td>
      <td>0.079038</td>
      <td>0.015786</td>
      <td>-0.081685</td>
      <td>1.000000</td>
      <td>0.136181</td>
      <td>-0.017639</td>
      <td>0.063571</td>
      <td>0.088375</td>
      <td>0.102128</td>
      <td>-0.025439</td>
      <td>-0.014558</td>
    </tr>
    <tr>
      <th>open_acc</th>
      <td>0.198556</td>
      <td>0.011649</td>
      <td>0.188973</td>
      <td>0.136150</td>
      <td>0.136181</td>
      <td>1.000000</td>
      <td>-0.018392</td>
      <td>0.221192</td>
      <td>-0.131420</td>
      <td>0.680728</td>
      <td>0.109205</td>
      <td>-0.027732</td>
    </tr>
    <tr>
      <th>pub_rec</th>
      <td>-0.077779</td>
      <td>0.060986</td>
      <td>-0.067892</td>
      <td>-0.013720</td>
      <td>-0.017639</td>
      <td>-0.018392</td>
      <td>1.000000</td>
      <td>-0.101664</td>
      <td>-0.075910</td>
      <td>0.019723</td>
      <td>0.011552</td>
      <td>0.699408</td>
    </tr>
    <tr>
      <th>revol_bal</th>
      <td>0.328320</td>
      <td>-0.011280</td>
      <td>0.316455</td>
      <td>0.299773</td>
      <td>0.063571</td>
      <td>0.221192</td>
      <td>-0.101664</td>
      <td>1.000000</td>
      <td>0.226346</td>
      <td>0.191616</td>
      <td>0.194925</td>
      <td>-0.124532</td>
    </tr>
    <tr>
      <th>revol_util</th>
      <td>0.099911</td>
      <td>0.293659</td>
      <td>0.123915</td>
      <td>0.027871</td>
      <td>0.088375</td>
      <td>-0.131420</td>
      <td>-0.075910</td>
      <td>0.226346</td>
      <td>1.000000</td>
      <td>-0.104273</td>
      <td>0.007514</td>
      <td>-0.086751</td>
    </tr>
    <tr>
      <th>total_acc</th>
      <td>0.223886</td>
      <td>-0.036404</td>
      <td>0.202430</td>
      <td>0.193023</td>
      <td>0.102128</td>
      <td>0.680728</td>
      <td>0.019723</td>
      <td>0.191616</td>
      <td>-0.104273</td>
      <td>1.000000</td>
      <td>0.381072</td>
      <td>0.042035</td>
    </tr>
    <tr>
      <th>mort_acc</th>
      <td>0.222315</td>
      <td>-0.082583</td>
      <td>0.193694</td>
      <td>0.236320</td>
      <td>-0.025439</td>
      <td>0.109205</td>
      <td>0.011552</td>
      <td>0.194925</td>
      <td>0.007514</td>
      <td>0.381072</td>
      <td>1.000000</td>
      <td>0.027239</td>
    </tr>
    <tr>
      <th>pub_rec_bankruptcies</th>
      <td>-0.106539</td>
      <td>0.057450</td>
      <td>-0.098628</td>
      <td>-0.050162</td>
      <td>-0.014558</td>
      <td>-0.027732</td>
      <td>0.699408</td>
      <td>-0.124532</td>
      <td>-0.086751</td>
      <td>0.042035</td>
      <td>0.027239</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(9,6))
sns.heatmap(df.corr(), annot=True, cmap='YlGnBu')
```

    C:\Users\arezo\AppData\Local\Temp\ipykernel_25184\2692311087.py:2: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
      sns.heatmap(df.corr(), annot=True, cmap='YlGnBu')
    




    <Axes: >





![alt text](/img/posts/output_16_2.png)
    


### We noticed an almost perfect correlation between the loan amount and "installment" features. Let's explore these features further. For that, we will first find their description and then draw a scatterplot between them.


```python
data_info = pd.read_csv('../DATA/lending_club_info.csv',index_col='LoanStatNew')

def feature_info(col_name):
    print(data_info.loc[col_name]['Description'])
```


```python
feature_info('installment')
```

    The monthly payment owed by the borrower if the loan originates.
    


```python
feature_info('loan_amnt')
```

    The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.
    


```python
sns.scatterplot(y='loan_amnt', x='installment', data=df)
```




    <Axes: xlabel='installment', ylabel='loan_amnt'>





![alt text](/img/posts/output_21_1.png)
    


#### A perfect correlation is noticed between these two features.

### Let's see if there is any relationship between the loan status and the amount of the loan. For that, we will create a boxplot.


```python
sns.boxplot(data=df, x='loan_status', y='loan_amnt')
```




    <Axes: xlabel='loan_status', ylabel='loan_amnt'>





![alt text](/img/posts/output_24_1.png)
    


#### They look pretty similar. 

### Let's calculate the summary statistics for the loan amount, grouped by the loan_status.


```python
df.groupby('loan_status')['loan_amnt'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>loan_status</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Charged Off</th>
      <td>77673.0</td>
      <td>15126.300967</td>
      <td>8505.090557</td>
      <td>1000.0</td>
      <td>8525.0</td>
      <td>14000.0</td>
      <td>20000.0</td>
      <td>40000.0</td>
    </tr>
    <tr>
      <th>Fully Paid</th>
      <td>318357.0</td>
      <td>13866.878771</td>
      <td>8302.319699</td>
      <td>500.0</td>
      <td>7500.0</td>
      <td>12000.0</td>
      <td>19225.0</td>
      <td>40000.0</td>
    </tr>
  </tbody>
</table>
</div>



#### The average loan amount of the group is a little higher than fully paid ones which makes sense as higher loan amounts are more difficult to pay off.

### Let's explore the Grade and SubGrade columns that LendingClub attributes to the loans and check the unique possible grades and subgrades.


```python
grade_order = sorted(df['grade'].unique())
```


```python
Subgrade_order = sorted(df['sub_grade'].unique())
```

### Let's create a countplot per grade to see how loan status . Set the hue to the loan_status label.**


```python
sns.countplot(x='grade', hue='loan_status', data=df)
```




    <Axes: xlabel='grade', ylabel='count'>





![alt text](/img/posts/output_33_1.png)
    


### Let's order the bars from grade A to E


```python
sns.countplot(x='grade', hue='loan_status', data=df, order=grade_order)
```




    <Axes: xlabel='grade', ylabel='count'>





![alt text](/img/posts/output_35_1.png)
    


#### When we move from grade A to G, the ratio of fully paid loans to charged off loans decreases meaning that we face riskier groups.

### Let's draw a count plot per subgrade. 


```python
plt.figure(figsize=(10,4))
sns.countplot(x='sub_grade', data=df, order=Subgrade_order, palette='coolwarm')
```




    <Axes: xlabel='sub_grade', ylabel='count'>





![alt text](/img/posts/output_38_1.png)
    


#### A decrease in the number of loans can be seen when we move from sub_grades A to G which makes sense as we move towards riskier groups.

### Let's check all loans made per subgrade separated based on the loan_status. 


```python
plt.figure(figsize=(10,4))
sns.countplot(x='sub_grade', data=df, order=Subgrade_order, hue='loan_status', palette='coolwarm')
```




    <Axes: xlabel='sub_grade', ylabel='count'>





![alt text](/img/posts/output_41_1.png)
    


### It looks like F and G subgrades don't get paid back that often. Let's isolate those and recreate the countplot just for those subgrades.


```python
F_and_G = df[(df['grade']=='F')|(df['grade']=='G')]
```


```python
F_and_G_sub_grade_order = sorted((F_and_G)['sub_grade'].unique())
```


```python
plt.figure(figsize=(10,4))
sns.countplot(x='sub_grade', data=df, order=F_and_G_sub_grade_order, hue='loan_status', palette='coolwarm')
```




    <Axes: xlabel='sub_grade', ylabel='count'>





![alt text](/img/posts/output_45_1.png)
    


#### As shown in the above figure, the number of fully paid loans is almost equal to the number of charged-off ones for grades F and G.

### Let's create a new column called 'loan_repaid' which will contain a 1 if the loan status was "Fully Paid" and a 0 if it was "Charged Off".**


```python
def status(x):
    if x=='Fully Paid':
        return 1
    else:
        return 0
df['loan_repaid'] = df['loan_status'].apply(status)
```

### Alternative method


```python
df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})
```


```python
df[['loan_repaid','loan_status']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_repaid</th>
      <th>loan_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Fully Paid</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Fully Paid</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Fully Paid</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Fully Paid</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>Charged Off</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>396025</th>
      <td>1</td>
      <td>Fully Paid</td>
    </tr>
    <tr>
      <th>396026</th>
      <td>1</td>
      <td>Fully Paid</td>
    </tr>
    <tr>
      <th>396027</th>
      <td>1</td>
      <td>Fully Paid</td>
    </tr>
    <tr>
      <th>396028</th>
      <td>1</td>
      <td>Fully Paid</td>
    </tr>
    <tr>
      <th>396029</th>
      <td>1</td>
      <td>Fully Paid</td>
    </tr>
  </tbody>
</table>
<p>396030 rows × 2 columns</p>
</div>




```python
df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')
```

    C:\Users\arezo\AppData\Local\Temp\ipykernel_25184\660239616.py:1: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
      df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')
    




    <Axes: >





![alt text](/img/posts/output_52_2.png)
    


#### We can see that interest rate has the highest negative correlation with loan_repaid which totally makes sense as higher interest rate makes it more difficult to repay the loan.

---
# Part 2: Data PreProcessing


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_amnt</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>emp_title</th>
      <th>emp_length</th>
      <th>home_ownership</th>
      <th>annual_inc</th>
      <th>...</th>
      <th>pub_rec</th>
      <th>revol_bal</th>
      <th>revol_util</th>
      <th>total_acc</th>
      <th>initial_list_status</th>
      <th>application_type</th>
      <th>mort_acc</th>
      <th>pub_rec_bankruptcies</th>
      <th>address</th>
      <th>loan_repaid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10000.0</td>
      <td>36 months</td>
      <td>11.44</td>
      <td>329.48</td>
      <td>B</td>
      <td>B4</td>
      <td>Marketing</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>117000.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>36369.0</td>
      <td>41.8</td>
      <td>25.0</td>
      <td>w</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0174 Michelle Gateway\nMendozaberg, OK 22690</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8000.0</td>
      <td>36 months</td>
      <td>11.99</td>
      <td>265.68</td>
      <td>B</td>
      <td>B5</td>
      <td>Credit analyst</td>
      <td>4 years</td>
      <td>MORTGAGE</td>
      <td>65000.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>20131.0</td>
      <td>53.3</td>
      <td>27.0</td>
      <td>f</td>
      <td>INDIVIDUAL</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1076 Carney Fort Apt. 347\nLoganmouth, SD 05113</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15600.0</td>
      <td>36 months</td>
      <td>10.49</td>
      <td>506.97</td>
      <td>B</td>
      <td>B3</td>
      <td>Statistician</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>43057.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>11987.0</td>
      <td>92.2</td>
      <td>26.0</td>
      <td>f</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>87025 Mark Dale Apt. 269\nNew Sabrina, WV 05113</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7200.0</td>
      <td>36 months</td>
      <td>6.49</td>
      <td>220.65</td>
      <td>A</td>
      <td>A2</td>
      <td>Client Advocate</td>
      <td>6 years</td>
      <td>RENT</td>
      <td>54000.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>5472.0</td>
      <td>21.5</td>
      <td>13.0</td>
      <td>f</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>823 Reid Ford\nDelacruzside, MA 00813</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24375.0</td>
      <td>60 months</td>
      <td>17.27</td>
      <td>609.33</td>
      <td>C</td>
      <td>C5</td>
      <td>Destiny Management Inc.</td>
      <td>9 years</td>
      <td>MORTGAGE</td>
      <td>55000.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>24584.0</td>
      <td>69.8</td>
      <td>43.0</td>
      <td>f</td>
      <td>INDIVIDUAL</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>679 Luna Roads\nGreggshire, VA 11650</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>



# Missing Data

### Let's explore missing data columns. We use a variety of factors to decide whether or not they would be useful, and to see if we should keep, discard, or fill in the missing data.

## Length of the dataframe:


```python
df_length = len(df)
```

### Total count of missing values per column:


```python
df.isna().sum()
```




    loan_amnt                   0
    term                        0
    int_rate                    0
    installment                 0
    grade                       0
    sub_grade                   0
    emp_title               22927
    emp_length              18301
    home_ownership              0
    annual_inc                  0
    verification_status         0
    issue_d                     0
    loan_status                 0
    purpose                     0
    title                    1755
    dti                         0
    earliest_cr_line            0
    open_acc                    0
    pub_rec                     0
    revol_bal                   0
    revol_util                276
    total_acc                   0
    initial_list_status         0
    application_type            0
    mort_acc                37795
    pub_rec_bankruptcies      535
    address                     0
    loan_repaid                 0
    dtype: int64



### Let's convert this Series to be in term of percentage of the total DataFrame


```python
df.isna().sum()/(df_length)*100
```




    loan_amnt               0.000000
    term                    0.000000
    int_rate                0.000000
    installment             0.000000
    grade                   0.000000
    sub_grade               0.000000
    emp_title               5.789208
    emp_length              4.621115
    home_ownership          0.000000
    annual_inc              0.000000
    verification_status     0.000000
    issue_d                 0.000000
    loan_status             0.000000
    purpose                 0.000000
    title                   0.443148
    dti                     0.000000
    earliest_cr_line        0.000000
    open_acc                0.000000
    pub_rec                 0.000000
    revol_bal               0.000000
    revol_util              0.069692
    total_acc               0.000000
    initial_list_status     0.000000
    application_type        0.000000
    mort_acc                9.543469
    pub_rec_bankruptcies    0.135091
    address                 0.000000
    loan_repaid             0.000000
    dtype: float64



### Let's examine emp_title and emp_length to see whether it will be okay to drop them.


```python
feature_info('emp_title')
```

    The job title supplied by the Borrower when applying for the loan.*
    


```python
feature_info('emp_length')
```

    Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years. 
    

### Let's check the number of unique employment job titles.


```python
df['emp_title'].nunique()
```




    173105




```python
df['emp_title'].value_counts()
```




    Teacher                    4389
    Manager                    4250
    Registered Nurse           1856
    RN                         1846
    Supervisor                 1830
                               ... 
    Postman                       1
    McCarthy & Holthus, LLC       1
    jp flooring                   1
    Histology Technologist        1
    Gracon Services, Inc          1
    Name: emp_title, Length: 173105, dtype: int64



#### Realistically there are too many unique job titles to try to convert to numeric feature. Let's remove that emp_title column.


```python
df = df.drop('emp_title', axis=1)
```

**TASK: Create a count plot of the emp_length feature column. Challenge: Sort the order of the values.**


```python
sorted(df['emp_length'].dropna().unique())
```




    ['1 year',
     '10+ years',
     '2 years',
     '3 years',
     '4 years',
     '5 years',
     '6 years',
     '7 years',
     '8 years',
     '9 years',
     '< 1 year']




```python
emp_length_order = [ '< 1 year',
                      '1 year',
                     '2 years',
                     '3 years',
                     '4 years',
                     '5 years',
                     '6 years',
                     '7 years',
                     '8 years',
                     '9 years',
                     '10+ years']
```


```python
plt.figure(figsize=(10,4))
sns.countplot(data=df, x='emp_length', order=emp_length_order)
```




    <Axes: xlabel='emp_length', ylabel='count'>





![alt text](/img/posts/output_74_1.png)
    


#### It seems that the majority of people who took loan have been working for more than 10 years which makes sense as you have to have a kind of job security to be able to repay the loan.

### Let's plot out the countplot with a hue separating Fully Paid vs Charged Off


```python
plt.figure(figsize=(10,4))
sns.countplot(data=df, x='emp_length', order=emp_length_order, hue= 'loan_status')
```




    <Axes: xlabel='emp_length', ylabel='count'>





![alt text](/img/posts/output_77_1.png)
    


#### For people with more than 10+ years of employment length, the number of fully paid loans is much higher than charged-off ones.

### Let's find the percentage of charge-offs per category to see what percent of people per employment category didn't pay back their loan. It may help us to understand if there is a strong relationship between employment length and being charged off.


```python
fully_paid_number = df[df['loan_status']=='Fully Paid'].groupby('emp_length').count()['loan_status']
```


```python
charged_off_number = df[df['loan_status']=='Charged Off'].groupby('emp_length').count()['loan_status']
```


```python
total_loan_number = fully_paid_number + charged_off_number
```


```python
charged_off_percentage = charged_off_number/total_loan_number*100
```


```python
plt.figure(figsize=(8,4))
charged_off_percentage.plot(kind='bar')
```




    <Axes: xlabel='emp_length'>





![alt text](/img/posts/output_84_1.png)
    


#### Charge off rates are similar across all employment lengths. 

### Let's  drop the emp_length column.


```python
df = df.drop('emp_length', axis=1)
```

### Let's revisit the DataFrame to see what feature columns still have missing data.


```python
df.isna().sum()
```




    loan_amnt                   0
    term                        0
    int_rate                    0
    installment                 0
    grade                       0
    sub_grade                   0
    home_ownership              0
    annual_inc                  0
    verification_status         0
    issue_d                     0
    loan_status                 0
    purpose                     0
    title                    1755
    dti                         0
    earliest_cr_line            0
    open_acc                    0
    pub_rec                     0
    revol_bal                   0
    revol_util                276
    total_acc                   0
    initial_list_status         0
    application_type            0
    mort_acc                37795
    pub_rec_bankruptcies      535
    address                     0
    loan_repaid                 0
    dtype: int64



**TASK: Review the title column vs the purpose column. Is this repeated information?**


```python
df['purpose'].head(10)
```




    0              vacation
    1    debt_consolidation
    2           credit_card
    3           credit_card
    4           credit_card
    5    debt_consolidation
    6      home_improvement
    7           credit_card
    8    debt_consolidation
    9    debt_consolidation
    Name: purpose, dtype: object




```python
df['title'].head(10)
```




    0                   Vacation
    1         Debt consolidation
    2    Credit card refinancing
    3    Credit card refinancing
    4      Credit Card Refinance
    5         Debt consolidation
    6           Home improvement
    7       No More Credit Cards
    8         Debt consolidation
    9         Debt Consolidation
    Name: title, dtype: object



#### It seems that the title column is simply a string subcategory/description of the purpose column. 

### Let's drop the title column.


```python
df = df.drop('title', axis=1)
```

### Let's find out what the mort_acc feature represents.


```python
feat_info('mort_acc')
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[66], line 1
    ----> 1 feat_info('mort_acc')
    

    NameError: name 'feat_info' is not defined


### Let's create a value_counts of the mort_acc column.


```python
df['mort_acc'].value_counts()
```




    0.0     139777
    1.0      60416
    2.0      49948
    3.0      38049
    4.0      27887
    5.0      18194
    6.0      11069
    7.0       6052
    8.0       3121
    9.0       1656
    10.0       865
    11.0       479
    12.0       264
    13.0       146
    14.0       107
    15.0        61
    16.0        37
    17.0        22
    18.0        18
    19.0        15
    20.0        13
    24.0        10
    22.0         7
    21.0         4
    25.0         4
    27.0         3
    32.0         2
    31.0         2
    23.0         2
    26.0         2
    28.0         1
    30.0         1
    34.0         1
    Name: mort_acc, dtype: int64



### Let's review the other columns to see which most highly correlates to mort_acc.


```python
df.corr()['mort_acc'].sort_values()
```

    C:\Users\arezo\AppData\Local\Temp\ipykernel_25184\2388834679.py:1: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
      df.corr()['mort_acc'].sort_values()
    




    int_rate               -0.082583
    dti                    -0.025439
    revol_util              0.007514
    pub_rec                 0.011552
    pub_rec_bankruptcies    0.027239
    loan_repaid             0.073111
    open_acc                0.109205
    installment             0.193694
    revol_bal               0.194925
    loan_amnt               0.222315
    annual_inc              0.236320
    total_acc               0.381072
    mort_acc                1.000000
    Name: mort_acc, dtype: float64



#### Looks like the total_acc feature correlates with the mort_acc , this makes sense! 


```python
feat_info('total_acc')
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[69], line 1
    ----> 1 feat_info('total_acc')
    

    NameError: name 'feat_info' is not defined


### Let's try the fillna() approach to replace missing data. We will group the dataframe by the total_acc and calculate the mean value for the mort_acc per total_acc entry. 


```python
df_acc = df[['total_acc','mort_acc']].sort_values(by='total_acc')
df_acc
```


```python
mort_acc_mean = df.groupby('total_acc')['mort_acc'].mean()
```

### Let's fill in the missing mort_acc values based on their total_acc value. If the mort_acc is missing, then we will fill in that missing value with the mean value corresponding to its total_acc value. 


```python
mort_acc_mean = pd.DataFrame(mort_acc_mean)
mort_acc_mean.columns = ['mort_acc_mean']
df_acc = df_acc.merge(mort_acc_mean, on='total_acc', how='inner')
```


```python
df['mort_acc'].fillna(value=df_acc['mort_acc_mean'], inplace=True)
```


```python
df.isna().sum()
```

### revol_util and the pub_rec_bankruptcies have missing data points, but they account for less than 0.5% of the total data. Therefore, we will remove the rows that are missing those values in those columns.


```python
df = df.dropna()
```


```python
df.isna().sum()
```




    loan_amnt                   0
    term                        0
    int_rate                    0
    installment                 0
    grade                       0
    sub_grade                   0
    home_ownership              0
    annual_inc                  0
    verification_status         0
    issue_d                     0
    loan_status                 0
    purpose                     0
    dti                         0
    earliest_cr_line            0
    open_acc                    0
    pub_rec                     0
    revol_bal                   0
    revol_util                276
    total_acc                   0
    initial_list_status         0
    application_type            0
    mort_acc                37795
    pub_rec_bankruptcies      535
    address                     0
    loan_repaid                 0
    dtype: int64



## Categorical Variables

### Let's List all the columns that are currently non-numeric.


```python
df.select_dtypes(['object']).columns
```




    Index(['term', 'grade', 'sub_grade', 'home_ownership', 'verification_status',
           'issue_d', 'loan_status', 'purpose', 'earliest_cr_line',
           'initial_list_status', 'application_type', 'address'],
          dtype='object')



### term feature


```python
df['term'].value_counts()
```




     36 months    302005
     60 months     94025
    Name: term, dtype: int64



#### term feature has just 2 values of 36 and 60 months. Let's remove the month and simply convert the categorical data to numeric one.


```python
def  conversion(x):
    return int(x[1:3])

df['term'] = df['term'].apply(conversion)
# Alternative: df['term'] = df['term'].apply(lambda term: int(term[:3]))
```


```python
df['term'].value_counts()
```




    36    302005
    60     94025
    Name: term, dtype: int64



### grade feature: We already know grade is part of sub_grade, so we will drop the grade feature.


```python
df = df.drop('grade', axis=1)
```

### sub_grade feature: We will keep it to convert it to a numeric feature later on.

### home_ownership feature


```python
df['home_ownership'].value_counts()
```




    MORTGAGE    198348
    RENT        159790
    OWN          37746
    OTHER          112
    NONE            31
    ANY              3
    Name: home_ownership, dtype: int64




```python
df['verification_status'].value_counts()
```




    Verified           139563
    Source Verified    131385
    Not Verified       125082
    Name: verification_status, dtype: int64



### issue_d feature


```python
df['issue_d'].value_counts()
```




    Oct-2014    14846
    Jul-2014    12609
    Jan-2015    11705
    Dec-2013    10618
    Nov-2013    10496
                ...  
    Jul-2007       26
    Sep-2008       25
    Nov-2007       22
    Sep-2007       15
    Jun-2007        1
    Name: issue_d, Length: 115, dtype: int64



#### This would be data leakage as we wouldn't know beforehand whether or not a loan would be issued when using our model, so in theory, we wouldn't have an issue_date. Let's drop this feature.


```python
df = df = df.drop('issue_d', axis=1)
```

### loan_status feature: As loan_status column is a duplicate of the loan_repaid column, we will drop the load_status column and use the loan_repaid column since its already in 0s and 1s.


```python
df = df.drop('loan_status', axis=1)
```

### purpose feature


```python
df['purpose'].value_counts()
```




    debt_consolidation    234507
    credit_card            83019
    home_improvement       24030
    other                  21185
    major_purchase          8790
    small_business          5701
    car                     4697
    medical                 4196
    moving                  2854
    vacation                2452
    house                   2201
    wedding                 1812
    renewable_energy         329
    educational              257
    Name: purpose, dtype: int64



### earliest_cr_line feature


```python
df['earliest_cr_line'].value_counts()
```




    Oct-2000    3017
    Aug-2000    2935
    Oct-2001    2896
    Aug-2001    2884
    Nov-2000    2736
                ... 
    Jul-1958       1
    Nov-1957       1
    Jan-1953       1
    Jul-1955       1
    Aug-1959       1
    Name: earliest_cr_line, Length: 684, dtype: int64



#### This appears to be a historical time stamp feature. Let's extract the year from this feature and convert it to a numeric feature. 


```python
def year(x):
    return int(x[4:])

df['earliest_cr_line'].apply(year)
```




    0         1990
    1         2004
    2         2007
    3         2006
    4         1999
              ... 
    396025    2004
    396026    2006
    396027    1997
    396028    1990
    396029    1998
    Name: earliest_cr_line, Length: 396030, dtype: int64



#### Let's set this new data to a feature column called 'earliest_cr_year'.Then drop the earliest_cr_line feature.


```python
df['earliest_cr_year']= df['earliest_cr_line'].apply(year)
```


```python
df = df.drop('earliest_cr_line', axis=1)
```

### initial_list_status feature


```python
df['initial_list_status'].value_counts()
```




    f    238066
    w    157964
    Name: initial_list_status, dtype: int64



### application_type feature


```python
df['application_type'].value_counts()
```




    INDIVIDUAL    395319
    JOINT            425
    DIRECT_PAY       286
    Name: application_type, dtype: int64



### address feature


```python
df['address'].value_counts()
```




    USCGC Smith\nFPO AE 70466                           8
    USS Johnson\nFPO AE 48052                           8
    USNS Johnson\nFPO AE 05113                          8
    USS Smith\nFPO AP 70466                             8
    USNS Johnson\nFPO AP 48052                          7
                                                       ..
    455 Tricia Cove\nAustinbury, FL 00813               1
    7776 Flores Fall\nFernandezshire, UT 05113          1
    6577 Mia Harbors Apt. 171\nRobertshire, OK 22690    1
    8141 Cox Greens Suite 186\nMadisonstad, VT 05113    1
    787 Michelle Causeway\nBriannaton, AR 48052         1
    Name: address, Length: 393700, dtype: int64



#### Let's extract the zip_code from address.


```python
def zip_code(x):
    return x[-5:]

df ['zip_code'] = df['address'].apply(zip_code)
```

### Let's drop address column.


```python
df = df.drop('address', axis=1)
```

### Let's apply OneHotEncoder to convert all categorical features except loan_status to numeric ones.


```python
from sklearn.preprocessing import OneHotEncoder

# Create a list of categorical variables                    
categorical_vars = ['sub_grade', 'home_ownership', 'verification_status',
       'purpose', 'initial_list_status', 'application_type', 'zip_code'] 

# Create and apply OneHotEncoder while removing the dummy variable
one_hot_encoder = OneHotEncoder(sparse = False, drop = 'first') 

# Apply fit_transform on data
df_encoded = one_hot_encoder.fit_transform(df[categorical_vars])

# Get feature names to see what each column in the 'encoder_vars_array' presents
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

# Convert our result from an array to a DataFrame
df_encoded = pd.DataFrame(df_encoded, columns = encoder_feature_names)

# Concatenate (Link together in a series or chain) new DataFrame to our original DataFrame 
df = pd.concat([df.reset_index(drop = True),df_encoded.reset_index(drop = True)], axis = 1)

# Drop the original categorical variable columns
df.drop(categorical_vars, axis = 1, inplace = True)
```

    C:\Users\arezo\anaconda3\lib\site-packages\sklearn\preprocessing\_encoders.py:828: FutureWarning: `sparse` was renamed to `sparse_output` in version 1.2 and will be removed in 1.4. `sparse_output` is ignored unless you leave `sparse` to its default value.
      warnings.warn(
    

## Train Test Split


```python
from sklearn.model_selection import train_test_split
```

### Create input and output variables


```python
X = df.drop('loan_repaid', axis=1)
```


```python
y = df['loan_repaid']
```

### Due to low RAM, let's grab a sample for data training to save time on training.


```python
df = df.sample(frac=0.1,random_state=101)
print(len(df))
```

    39603
    


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)
```

## Normalizing the Data


```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train))         
X_test = pd.DataFrame(scaler.transform(X_test)) 
```

# Creating the Model


```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
```


```python
X_train.shape
```




    (316824, 80)




```python
model = Sequential()

# input layer
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))

# output layer
model.add(Dense(units=1,activation='sigmoid'))                       # output is either 0 or 1

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam')
```

### Let's fit the model to the training data. 


```python
model.fit(x=X_train, 
          y=y_train, 
          epochs=50,
          batch_size=256,
          validation_data=(X_test, y_test))
```

    Epoch 1/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 6ms/step - loss: 0.5360 - val_loss: 0.4944
    Epoch 2/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 5ms/step - loss: 0.4998 - val_loss: 0.4945
    Epoch 3/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 6ms/step - loss: 0.4974 - val_loss: 0.4944
    Epoch 4/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 6ms/step - loss: 0.4957 - val_loss: 0.4944
    Epoch 5/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 6ms/step - loss: 0.4973 - val_loss: 0.4943
    Epoch 6/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 6ms/step - loss: 0.4949 - val_loss: 0.4943
    Epoch 7/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 6ms/step - loss: 0.4948 - val_loss: 0.4943
    Epoch 8/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 5ms/step - loss: 0.4953 - val_loss: 0.4943
    Epoch 9/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 4ms/step - loss: 0.4959 - val_loss: 0.4943
    Epoch 10/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 5ms/step - loss: 0.4969 - val_loss: 0.4943
    Epoch 11/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 6ms/step - loss: 0.4947 - val_loss: 0.4944
    Epoch 12/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 6ms/step - loss: 0.4950 - val_loss: 0.4943
    Epoch 13/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 6ms/step - loss: 0.4960 - val_loss: 0.4944
    Epoch 14/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 6ms/step - loss: 0.4931 - val_loss: 0.4944
    Epoch 15/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 6ms/step - loss: 0.4938 - val_loss: 0.4943
    Epoch 16/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 4ms/step - loss: 0.4952 - val_loss: 0.4943
    Epoch 17/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 5ms/step - loss: 0.4964 - val_loss: 0.4944
    Epoch 18/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 5ms/step - loss: 0.4949 - val_loss: 0.4943
    Epoch 19/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 6ms/step - loss: 0.4961 - val_loss: 0.4943
    Epoch 20/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 6ms/step - loss: 0.4952 - val_loss: 0.4944
    Epoch 21/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 4ms/step - loss: 0.4939 - val_loss: 0.4943
    Epoch 22/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 4ms/step - loss: 0.4946 - val_loss: 0.4943
    Epoch 23/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 6ms/step - loss: 0.4957 - val_loss: 0.4943
    Epoch 24/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 6ms/step - loss: 0.4950 - val_loss: 0.4943
    Epoch 25/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 6ms/step - loss: 0.4956 - val_loss: 0.4943
    Epoch 26/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 6ms/step - loss: 0.4947 - val_loss: 0.4943
    Epoch 27/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 5ms/step - loss: 0.4937 - val_loss: 0.4943
    Epoch 28/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 5ms/step - loss: 0.4966 - val_loss: 0.4943
    Epoch 29/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 6ms/step - loss: 0.4954 - val_loss: 0.4943
    Epoch 30/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 5ms/step - loss: 0.4951 - val_loss: 0.4943
    Epoch 31/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 5ms/step - loss: 0.4955 - val_loss: 0.4943
    Epoch 32/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 5ms/step - loss: 0.4946 - val_loss: 0.4943
    Epoch 33/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 6ms/step - loss: 0.4950 - val_loss: 0.4943
    Epoch 34/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 6ms/step - loss: 0.4961 - val_loss: 0.4943
    Epoch 35/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 5ms/step - loss: 0.4952 - val_loss: 0.4943
    Epoch 36/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 4ms/step - loss: 0.4961 - val_loss: 0.4943
    Epoch 37/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 6ms/step - loss: 0.4957 - val_loss: 0.4943
    Epoch 38/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 6ms/step - loss: 0.4954 - val_loss: 0.4943
    Epoch 39/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 6ms/step - loss: 0.4967 - val_loss: 0.4943
    Epoch 40/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 6ms/step - loss: 0.4948 - val_loss: 0.4943
    Epoch 41/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 6ms/step - loss: 0.4942 - val_loss: 0.4943
    Epoch 42/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 6ms/step - loss: 0.4956 - val_loss: 0.4943
    Epoch 43/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 6ms/step - loss: 0.4939 - val_loss: 0.4944
    Epoch 44/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 6ms/step - loss: 0.4953 - val_loss: 0.4943
    Epoch 45/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 5ms/step - loss: 0.4942 - val_loss: 0.4943
    Epoch 46/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 6ms/step - loss: 0.4963 - val_loss: 0.4943
    Epoch 47/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 5ms/step - loss: 0.4962 - val_loss: 0.4943
    Epoch 48/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 5ms/step - loss: 0.4957 - val_loss: 0.4943
    Epoch 49/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 5ms/step - loss: 0.4966 - val_loss: 0.4944
    Epoch 50/50
    [1m1238/1238[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 6ms/step - loss: 0.4955 - val_loss: 0.4943
    




    <keras.src.callbacks.history.History at 0x20b8a01ca90>



### Let's save our model.


```python
from tensorflow.keras.models import load_model
```


```python
model.save('full_data_project_model.h5')  
```

    WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
    

# Part 3: Evaluating Model Performance

### Let's plot out the validation loss versus the training loss


```python
loss = pd.DataFrame(model.history.history)
```


```python
loss.plot()
```




    <Axes: >





![alt text](/img/posts/output_176_1.png)
    


### Let's create our prediction from the X_test set and display a classification report and confusion matrix for the X_test set.


```python
y_predict = model.predict(X_test)
```

    [1m2476/2476[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 2ms/step
    


```python
y_predict = pd.DataFrame(model.predict(X_test), columns=['Predicted Y'])
```

    [1m2476/2476[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 2ms/step
    


```python
def p_class(x):
    if x>0.5:
        return 1
    else:
        return 0
    
y_predict_class = y_predict['Predicted Y'].apply(p_class)
y_predict_class
```




    0        1
    1        1
    2        1
    3        1
    4        1
            ..
    79201    1
    79202    1
    79203    1
    79204    1
    79205    1
    Name: Predicted Y, Length: 79206, dtype: int64




```python
from sklearn.metrics import classification_report,confusion_matrix
```


```python
print(classification_report(y_test, y_predict_class))
```

                  precision    recall  f1-score   support
    
               0       0.00      0.00      0.00     15493
               1       0.80      1.00      0.89     63713
    
        accuracy                           0.80     79206
       macro avg       0.40      0.50      0.45     79206
    weighted avg       0.65      0.80      0.72     79206
    
    

    C:\Users\arezo\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\arezo\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\arezo\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    


```python
confusion_matrix(y_test, y_predict_class)
```




    array([[    0, 15493],
           [    0, 63713]], dtype=int64)



# Part 4: New Case

### Given the customer below, would you offer this person a loan?


```python
import random
random.seed(101)
random_ind = random.randint(0,len(df))

new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]
new_customer
```




    loan_amnt          2000.00
    term                 36.00
    int_rate              7.90
    installment          62.59
    annual_inc        20400.00
                        ...   
    zip_code_30723        0.00
    zip_code_48052        0.00
    zip_code_70466        0.00
    zip_code_86630        1.00
    zip_code_93700        0.00
    Name: 87921, Length: 80, dtype: float64



### We have to make sure that our data is numpy array not a dataframe.


```python
new_customer = new_customer.values.reshape(1,80)
```


```python
new_customer = scaler.transform(new_customer)
```

    C:\Users\arezo\anaconda3\lib\site-packages\sklearn\base.py:420: UserWarning: X does not have valid feature names, but MinMaxScaler was fitted with feature names
      warnings.warn(
    


```python
y_predict = model.predict(new_customer)
```

    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 45ms/step
    


```python
y_predict = pd.DataFrame(model.predict(new_customer), columns=['Predicted Y'])
```

    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 40ms/step
    


```python
def p_class(x):
    if x>0.5:
        return 1
    else:
        return 0
    
y_predict_class = y_predict['Predicted Y'].apply(p_class)
y_predict_class
```




    0    1
    Name: Predicted Y, dtype: int64



#### We will probably give the loan to this person according to this model prediction.

### Now check if this person actually ended up paying back their loan.


```python
df['loan_repaid'].iloc[random_ind]
```




    0



#### It seems that the customer had not actually repaid the loan.
