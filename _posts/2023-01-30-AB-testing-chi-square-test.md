---
layout: post
title: Assessing Campaign Performance in Promoting a New Service (Delivery Club) Using Chi-Square Test For Independence
image: "/posts/ab-testing-title-img.png"
tags: [AB Testing, Hypothesis Testing, Chi-Square, Python]
---

In this project, I applied the Chi-Square Test For Independence (a Hypothesis Test) to assess the performance of two types of mailers that were sent out to promote a new service! 

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
- [01. Data Overview & Preparation](#data-overview)
- [02. Applying Chi-Square Test For Independence](#chi-square-application)
- [03. Analysing The Results](#chi-square-results)
- [04. Discussion](#discussion)

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Earlier in the year, a grocery retailer ran a campaign to promote their new "Delivery Club" - an initiative that costed a customer $100 per year for membership, but offered free grocery deliveries rather than the normal cost of $10 per delivery.

For the campaign promoting the club, customers were put randomly into three groups - the first group received a low-quality, low-cost mailer, the second group received a high-quality, high-cost mailer, and the third group was a control group, receiving no mailer at all.

The grocery retailer knows that customers who were contacted, signed up for the Delivery Club at a far higher rate than the control group, but now wants to understand if there is a significant difference in signup rate between the cheap mailer and the expensive mailer.  This will allow them to make more informed decisions in the future, aiming to optimize campaign return on investment!

<br>
<br>
### Actions <a name="overview-actions"></a>

For this test, as it was focused on comparing the *rates* of two groups and the response was Yes or No (binary) for each mailer- I applied the Chi-Square Test For Independence. 

**Note:** Another option when comparing "rates" was a test known as the *Z-Test For Proportions*. While I could use this test here, I chose the Chi-Square Test For Independence because:

* The resulting test statistic for both tests would be the same
* The Chi-Square Test could be represented using 2x2 tables of data - meaning it could be easier to explain to stakeholders
* The Chi-Square Test could extend out to more than 2 groups - meaning the client could have one consistent approach to measuring significance

From the *campaign_data* table in the client database, I excluded customers who were in the control group and isolated customers who received "Mailer 1" (low cost) and "Mailer 2" (high cost) for this campaign.

I set out my hypotheses and Acceptance Criteria for the test, as follows:

**Null Hypothesis:** There is no relationship between mailer type and signup rate. They are independent.
**Alternate Hypothesis:** There is a relationship between mailer type and signup rate. They are not independent.
**Acceptance Criteria:** 0.05

As a requirement of the Chi-Square Test For Independence, I aggregated this data down to a 2x2 matrix for *signup_flag* by *mailer_type* and fed this into the algorithm (using the *scipy* library) to calculate the Chi-Square Statistic, p-value, Degrees of Freedom, and expected values

<br>
<br>

<br>

___

<br>
# Data Overview & Preparation  <a name="data-overview"></a>

In the client database, I had a *campaign_data* table which showed me which customers received each type of "Delivery Club" mailer, which customers were in the control group, and which customers joined the club.

For this task, I was looking to find evidence that the Delivery Club signup rate for customers who received "Mailer 1" (low cost) was different from those who received "Mailer 2" (high cost) and thus from the *campaign_data* table, I just extracted customers in those two groups, and excluded customers who were in the control group.

In the code below, I:

* Loaded the Python libraries I required for importing the data and performing the chi-square test (using scipy)
* Imported the required data from the *campaign_data* table
* Excluded customers in the control group, giving me a dataset with Mailer 1 & Mailer 2 customers only

<br>
```python

# install the required python libraries
import pandas as pd
from scipy.stats import chi2_contingency, chi2

# import campaign data
campaign_data = ...

# remove customers who were in the control group
campaign_data = campaign_data.loc[campaign_data["mailer_type"] != "Control"]

```
<br>
A sample of this data (the first 10 rows) can be seen below:
<br>
<br>

| **customer_id** | **campaign_name** | **mailer_type** | **signup_flag** |
|---|---|---|---|
| 74 | delivery_club | Mailer1 | 1 |
| 524 | delivery_club | Mailer1 | 1 |
| 607 | delivery_club | Mailer2 | 1 |
| 343 | delivery_club | Mailer1 | 0 |
| 322 | delivery_club | Mailer2 | 1 |
| 115 | delivery_club | Mailer2 | 0 |
| 1 | delivery_club | Mailer2 | 1 |
| 120 | delivery_club | Mailer1 | 1 |
| 52 | delivery_club | Mailer1 | 1 |
| 405 | delivery_club | Mailer1 | 0 |
| 435 | delivery_club | Mailer2 | 0 |

<br>
In the DataFrame we have:

* customer_id
* campaign name
* mailer_type (either Mailer1 or Mailer2)
* signup_flag (either 1 or 0)

___

<br>
# Applying Chi-Square Test For Independence <a name="chi-square-application"></a>

<br>
#### State Hypotheses & Acceptance Criteria For Test

The very first thing we need to do in any form of Hypothesis Test is stating our Null Hypothesis, our Alternate Hypothesis, and the Acceptance Criteria.

In the code below, I coded these in explcitly & clearly.  I specified the common Acceptance Criteria value of 0.05.

```python

# specify hypotheses & acceptance criteria for test
null_hypothesis = "There is no relationship between mailer type and signup rate.  They are independent"
alternate_hypothesis = "There is a relationship between mailer type and signup rate.  They are not independent"
acceptance_criteria = 0.05

```

<br>
#### Calculate Observed Frequencies & Expected Frequencies

As mentioned in the section above, in a Chi-Square Test For Independence, the *observed frequencies* are the true values that we’ve seen, in other words the actual rates per group in the data itself.  The *expected frequencies* are what we would *expect* to see based on *all* of the data combined.

The below code:

* Summarises my dataset to a 2x2 matrix for *signup_flag* by *mailer_type*
* Based on this, calculates the:
    * Chi-Square Statistic
    * p-value
    * Degrees of Freedom
    * Expected Values
* Prints out the Chi-Square Statistic & p-value from the test
* Calculates the Critical Value based upon our Acceptance Criteria & the Degrees Of Freedom
* Prints out the Critical Value

```python

# aggregate our data to get observed values
observed_values = pd.crosstab(campaign_data["mailer_type"], campaign_data["signup_flag"]).values

# run the chi-square test
chi2_statistic, p_value, dof, expected_values = chi2_contingency(observed_values, correction = False)

# print chi-square statistic
print(chi2_statistic)
>> 1.94

# print p-value
print(p_value)
>> 0.16

# find the critical value for our test
critical_value = chi2.ppf(1 - acceptance_criteria, dof)

# print critical value
print(critical_value)
>> 3.84

```
<br>
Based upon my observed values, I got:

* Mailer 1 (Low Cost): **32.8%** signup rate
* Mailer 2 (High Cost): **37.8%** signup rate

From this, I saw that the higher cost mailer led to a higher signup rate.  The results from my Chi-Square Test provided me more information about how confident I could be that this difference was robust, or if it might have occured by chance.

I got a Chi-Square Statistic of **1.94** and a p-value of **0.16**.  The critical value for my specified Acceptance Criteria of 0.05 was **3.84**

**Note** When applying the Chi-Square Test above, I used the parameter *correction = False* which meant we were applying what was known as the *Yate's Correction* which was applied when my Degrees of Freedom was equal to one.  This correction helped to prevent overestimation of statistical signficance in this case.

___

<br>
# Analysing The Results <a name="chi-square-results"></a>

At this point, I had everything I needed to understand the results of my Chi-Square test - and just from the results above I could see that, since my resulting p-value of **0.16** was *greater* than my Acceptance Criteria of 0.05 then I retained the Null Hypothesis and concluded that there was no significant difference between the signup rates of Mailer 1 and Mailer 2.

I made the same conclusion based upon my resulting Chi-Square statistic of **1.94** being _lower_ than my Critical Value of **3.84**

To make this script more dynamic, I created code to automatically interpret the results and explain the outcome to us...

```python

# print the results (based upon p-value)
if p_value <= acceptance_criteria:
    print(f"As our p-value of {p_value} is lower than our acceptance_criteria of {acceptance_criteria} - we reject the null hypothesis, and conclude that: {alternate_hypothesis}")
else:
    print(f"As our p-value of {p_value} is higher than our acceptance_criteria of {acceptance_criteria} - we retain the null hypothesis, and conclude that: {null_hypothesis}")

>> As our p-value of 0.16351 is higher than our acceptance_criteria of 0.05 - we retain the null hypothesis, and conclude that: There is no relationship between mailer type and signup rate.  They are independent


# print the results (based upon p-value)
if chi2_statistic >= critical_value:
    print(f"As our chi-square statistic of {chi2_statistic} is higher than our critical value of {critical_value} - we reject the null hypothesis, and conclude that: {alternate_hypothesis}")
else:
    print(f"As our chi-square statistic of {chi2_statistic} is lower than our critical value of {critical_value} - we retain the null hypothesis, and conclude that: {null_hypothesis}")
    
>> As our chi-square statistic of 1.9414 is lower than our critical value of 3.841458820694124 - we retain the null hypothesis, and conclude that: There is no relationship between mailer type and signup rate.  They are independent

```
<br>
As we can see from the outputs of these print statements, we do indeed retain the null hypothesis.  We could not find enough evidence that the signup rates for Mailer 1 and Mailer 2 were different - and thus conclude that there was no significant difference.

___

<br>
# Discussion <a name="discussion"></a>

While I saw that the higher cost Mailer 2 had a higher signup rate (37.8%) than the lower cost Mailer 1 (32.8%) it appeared that this difference was not significant, at least at my Acceptance Criteria of 0.05.

Without running this Hypothesis Test, the client may have concluded that they should always look to go with higher cost mailers - and from what I saw in this test, that may not be a great decision.  It would result in them spending more, but not *necessarily* gaining any extra revenue as a result

My results here also do not say that there *definitely isn't a difference between the two mailers* - I was only advising that we should not make any rigid conclusions *at this point*.  

Running more A/B Tests like this, gathering more data, and then re-running this test may provide me, and the client more insight!
