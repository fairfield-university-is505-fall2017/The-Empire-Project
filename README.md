# The Empire's Greatest Threat: Medical Appointment No Shows

## Business Applications
The Empire is facing a serious problem with Stormtroopers & Officers not turning up for scheduled medical appointments. This take up resources in the form of time wasted by losing a slot that a different Stromtroopere could have taken by not turning up. We have been tasked to come up with a logistic regression model that can be used to predict whether a person will make their appointment that is more accurate than the Force. Failure to do so will encur the wrath of the Emperor.

## Research Questions
* Number of appointments per day
* Number of troopers with pre existing conditions (% of pop)
* Average number of appointments made at different times throughout the day
* Investigate the relationship between neighbourhood and no-show
* Age, gender and pre existing conditions summary
* Examining relationship with neighbourhood, and pre existing condition/age/no show
* A logistic regression model that predicts how likely somebody is to make an appointment
* Investigate if Region can be a factor for not showing up (policies in area)

Section 1: Reading in the Data
In this section we will be reading in the data and setting up our table for futher analysis. such as fixing column names and ensuring our data is of the correct type. All this data was received directly from the Kaggle archives on the planet Eadu. The data set contains 15 variables and over 100K rows. This data was transmitted to our ship's computers and will be uploaded here for reproducability, using the following code:

~~~~{.python}
import numpy as np
import pandas as pd
import sklearn as sk
import seaborn as sb
import datetime as dt
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.cross_validation import cross_val_score
%matplotlib inline


#Reading in our data and checking the first 10 rows
medical_data=pd.read_csv('KaggleV2-May-2016.csv',header=0,parse_dates=['ScheduledDay','AppointmentDay'],na_values='',names=['PatientId','AppointmentID','Gender','ScheduledDay','AppointmentDay','Age','Neighbourhood','Scholarship','Hypertension','Diabetes','Alcoholism','Disabled','SMS Received','No-show' ])
~~~~

This produced the following table. Rather than print 100K rows, we printed the top ten to give an idea of what our data looks like, so that the reader can visualise easier how our data is stored. 

| Patient ID   | Appointment ID | Gender | Scheduled Day       | Appointment Day | Age | Neighbourhood     | Scholarship | Hypertension | Diabetes | Alcoholism | Disabled | SMS Received | No-SHow |
|--------------|----------------|--------|---------------------|-----------------|-----|-------------------|-------------|--------------|----------|------------|----------|--------------|---------|
| 2.987250e+13 | 5642903        | F      | 2016-04-29 18:38:08 | 2016-04-29      | 62  | JARDIM DA PENHA   | 0           | 1            | 0        | 0          | 0        | 0            | No      |
| 5.589978e+14 | 5642503        | M      | 2016-04-29 16:08:27 | 2016-04-29      | 56  | JARDIM DA PENHA   | 0           | 0            | 0        | 0          | 0        | 0            | No      |
| 4.262962e+12 | 5642549        | F      | 2016-04-29 16:19:04 | 2016-04-29      | 62  | MATA DA PRAIA     | 0           | 0            | 0        | 0          | 0        | 0            | No      |
| 8.679512e+11 | 5642828        | F      | 2016-04-29 17:29:31 | 2016-04-29      | 8   | PONTAL DE CAMBURI | 0           | 0            | 0        | 0          | 0        | 0            | No      |
| 8.841186e+12 | 5642494        | F      | 2016-04-29 16:07:23 | 2016-04-29      | 56  | JARDIM DA PENHA   | 0           | 1            | 1        | 0          | 0        | 0            | No      |
| 9.598513e+13 | 5626772        | F      | 2016-04-27 08:36:51 | 2016-04-29      | 76  | REPÚBLICA         | 0           | 1            | 0        | 0          | 0        | 0            | No      |
| 7.336882e+14 | 5630279        | F      | 2016-04-27 15:05:12 | 2016-04-29      | 23  | GOIABEIRAS        | 0           | 0            | 0        | 0          | 0        | 0            | Yes     |
| 3.449833e+12 | 5630575        | F      | 2016-04-27 15:39:58 | 2016-04-29      | 39  | GOIABEIRAS        | 0           | 0            | 0        | 0          | 0        | 0            | Yes     |
| 5.639473e+13 | 5638447        | F      | 2016-04-29 08:02:16 | 2016-04-29      | 21  | ANDORINHAS        | 0           | 0            | 0        | 0          | 0        | 0            | No      |
| 7.812456e+13 | 5629123        | F      | 2016-04-27 12:48:25 | 2016-04-29      | 19  | CONQUISTA         | 0           | 0            | 0        | 0          | 0        | 0            | No      |

 As we can see, we have the necessary fields to conduct some analysis on medical appointment no-shows. To begin with we will compare the number of people turning up vs the number of people not turning up for the medical appointments.
 
![Shows vs No shows](https://github.com/fairfield-university-is505-fall2017/health-stats-part-5-the-empire/blob/master/Medical%20Appointments/Graphs/No_of_app_over_time.png "Shows vs No-shows")

![Appointments vs Time](https://github.com/fairfield-university-is505-fall2017/health-stats-part-5-the-empire/blob/master/Medical%20Appointments/Graphs/Appointments_vs_Time.png "Appointments")

![No Show Appointments vs Time](https://github.com/fairfield-university-is505-fall2017/health-stats-part-5-the-empire/blob/master/Medical%20Appointments/Graphs/No-Show_Appointments_vs_Time.png "No show Appointments")
 
As we can see, the vast majority of people make their medical appointments. Notice for the most part, the number of appointments on a given day stays the same except for the **14th May, 2016**. After a brief investigation, it was discovered that this day is in fact Saturday. There was an increase in Chemo-therapy appointments at this time which might explain the number why doctors were taking appointments on this Saturday. 
As we cans see in the second diagram, around 8am is the most popular time to book a Doctors appointment. This is probabaly due to people waking and feeling ill. Additionally, we see another rise in the early afternoon. This could be due to people deteriorating over the course of a given day.
Finally, we can see the times appointments were made, and then the patient did not turn up. As expected it shares a similar shape to our second graph. A logical reason for this shape is that patients awoke feeling ill, but as time progressed, they began to improve before deciding not to attend their appointment.

The code for all of the previous graphs can be found [here](https://github.com/fairfield-university-is505-fall2017/health-stats-part-5-the-empire/blob/master/Medical%20Appointments/Generate_Time_Graphs.ipynb). 


## Logistic Regression Model
A Logistic Regression Model (LR) is a regression model where the dependent variable (DV) is categorical. This report covers the case of a binary dependent variable—that is, where the output can take only two values, "0" and "1", which represent outcomes such as pass/fail, win/lose, alive/dead or in our case, show/no-show. The first thing we need to do is to create a deep copy of our data set as we will be transformations to the data and want to keep the integrity of the original data. The code for this portion of the report is quite extense, you can find a well documented version of it [here](https://github.com/fairfield-university-is505-fall2017/health-stats-part-5-the-empire/blob/master/Medical%20Appointments/Logistic%20Regression%20.ipynb). 

An additional variable was created for the use in the LR model. Using the "AppointmentDay" and "ScheduledDay", we calculate the difference between the two, in days, and return the result. Ignoring the variables PatientID, AppointmentID, our date variables due to limitations in the package and replacing our "Gender" variable with a dummy variable we create our model. We create a heatmap scaling correlation from -1 to 1 to test for multicollinearity. Multicollinearity  is a phenomenon in which one predictor variable in a multiple regression model can be linearly predicted from the others with a substantial degree of accuracy.

![Heatmap](https://github.com/fairfield-university-is505-fall2017/health-stats-part-5-the-empire/blob/master/Medical%20Appointments/Graphs/Multicollinearity_check.png "Heatmap")

As we can see, we have some instances of high multicollinearity. Our variables "F" and "M" are highly correlated which makes sense as they are our two dummy variables created earlier. Additionally we see that "Hypertension" and "Diabetes" are highly correlated, along with "Difference" and "SMS Received". We exclude "M", "Hypertension" and "SMS Received" as they can be explained by their counterpart variable. We create our model with the following code

~~~~(.python}
# Assigning our independent and dependent variables
X = medical_dmy.loc[:,['Neighbourhood','Age','Difference','F','Scholarship','Alcoholism','Diabetes',]].values
y = medical_dmy.loc[:,'No-show'].values

# Split data in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)

# Fit a logistic regression model
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)

# Predict the DV using the test set
y_pred = LogReg.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)

# Printing our results
print(LogReg.score(X,y))
print(confusion_matrix)

print(classification_report(y_test, y_pred))
~~~~~

In order to truly our test our model, we adapted the Trial and Test methodology. This involves splitting our data into a "Trail" set which is used to generate the model and then a "Test" set to test its predicitve power on. Our model possessed a predictive power of 79%, however further examination using a confusion matrix, saw that this was achieved by assigning most observations in our test set to "0" i.e. turning up. We had 6,628 "No-shows" in our test set, but only 137 were correctly identified. So we successfully created a model, but its predictive power is lacking somewhat. We decided to examine this further by testing our model 10 times and investigating if the predicitve power remains the same. 

~~~~{.python}
scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
print (scores)
print (scores.mean())
~~~~

This returned an average of 79%, further allowing us to conclude our model's predictive power is 79%. Examining the Null error, we observed it to be around 20%.   

We also wanted to see if there was any trend in medical appointment no-shows for men vs. women. In order to look at this, we build a line graph displaying the number of show vs. no-show appointments for each gender:

![Male vs Female Shows vs No-shows](https://github.com/fairfield-university-is505-fall2017/The-Empire-Project/tree/master/Graphs/Male_vs_Female_Shows_vs_No-shows.png "Male vs Female Shows vs No-shows")

While women clearly make more medical appointments than men, both are more likely to show than not. Additionally, it seems that the number of appointments for men vs. women tends to follow the same general pattern.

Our next task was to look some views of medical apointments by neighbourhood. We first started by plotting the count of appointments by neighbourhood in a bar chart:

![Neighbourhood Appt Frequency](https://github.com/fairfield-university-is505-fall2017/The-Empire-Project/tree/master/Graphs/Neighbourhood_Appt_Frequency.png "Neighbourhood Appt Frequency")

Given the large number of neighbourhoods and variation of min and max appointments, we decided to only display the neighbourhoods with more than 2000 appointments in the total timeframe, displayed below:

![Top Neighbourhoods by Appt Frequency](https://github.com/fairfield-university-is505-fall2017/The-Empire-Project/tree/master/Graphs/Top_Neighbourhoods_by_Appt_Frequency.png "Top Neighbourhoods by Appt Frequency")

Limitting to the top neighbourhoods by number of appointments allowed for a much better visual. While the majority of the top neighbourhoods by number of appointments hover around 2-3k appointments, there are several neighbourhoods that seem to have abnormally high appointments in the timeframe. Jardim Camburi has more than double the number of appointments than the majority of other neighbourhoods.

Given the fact that Jardim Camburi is the neighbourhood with the most medical appointments, what is the show vs. no-show trend? Is the trend within the neighbourhood any different than the average trend?

![Jardim Camburi Shows vs No-shows](https://github.com/fairfield-university-is505-fall2017/The-Empire-Project/tree/master/Graphs/Jardim_Camburi.png "Jardim Camburi Shows vs No-shows")

Medical appointments in Jardim Camburi drop off towards the 3rd week of May, and then spike toward the beginning of June. It also seems that the number of no-shows in this neighbourhood tends to stabalize around the 22nd of may through the end of the time frame.

Now that we understand how medical appointments vary in terms of gender and neighbourhood, let's look for any trends by medical condition.

![Shows vs No-shows by Condition](https://github.com/fairfield-university-is505-fall2017/The-Empire-Project/tree/master/Graphs/Shows_vs_No-shows_by_Condition.png "Shows vs No-shows by Condition")

The highest appointment frequency by medical condition is hypertension, followed by diabetes. The rate of hypertension in this area seems to be very high. Individuals with alcoholism and/or are disabled are much less likely to schedule medical appointments.


The code for all of the previous graphs can be found [here](https://github.com/fairfield-university-is505-fall2017/health-stats-part-5-the-empire/blob/master/The-Empire-Project/MedicalAppts_DE.ipynb). 

