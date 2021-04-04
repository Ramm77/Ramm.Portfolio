# Ramm.Portfolio
Beispiel
# 1- Project 1
Risk Advisory Karriere @ Deloitte

import operator as opt
import numpy as np 
import pandas as pd 
import os
import gc
from contextlib import contextmanager
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import matplotlib
warnings.filterwarnings('ignore')
###############################################################################
os.chdir('D://3 Python//1 Data Camp//1 Credit Risk Modeling in Python//Data')
# D:\1 Data Camp\1 Credit Risk Modeling in Python\Data

cr_loan =  pd.read_csv('cr_loan2.csv')
cr_loan_clean  = pd.read_csv('cr_loan_nout_nmiss.csv')
cr_loan_prep = pd.read_csv('cr_loan_w2.csv')

###############################################################################
### 1 .Exploring and Preparing Loan Data

### Explore the credit data
""" Begin by looking at the data set cr_loan. In this data set, loan_status shows whether the loan is 
currently in default with  1 being default and 0 being non-default.

You have more columns within the data, and many could have a relationship with the values in loan_status. 
You need to explore the data and these relationships more with further analysis to understand the impact 
of the data on credit loan defaults.
Checking the structure of the data as well as seeing a snapshot helps us better understand what's inside
the set. 
Similarly, visualizations provide a high level view of the data in addition to important trends and patterns.
The data set cr_loan has already been loaded in the workspace.
Instructions 1/3
25 XP
Print the structure of the cr_loan data.
Look at the first five rows of the data."""

# Check the structure of the data
print(cr_loan.dtypes)

# Check the first five rows of the data
print(cr_loan.head())
##############################
# Look at the distribution of loan amounts with a histogram
n, bins, patches = plt.hist(x=cr_loan['loan_amnt'], bins='auto', color='blue', alpha=0.8, rwidth=0.85)
plt.xlabel("Loan Amount")
plt.show()

print("There are 32 000 rows of data so the scatter plot may take a little while to plot.")
#######################
# Plot a scatter plot of income against age
plt.scatter(cr_loan['person_income'], cr_loan['person_age'],c='blue', alpha=0.5)
plt.xlabel('Personal Income')
plt.ylabel('Persone Age')
plt.show()
'''
Great start! Starting with data exploration helps us keep from getting a.head() of ourselves! 
We can already see a positive correlation with age and income, which could mean these older recipients 
are further along in their career and therefore earn higher salaries. 
There also appears to be an outlier in the data.
'''
###############################################################################

### Crosstab and pivot tables
"""    Crosstab and pivot tables
Often, financial data is viewed as a pivot table in spreadsheets like Excel.

With cross tables, you get a high level view of selected columns and even aggregation like a count or average. 
For most credit risk models, especially for probability of default, columns like person_emp_length and person_home_ownership are common to
begin investigating.

You will be able to see how the values are populated throughout the data, and visualize them. For now, you need to check how 
loan_status is affected by factors like home ownership status, loan grade, and loan percentage of income.

The data set cr_loan has been loaded in the workspace.

Instructions 4/4

1- Create a boxplot of the loan's percent of the person's income grouped by loan_status.
2-  Create a cross table of home ownership grouped by loan_status and loan_grade.          
3- Create a cross table of home ownership, loan status, and average loan_percent_income.
4-  Create a boxplot of the loan's percent of the person's income grouped by loan_status.
   
"""
# Create a cross table of the loan intent and loan status
print(pd.crosstab(cr_loan['loan_intent'], cr_loan['loan_status'], margins = True))
p1 = pd.crosstab(cr_loan['loan_intent'], cr_loan['loan_status'], margins = True)

# Create a cross table of home ownership, loan status, and grade
print(pd.crosstab(cr_loan['person_home_ownership'],[cr_loan['loan_status'],cr_loan['loan_grade']]))
p2= pd.crosstab(cr_loan['person_home_ownership'],[cr_loan['loan_status'],cr_loan['loan_grade']])

# Create a cross table of home ownership, loan status, and average percent income
print(pd.crosstab(cr_loan['person_home_ownership'], cr_loan['loan_status'],
              values=cr_loan['loan_percent_income'], aggfunc='mean'))

p3= pd.crosstab(cr_loan['person_home_ownership'], cr_loan['loan_status'],
              values=cr_loan['loan_percent_income'], aggfunc='mean')
##########################

# Create a box plot of percentage income by loan status
cr_loan.boxplot(column = ['loan_percent_income'], by = 'loan_status')
plt.title('Average Percent Income by Loan Status')
plt.suptitle('')
plt.show()
'''
Nice work! It looks like the average percentage of income for defaults is higher. 
This could indicate those recipients have 
a debt-to-income ratio that's already too high.

'''
###############################################################################
###############################################################################

### Finding outliers with cross tables
"""  Now you need to find and remove outliers you suspect might be in the data. 
For this exercise, you can use cross tables and 
aggregate functions.

Have a look at the person_emp_length column. You've used the aggfunc = 'mean' argument to see the 
average of a numeric column before, but to detect outliers you can use other functions like min and max.

It may not be possible for a person to have an employment length of less than 0 or greater than 60. 
You can use cross tables to check the data and see if there are any instances of this!

The data set cr_loan has been loaded in the workspace.

Instructions 1/4

1- Print the cross table of loan_status and person_home_ownership with the max person_emp_length.
2- Create and array of indices for records with an employment length greater than 60. 
Store it as indices.
3- Drop the records from the data using the array indices and create a new dataframe called 
cr_loan_new.
4- Print the cross table from earlier, but instead use both min and max
          """
# Create the cross table for loan status, home ownership, and the max employment length
print(pd.crosstab(cr_loan['loan_status'],cr_loan['person_home_ownership'],
        values=cr_loan['person_emp_length'], aggfunc='max'))
########################

# Create an array of indices where employment length is greater than 60
indices = cr_loan[cr_loan['person_emp_length'] > 60].index
###################
# Drop the records from the data based on the indices and create a new dataframe
cr_loan_new = cr_loan.drop(indices)
#####################


# Create the cross table for loan status, home ownership, and the max employment length
print(pd.crosstab(cr_loan['loan_status'],cr_loan['person_home_ownership'],
                  values=cr_loan['person_emp_length'], aggfunc='max'))

# Create an array of indices where employment length is greater than 60
indices = cr_loan[cr_loan['person_emp_length'] > 60].index

# Drop the records from the data based on the indices and create a new dataframe
cr_loan_new = cr_loan.drop(indices)

# Create the cross table from earlier and include minimum employment length
print(pd.crosstab(cr_loan_new['loan_status'],cr_loan_new['person_home_ownership'],
            values=cr_loan_new['person_emp_length'], aggfunc=['min','max']))
'''
Off to a great start! Generally with credit data, key columns like person_emp_length are of high quality, 
but there is always room for error. With this in mind, we build our intuition for detecting outliers!
'''
###############################################################################

### Visualizing credit outliers
"""  You discovered outliers in person_emp_length where values greater than 60 were far above the norm. 
person_age is another column in which a person can use a common sense approach to say it is very unlikely 
that a person applying for a loan will be over 100 years old.

Visualizing the data here can be another easy way to detect outliers. You can use other numeric columns 
like loan_amnt and loan_int_rate to create plots with person_age to search for outliers.

The data set cr_loan has been loaded in the workspace.

Instructions 1/2

1 Create a scatter plot of person age on the x-axis and loan_amnt on the y-axis.
 
2 Use the .drop() method from Pandas to remove the outliers and create cr_loan_new.
Create a scatter plot of age on the x-axis and loan interest rate on the y-axis with a label for loan_status.          
""" 
# Create the scatter plot for age and amount
plt.scatter(cr_loan['person_age'], cr_loan['loan_amnt'], c='blue', alpha=0.5)
plt.xlabel("Person Age")
plt.ylabel("Loan Amount")
plt.show()
################
# Use Pandas to drop the record from the data frame and create a new one
cr_loan_new = cr_loan.drop(cr_loan[cr_loan['person_age'] > 100].index)

# Create a scatter plot of age and interest rate
colors = ["blue","red"]
plt.scatter(cr_loan_new['person_age'], cr_loan_new['loan_int_rate'],
            c = cr_loan_new['loan_status'],
            cmap = matplotlib.colors.ListedColormap(colors),
            alpha=0.5)
plt.xlabel("Person Age")
plt.ylabel("Loan Interest Rate")
plt.show()

'''
You really .drop() those rows like they're hot! Notice that in the last plot we have loan_status as a label
 for colors. This 
shows a different color depending on the class. In this case, it's loan default and non-default, and it 
looks like there are more defaults with high interest rates.
'''

#################################################################################

### Replacing missing credit data
"""  
Now, you should check for missing data. If you find missing data within loan_status, you would not be able
to use the data for predicting probability of default because you wouldn't know if the loan was a default 
or not. 
Missing data within person_emp_length would not be as damaging, but would still cause training errors.

So, check for missing data in the person_emp_length column and replace any missing values with the median.

The data set cr_loan has been loaded in the workspace.

Instructions
100 XP
Print an array of column names that contain missing data using .isnull().
Print the top five rows of the data set that has missing data for person_emp_length.
Replace the missing data with the median of all the employment length using .fillna().
Create a histogram of the person_emp_length column to check the distribution.
       """
# Print a null value column array
print(cr_loan.columns[cr_loan.isnull().any()])

# Print the top five rows with nulls for employment length
print(cr_loan[cr_loan['person_emp_length'].isnull()].head())

# Impute the null values with the median value for all employment lengths
cr_loan['person_emp_length'].fillna((cr_loan['person_emp_length'].median()), inplace=True)

# Create a histogram of employment length
n, bins, patches = plt.hist(cr_loan['person_emp_length'], bins='auto', color='blue')
plt.xlabel("Person Employment Length")
plt.show()
'''
Correct! You can use several different functions like mean() and median() to replace missing data. 
The goal here is to keep as much of our data as we can! It's also important to check the distribution of 
that feature to see if it changed.
'''
#################################################################################

### Removing missing data
""" 
You replaced missing data in person_emp_length, but in the previous exercise you saw that loan_int_rate 
has missing data as well.
Similar to having missing data within loan_status, having missing data within loan_int_rate will make 
predictions difficult.

Because interest rates are set by your company, having missing data in this column is very strange. 
It's possible that data ingestion issues created errors, but you cannot know for sure. 
For now, it's best to .drop() these records before moving forward.

The data set cr_loan has been loaded in the workspace.

Instructions
100 XP
Print the number of records that contain missing data for interest rate.
Create an array of indices for rows that contain missing interest rate called indices.
Drop the records with missing interest rate data and save the results to cr_loan_clean.
      """
# Print the number of nulls
print(cr_loan['loan_int_rate'].isnull().sum())

# Store the array on indices
indices = cr_loan[cr_loan['loan_int_rate'].isnull()].index

# Save the new data without missing data
cr_loan_clean = cr_loan.drop(indices)

'''
Nice! Now that the missing data and outliers have been processed, the data is ready for modeling! 
More often than not, financial data is fairly tidy, but it's always good to practice preparing data for 
analytical work.

'''
###############################################################################
###############################################################################
###############################################################################

# 2. Logistic Regression for Defaults

###############################################################################
###############################################################################
###############################################################################

### Logistic regression basics
"""    You've now cleaned up the data and created the new data set cr_loan_clean.
Think back to the final scatter plot from chapter 1 which showed more defaults with high loan_int_rate. Interest rates are easy 
to understand, but what how useful are they for predicting the probability of default?

Since you haven't tried predicting the probability of default yet, test out creating and training a logistic regression model 
with just loan_int_rate. Also check the model's internal parameters, which are like settings, to see the structure of the model 
with this one column.

The data cr_loan_clean has already been loaded in the workspace.

Instructions
100 XP
Create the X and y sets using the loan_int_rate and loan_status columns.
Create and fit a logistic regression model to the training data and call it clf_logistic_single.
Print the parameters of the model with .get_params().
Check the intercept of the model with the .intercept_ attribute.
    """ 
from sklearn.linear_model import LogisticRegression
# Create the X and y data sets
X = cr_loan_clean[['loan_int_rate']]
y = cr_loan_clean[['loan_status']]

# Create and fit a logistic regression model
clf_logistic_single = LogisticRegression()
clf_logistic_single.fit(X, np.ravel(y))

# Print the parameters of the model
print(clf_logistic_single.get_params())

# Print the intercept of the model
print(clf_logistic_single.intercept_)

'''
You're on your way to making predictions! Notice that the model was able to fit to the data, and establish some parameters 
internally. It's even produced an .intercept_ value like in the lecture. What if you use more than one column?

'''
###############################################################################

### Multivariate logistic regression
""" 
Generally, you won't use only loan_int_rate to predict the probability of default. You will want to use all
the data you have to make predictions.

With this in mind, try training a new model with different columns, called features, from the 
cr_loan_cleandata. 
Will this model differ from the first one? For this, you can easily check the .intercept_ of the 
logistic regression. 
Remember that this is the  y-intercept of the function and the overall log-odds of non-default.

The cr_loan_clean data has been loaded in the workspace along with the previous model clf_logistic_single.

Instructions
100 XP
Create a new X data set with loan_int_rate and person_emp_length. Store it as X_multi.
Create a y data set with just loan_status.
Create and .fit() a LogisticRegression() model on the new X data. Store it as clf_logistic_multi.
Print the .intercept_ value of the model      """ 
# Create X data for the model
X_multi = cr_loan_clean[['loan_int_rate','person_emp_length']]

# Create a set of y data for training
y = y[['loan_status']]

# Create and train a new logistic regression
clf_logistic_multi = LogisticRegression(solver='lbfgs').fit(X_multi, np.ravel(y))

# Print the intercept of the model
print(clf_logistic_multi.intercept_)

'''
Great! Take a closer look at each model's .intercept_ value. The values have changed! The new 
clf_logistic_multi model has an .intercept_ value closer to zero. 
This means the log odds of a non-default is approaching zero.


'''

###############################################################################
"""      
You've just trained LogisticRegression() models on different columns.

You know that the data should be separated into training and test sets. 
test_train_split() is used to create both at the same time. 
The training set is used to make predictions, while the test set is used for evaluation. 
Without evaluating the model, you have no way to tell how well it will perform on new loan data.

In addition to the intercept_, which is an attribute of the model, LogisticRegression() models also have 
the .coef_ attribute. 
This shows how important each training column is for predicting the probability of default.

The data set cr_loan_clean is already loaded in the workspace.

Instructions
100 XP
Create the data set X using interest rate, employment length, and income. Create the y set using loan status.
Use train_test_split() to create the training and test sets from X and y.
Create and train a LogisticRegression() model and store it as clf_logistic.
Print the coefficients of the model using .coef_.
           """
### Creating training and test sets
from sklearn.model_selection import train_test_split
# Create the X and y data sets
X = cr_loan_clean[['loan_int_rate','person_emp_length','person_income']]
y = cr_loan_clean[['loan_status']]

# Use test_train_split to create the training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=123)

# Create and fit the logistic regression model
clf_logistic = LogisticRegression(solver='lbfgs').fit(X_train, np.ravel(y_train))

# Print the models coefficients
print(clf_logistic.coef_)
'''
Nicely done! Do you see that three columns were used for training and there are three values in .coef_? 
This tells you how important each column, or feature, was for predicting. 
The more positive the value, the more it predicts defaults. Look at the value for loan_int_rate.


'''
################################################################################
### Changing coefficients

""" 
With this understanding of the coefficients of a LogisticRegression() model, have a closer look at them to
 see how they change depending on what columns are used for training. 
 Will the column coefficients change from model to model?

You should .fit() two different LogisticRegression() models on different groups of columns to check. 
You should also consider what the potential impact on the probability of default might be.

The data set cr_loan_clean has already been loaded into the workspace along with the training sets 
X1_train, X2_train, and y_train.

Instructions
100 XP
Check the first five rows of both X training sets.
Train a logistic regression model, called clf_logistic1, with the X1 training set.
Train a logistic regression model, called clf_logistic2, with the X2 training set.
Print the coefficients for both logistic regression models.
      """
X1_train = cr_loan_clean[['person_income', 'person_emp_length', 'loan_amnt']]
X2_train = cr_loan_clean[['person_income', 'loan_percent_income', 'cb_person_cred_hist_length']]
y_train = cr_loan_clean[['loan_status']]
# Print the first five rows of each training set
print(X1_train.head())
print(X2_train.head())

# Create and train a model on the first training data
clf_logistic1 = LogisticRegression(solver='lbfgs').fit(X1_train, np.ravel(y_train))

# Create and train a model on the second training data
clf_logistic2 = LogisticRegression(solver='lbfgs').fit(X2_train, np.ravel(y_train))

# Print the coefficients of each model
print(clf_logistic1.coef_)
print(clf_logistic2.coef_)

###############################################################################

### One-hot encoding credit data
"""       
It's time to prepare the non-numeric columns so they can be added to your LogisticRegression() model.

Once the new columns have been created using one-hot encoding, you can concatenate them with the numeric 
columns to create a new data frame which will be used throughout the rest of the course for 
predicting probability of default.

Remember to only one-hot encode the non-numeric columns. Doing this to the numeric columns would create an
 incredibly wide data set!

The credit loan data, cr_loan_clean, has already been loaded in the workspace.

Instructions
100 XP
Create a data set for all the numeric columns called cred_num and one for the non-numeric columns called 
cred_str.
Use one-hot encoding on cred_str to create a new data set called cred_str_onehot.
Union cred_num with the new one-hot encoded data and store the results as cr_loan_prep.
Print the columns of the new data set.
      """
      
# Create two data sets for numeric and non-numeric data
cred_num = cr_loan_clean.select_dtypes(exclude=['object'])
cred_str = cr_loan_clean.select_dtypes(include=['object'])

# One-hot encode the non-numeric columns
cred_str_onehot = pd.get_dummies(cred_str)

# Union the one-hot encoded columns to the numeric ones
cr_loan_prep = pd.concat([cred_num, cred_str_onehot], axis=1)

# Print the columns in the new data set
print(cr_loan_prep.columns)
'''
Interesting! 

Notice that the coefficient for the person_income changed when we changed the data from X1 to X2. 
This is a reason to keep most of the data like we did in chapter 1, because the models will 
learn differently depending on what data they're given!


'''

################################################################################

### Predicting probability of default
"""        
All of the data processing is complete and it's time to begin creating predictions for probability of 
default. 
You want to train a LogisticRegression() model on the data, and examine how it predicts the 
probability of default.

So that you can better grasp what the model produces with predict_proba, you should look at an example 
record alongside the predicted probability of default. 
How do the first five predictions look against the actual values of loan_status?

The data set cr_loan_prep along with X_train, X_test, y_train, and y_test have already been loaded in 
the workspace.

Instructions
100 XP
Train a logistic regression model on the training data and store it as clf_logistic.
Use predict_proba() on the test data to create the predictions and store them in preds.
Create two data frames, preds_df and true_df, to store the first five predictions and true loan_status 
values.
Print the true_df and preds_df as one set using .concat().
      """ 
X = cr_loan_prep[['loan_int_rate','person_emp_length','person_income']]
y = cr_loan_prep[['loan_status']]

# Use test_train_split to create the training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=123)

# Train the logistic regression model on the training data
clf_logistic = LogisticRegression(solver='lbfgs').fit(X_train, np.ravel(y_train))

# Create predictions of probability for loan status using test data
preds = clf_logistic.predict_proba(X_test)
preds
# Create dataframes of first five predictions, and first five true labels
preds_df = pd.DataFrame(preds[:,1][0:5], columns = ['prob_default'])
true_df = y_test.head()

# Concatenate and print the two data frames for comparison
print(pd.concat([true_df.reset_index(drop = True), preds_df], axis = 1))
'''
Look at all those columns! If you've ever seen a credit scorecard, the column_name_value format should 
look familiar. If you haven't seen one, look up some pictures during your next break!


'''
################################################################################

### Default classification reporting
"""         
It's time to take a closer look at the evaluation of the model. Here is where setting the threshold for 
probability of default will help you analyze the model's performance through classification reporting.

Creating a data frame of the probabilities makes them easier to work with, because you can use all the 
power of pandas. 
Apply the threshold to the data and check the value counts for both classes of loan_status to see how 
many predictions of each are being created. This will help with insight into the scores from the 
classification report.

The cr_loan_prep data set, trained logistic regression clf_logistic, true loan status values y_test, 
and predicted probabilities, preds are loaded in the workspace.

Instructions
100 XP
Create a data frame of just the probabilities of default from preds called preds_df.
Reassign loan_status values based on a threshold of 0.50 for probability of default in preds_df.
Print the value counts of the number of rows for each loan_status.
Print the classification report using y_test and preds_df.
    """
from sklearn.metrics import classification_report, precision_recall_fscore_support

# Create a dataframe for the probabilities of default
preds_df = pd.DataFrame(preds[:,1], columns = ['prob_default'])

# Reassign loan status based on the threshold
preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.50 else 0)

# Print the row counts for each loan status
print(preds_df['loan_status'].value_counts())

# Print the classification report
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, preds_df['loan_status'], target_names=target_names))

'''
Well isn't this a surprise! It looks like almost all of our test set was predicted to be 
non-default. 
The recall for defaults is 0.16 meaning 16% of our true defaults were predicted correctly.


'''
'''
Neat! We have some predictions now, but they don't look very accurate do they? It looks like most of the 
rows with loan_status at 1 have a low probability of default. How good are the rest of the predictions? 
Next, let's see if we can determine how 
accurate the entire model is.


'''

###############################################################################


### Selecting report metrics
"""    
The classification_report() has many different metrics within it, but you may not always want to print out
the full report. Sometimes you just want specific values to compare models or use for other purposes.

There is a function within scikit-learn that pulls out the values for you. That function is 
precision_recall_fscore_support() and it takes in the same parameters as classification_report.

It is imported and used like this:

# Import function
from sklearn.metrics import precision_recall_fscore_support
# Select all non-averaged values from the report
precision_recall_fscore_support(y_true,predicted_values)
The cr_loan_prep data set along with the predictions in preds_df have already been loaded in the workspace.

Instructions 1/3
35 XP
1 Print the classification report for y_test and predicted loan status.
2 Print all the non-average values using precision_recall_fscore_support().
3 Print only the first two values from the report by subsetting the string with [].
      """
# Print the classification report
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, preds_df['loan_status'], target_names=target_names))

''' Print all the non-average values using precision_recall_fscore_support().
'''
# Print all the non-average values from the report
print(precision_recall_fscore_support(y_test,preds_df['loan_status']))
''' Print only the first two values from the report by subsetting the string with [].
''' 
# Print the first two numbers from the report
print(precision_recall_fscore_support(y_test,preds_df['loan_status'])[:2])

'''
Great! Now we know how to pull out specific values from the report to either store later for comparison, 
or use to check against portfolio performance. Remember the impact of recall for defaults? 
This way, you can store that value for later calculations.


'''
###############################################################################

"""                """
### Visually scoring credit models
'''
Now, you want to visualize the performance of the model. In ROC charts, the X and Y axes are two metrics 
you've already looked at: the false positive rate (fall-out), and the true positive rate (sensitivity).

You can create a ROC chart of it's performance with the following code:

fallout, sensitivity, thresholds = roc_curve(y_test, prob_default)
plt.plot(fallout, sensitivity)
To calculate the AUC score, you use roc_auc_score().

The credit data cr_loan_prep along with the data sets X_test and y_test have all been loaded into the 
workspace. A trained LogisticRegression() model named clf_logistic has also been loaded into the workspace.

Instructions
100 XP
Create a set of predictions for probability of default and store them in preds.
Print the accuracy score the model on the X and y test sets.
Use roc_curve() on the test data and probabilities of default to create fallout and sensitivity Then, create a ROC curve plot with fallout on the x-axis.
Compute the AUC of the model using test data and probabilities of default and store it in auc.
Take Hint (-30 XP)

'''
from sklearn.metrics import roc_curve, roc_auc_score

# Create predictions and store them in a variable
preds = clf_logistic.predict_proba(X_test)

# Print the accuracy score the model
print(clf_logistic.score(X_test, y_test))

# Plot the ROC curve of the probabilities of default
prob_default = preds[:, 1]
fallout, sensitivity, thresholds = roc_curve(y_test, prob_default)
plt.plot(fallout, sensitivity, color = 'darkorange')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.show()

# Compute the AUC and store it in a variable
auc = roc_auc_score(y_test, prob_default)
auc
'''
I wasn't worried about your .score() on this exercise! So the accuracy for this model is about 80% and 
the AUC score is 76%. 
Notice that what the ROC chart shows us is the tradeoff between all values of our false positive rate 
(fallout) and true positive rate (sensitivity).


'''
###############################################################################

### Thresholds and confusion matrices.
"""   
You've looked at setting thresholds for defaults, but how does this impact overall performance? 
To do this, you can start by looking at the effects with confusion matrices.

Recall the confusion matrix as shown here:

Set different values for the threshold on probability of default, and use a confusion matrix to see how 
the changing values affect the model's performance.

The data frame of predictions, preds_df, as well as the model clf_logistic have been loaded in the workspace.

Instructions 1/3
35 XP
1 Reassign values of loan_status using a threshold of 0.5 for probability of default within preds_df.
Print the confusion matrix of the y_test data and the new loan status values.
2 Reassign values of loan_status using a threshold of 0.4 for probability of default within preds_df.
Print the confusion matrix of the y_test data and the new loan status values.

3 Based on the confusion matrices you just created, calculate the default recall for each. Using these values, answer the following: which threshold gives us the highest value for default recall?
       """

from sklearn.metrics import confusion_matrix
# Set the threshold for defaults to 0.5
preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.5 else 0)

# Print the confusion matrix
print(confusion_matrix(y_test,preds_df['loan_status']))
###############################

# Set the threshold for defaults to 0.4
preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.4 else 0)

# Print the confusion matrix
print(confusion_matrix(y_test,preds_df['loan_status']))
########################
'''
Correct! The value for default recall at this threshold is actually pretty high! You can check out the 
non-default recalls as well to see how the threshold affected those values.


'''

###############################################################################
"""       """
cr_loan_prep.columns


###############################################################################
### How thresholds affect performance
"""        
Setting the threshold to 0.4 shows promising results for model evaluation. Now you can assess the financial
impact using the default recall which is selected from the classification reporting using the function 
precision_recall_fscore_support().

For this, you will estimate the amount of unexpected loss using the default recall to find what proportion
of defaults you did not catch with the new threshold. This will be a dollar amount which tells you how 
 much in losses you would have if all the unfound defaults were to default all at once.

The average loan value, avg_loan_amnt has been calculated and made available in the workspace along with 
preds_df and y_test.

Instructions
100 XP
Reassign the loan_status values using the threshold 0.4.
Store the number of defaults in preds_df by selecting the second value from the value counts and store it 
as num_defaults.
Get the default recall rate from the classification matrix and store it as default_recall
Estimate the unexpected loss from the new default recall by multiplying 1 - default_recall by the average 
loan amount and number of default loans.
   """
   
avg_loan_amnt = cr_loan_prep['loan_amnt'].sum() / cr_loan_prep['loan_amnt'].count()
# Reassign the values of loan status based on the new threshold
preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.4 else 0)

# Store the number of loan defaults from the prediction data
num_defaults = preds_df['loan_status'].value_counts()[1]

# Store the default recall from the classification report
default_recall = precision_recall_fscore_support(y_test,preds_df['loan_status'])[1][1]

# Calculate the estimated impact of the new default recall rate
print(num_defaults * avg_loan_amnt * (1 - default_recall))

'''
Nice Job! By our estimates, this loss would be around $9.8 million. That seems like a lot! Try rerunning 
this code with threshold 
values of 0.3 and 0.5. Do you see the estimated losses changing? How do we find a good threshold value 
based on these metrics alone?


'''

###############################################################################
### Threshold selection

"""       
You know there is a trade off between metrics like default recall, non-default recall, and model
accuracy.
One easy way to approximate a good starting threshold value is to look at a plot of all three 
using matplotlib. With this graph, you can see how each of these metrics look as you change the
threshold  values and find the point at which the performance of all three is good enough to 
use for the credit data.

The threshold values thresh, default recall values def_recalls, the non-default recall values 
nondef_recalls and the accuracy scores accs have been loaded into the workspace. To make the 
plot easier to read, the array ticks for x-axis tick marks has been loaded as well.

Instructions 1/2
50 XP
1 Plot the graph of thresh for the x-axis then def_recalls, non-default recall values, and 
accuracy scores on each y-axis.

2 Have a closer look at this plot. In fact, expand the window to get a really good look. Think 
about the threshold values from thresh and how they affect each of these three metrics. 
Approximately what starting threshold value would maximize these scores evenly?

         """ 

thresh = [0.2,0.225,0.25,0.275,0.3,0.325,0.35,0.375,0.4,0.425,0.45, 0.475,0.5,0.525,0.55,0.575,0.6,0.625,00.65]
def_recalls = [0.7981438515081206,
 0.7583139984532096,
 0.7157772621809745,
 0.6759474091260634,
 0.6349574632637278,
 0.594354215003867,
 0.5467904098994586,
 0.5054137664346481,
 0.46403712296983757,
 0.39984532095901004,
 0.32211910286156226,
 0.2354988399071926,
 0.16782675947409126,
 0.1148491879350348,
 0.07733952049497293,
 0.05529775715390565,
 0.03750966744006187,
 0.026295436968290797,
 0.017788089713843776]
nondef_recalls = [0.5342465753424658,
 0.5973037616873234,
 0.6552511415525114,
 0.708306153511633,
 0.756468797564688,
 0.8052837573385518,
 0.8482278756251359,
 0.8864970645792564,
 0.9215046749293324,
 0.9492280930637095,
 0.9646662317895195,
 0.9733637747336378,
 0.9809741248097412,
 0.9857577734290063,
 0.9902152641878669,
 0.992280930637095,
 0.9948901935203305,
 0.9966297021091541,
 0.997499456403566]
accs =[0.5921588594704684,
 0.6326374745417516,
 0.6685336048879837,
 0.7012050237610319,
 0.7298031228784793,
 0.7589952477936185,
 0.7820773930753564,
 0.8028682959945689,
 0.8211133740665308,
 0.8286659877800407,
 0.8236591989137814,
 0.811439239646979,
 0.8025288526816021,
 0.7946367956551256,
 0.7898845892735913,
 0.7866598778004074,
 0.7847929395790902,
 0.7836897488119484,
 0.7825016972165648]
ticks = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65] 


plt.plot(thresh,def_recalls)
plt.plot(thresh,nondef_recalls)
plt.plot(thresh,accs)
plt.xlabel("Probability Threshold")
plt.xticks(ticks)
plt.legend(["Default Recall","Non-default Recall","Model Accuracy"])
plt.show()

'''
Yes! This is the easiest pattern to see on this graph, because it's the point where all three lines converge. 
This threshold would make a great starting point, but declaring all loans about 0.275 to be a default is probably not practical.


'''
###############################################################################
###############################################################################
###############################################################################

# 3. Gradient Boosted Trees Using XGBoost

###############################################################################
###############################################################################
###############################################################################

### Trees for defaults
"""      
You will now train a gradient boosted tree model on the credit data, and see a sample of some of
the predictions. Do you remember when you first looked at the predictions of the logistic regression 
model? They didn't look good. Do you think this model be different?

The credit data cr_loan_prep, the training sets X_train and y_train, and the test data X_test is
available in the workspace. The XGBoost package is loaded as xgb.

Instructions
100 XP
Create and train a gradient boosted tree using XGBClassifier() and name it clf_gbt.
Predict probabilities of default on the test data and store the results in gbt_preds.
Create two data frames, preds_df and true_df, to store the first five predictions and true loan_status values.
Concatenate and print the data frames true_df and preds_df in order, and check the model's results.
           """
# Train a model
import xgboost as xgb
clf_gbt = xgb.XGBClassifier().fit(X_train, np.ravel(y_train))

# Predict with a model
gbt_preds = clf_gbt.predict_proba(X_test)
gbt_preds
# Create dataframes of first five predictions, and first five true labels
preds_df = pd.DataFrame(gbt_preds[:,1][0:5], columns = ['prob_default'])
true_df = y_test.head()

# Concatenate and print the two data frames for comparison
print(pd.concat([true_df.reset_index(drop = True), preds_df], axis = 1))

'''
Interesting! The predictions don't look the same as with the LogisticRegression(), do they? Notice that this model is already 
accurately predicting the probability of default for some loans with a true value of 1 in loan_status.


'''



###############################################################################

### Gradient boosted portfolio performance
"""  
At this point you've looked at predicting probability of default using both a LogisticRegression() 
and XGBClassifier(). 
You've looked at some scoring and have seen samples of the predictions, but what is the overall affect on 
portfolio performance? 
Try using expected loss as a scenario to express the importance of testing different models.

A data frame called portfolio has been created to combine the probabilities of default for both models, 
the loss given default (assume 20% for now), and the loan_amnt which will be assumed to be the 
exposure at default.

The data frame cr_loan_prep along with the X_train and y_train training sets have been loaded into 
the workspace.

Instructions
100 XP
Print the first five rows of portfolio.
Create the expected_loss column for the gbt and lr model named gbt_expected_loss and lr_expected_loss.
Print the sum of lr_expected_loss for the entire portfolio.
Print the sum of gbt_expected_loss for the entire portfolio.
       """
# Print the first five rows of the portfolio data frame
print(portfolio.head())

# Create expected loss columns for each model using the formula
portfolio['gbt_expected_loss'] = portfolio['gbt_prob_default'] * portfolio['lgd'] * portfolio['loan_amnt']
portfolio['lr_expected_loss'] = portfolio['lr_prob_default'] * portfolio['lgd'] * portfolio['loan_amnt']

# Print the sum of the expected loss for lr
print('LR expected loss: ', np.sum(portfolio['lr_expected_loss']))

# Print the sum of the expected loss for gbt
print('GBT expected loss: ', np.sum(portfolio['gbt_expected_loss']))

'''
Great! It looks like the total expected loss for the XGBClassifier() model is quite a bit lower. When we talk about accuracy and 
precision, the goal is to generate models which have a low expected loss. Looking at a classification_report() helps as well.


'''

###############################################################################
### Assessing gradient boosted trees
''' So you've now used XGBClassifier() models to predict probability of default. These models can also use
 the .predict() method for creating predictions that give the actual class for loan_status.

You should check the model's initial performance by looking at the metrics from the classification_report().
 Keep in mind that you have not set thresholds for these models yet.

The data sets cr_loan_prep, X_test, and y_test have already been loaded in the workspace. 
The model clf_gbt has been loaded as well. The classification_report() for the logistic regression will 
print automatically.

Instructions
100 XP
Predict the loan_status values for the X test data and store them in gbt_preds.
Check the contents of gbt_preds to see predicted loan_status values not probabilities of default.
Print a classification_report() of the model's performance against y_test.
'''
# Predict the labels for loan status
gbt_preds = clf_gbt.predict(X_test)

# Check the values created by the predict method
print(gbt_preds)

# Print the classification report of the model
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, gbt_preds, target_names=target_names))

'''
Wow. Have a look at the precision and recall scores! Remember the low default recall values we were getting from the 
LogisticRegression()? This model already appears to have serious potential.


'''


#######################

#### Column importance and default prediction
''' When using multiple training sets with many different groups of columns, it's important to keep and eye on which columns matter and which do not. It can be expensive or time-consuming to maintain a set of columns even though they might not have any impact on loan_status.

The X data for this exercise was created with the following code:

X = cr_loan_prep[['person_income','loan_int_rate',
                  'loan_percent_income','loan_amnt',
                  'person_home_ownership_MORTGAGE','loan_grade_F']]
Train an XGBClassifier() model on this data, and check the column importance to see how each one performs to predict loan_status.

The cr_loan_pret data set along with X_train and y_train have been loaded in the workspace.

Instructions
100 XP
Create and train a XGBClassifier() model on the X_train and y_train training sets and store it as clf_gbt.
Print the column importances for the columns in clf_gbt by using .get_booster() and .get_score().
'''

# Create and train the model on the training data
clf_gbt = xgb.XGBClassifier().fit(X_train,np.ravel(y_train))

# Print the column importances from the model
print(clf_gbt.get_booster().get_score(importance_type = 'weight'))
'''
That's how you do it! So, the importance for loan_grade_F is only 6 in this case. This could be because there are so few of the 
F-grade loans. While the F-grade loans don't add much to predictions here, they might affect the importance of other training 
columns.


'''

################################################################################

# Visualizing column importance
'''When the model is trained on different sets of columns it changes the performance, but does the importance for the same column change depending on which group it's in?

The data sets X2 and X3 have been created with the following code:

X2 = cr_loan_prep[['loan_int_rate','person_emp_length']]
X3 = cr_loan_prep[['person_income','loan_int_rate','loan_percent_income']]
Understanding how different columns are used to arrive at a loan_status prediction is very important for model interpretability.

The data sets cr_loan_prep, X2_train, X2_test, X3_train, X3_test, y_train, y_test are loaded in the workspace.

Instructions 1/2
0 XP
1 Create and train a XGBClassifier() model on X2_train and call it clf_gbt2.
Plot the column importances for the columns that clf_gbt2 trained on.

2Create and train another XGBClassifier() model on X3_train and call it clf_gbt3.
Plot the column importances for the columns that clf_gbt3 trained on.

'''
# Train a model on the X data with 2 columns
clf_gbt2 = xgb.XGBClassifier().fit(X2_train,np.ravel(y_train))

# Plot the column importance for this model
xgb.plot_importance(clf_gbt2, importance_type = 'weight')
plt.show()

''' Create and train another XGBClassifier() model on X3_train and call it clf_gbt3.
Plot the column importances for the columns that clf_gbt3 trained on.
'''

# Train a model on the X data with 3 columns
clf_gbt3 = xgb.XGBClassifier().fit(X3_train,np.ravel(y_train))

# Plot the column importance for this model
xgb.plot_importance(clf_gbt3, importance_type = 'weight')
plt.show()


'''
Very good! Take a closer look at the plots. Did you notice that the importance of loan_int_rate went from 456 to 195? 
Initially, this was the most important column, but person_income ended up taking the top spot here.


'''
###############################################################################


### Column selection and model performance
''' Creating the training set from different combinations of columns affects the model and the importance values of the columns. Does a different selection of columns also affect the F-1 scores, the combination of the precision and recall, of the model? You can answer this question by training two different models on two different sets of columns, and checking the performance.

Inaccurately predicting defaults as non-default can result in unexpected losses if the probability of default for these loans was very low. You can use the F-1 score for defaults to see how the models will accurately predict the defaults.

The credit data, cr_loan_prep and the two training column sets X and X2 have been loaded in the workspace. The models gbt and gbt2 have already been trained.

Instructions 1/2
50 XP
1
2
Use both gbt and gbt2 to predict loan_status and store the values in gbt_preds and gbt2_preds.
Print the classification_report() of the first model.
Print the classification_report() of the second model.
'''

# Predict the loan_status using each model
gbt_preds = gbt.predict(X_test)
gbt2_preds = gbt2.predict(X2_test)

# Print the classification report of the first model
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, gbt_preds, target_names=target_names))

# Print the classification report of the second model
print(classification_report(y_test, gbt2_preds, target_names=target_names))
'''
You're right! Originally, it looked like the selection of columns affected model accuracy the most, but now we see that the 
selection of columns also affects recall by quite a bit.


'''
###################

# Cross validating credit models
'''
Credit loans and their data change over time, and it won't always look like what's been loaded into the current test sets. 
So, you can use cross-validation to try several smaller training and test sets which are derived from the original X_train and y_train.

Use the XGBoost function cv() to perform cross-validation. You will need to set up all the parameters for cv() to use on the test data.

The data sets X_train, y_train are loaded in the workspace along with the trained model gbt, and the parameter dictionary params which will 
print once the exercise loads.

Instructions
0 XP
Set the number of folds to 5 and the stopping to 10. Store them as n_folds and early_stopping.
Create the matrix object DTrain using the training data.
Use cv() on the parameters, folds, and early stopping objects. Store the results as cv_df.
Print the contents of cv_df.
'''

# Set the values for number of folds and stopping iterations
n_folds = 5
early_stopping = 10

# Create the DTrain matrix for XGBoost
DTrain = xgb.DMatrix(X_train, label = y_train)

# Create the data frame of cross validations
cv_df = xgb.cv(params, DTrain, num_boost_round = 5, nfold=n_folds,
            early_stopping_rounds=early_stopping)

# Print the cross validations data frame
print(cv_df)

'''
Looks good! Do you see how the AUC for both train-auc-mean and test-auc-mean improves at each iteration of cross-validation? 
As the iterations progress the scores get better, but will they eventually reach 1.0?


'''
##########################

# Limits to cross-validation testing
'''You can specify very large numbers for both nfold and num_boost_round if you want to perform an extreme amount of cross-validation. The data frame cv_results_big has already been loaded in the workspace and was created with the following code:

cv = xgb.cv(params, DTrain, num_boost_round = 600, nfold=10,
            shuffle = True)
Here, cv() performed 600 iterations of cross-validation! The parameter shuffle tells the function to shuffle the records each time.

Have a look at this data to see what the AUC are, and check to see if they reach 1.0 using cross validation. You should also plot the test AUC score to see the progression.

The data frame cv_results_big has been loaded into the workspace.

Instructions
100 XP
Print the first five rows of the CV results data frame.
Print the average of the test set AUC from the CV results data frame rounded to two places.
Plot a line plot of the test set AUC over the course of each iteration.
'''

# Print the first five rows of the CV results data frame
print(cv_results_big.head())

# Calculate the mean of the test AUC scores
print(np.mean(cv_results_big['test-auc-mean']).round(2))

# Plot the test AUC scores for each iteration
plt.plot(cv_results_big['test-auc-mean'])
plt.title('Test AUC Score Over 600 Iterations')
plt.xlabel('Iteration Number')
plt.ylabel('Test AUC Score')
plt.show()
'''
Excellent! Notice that the test AUC score never quite reaches 1.0 and begins to decrease slightly after 100 iterations. This is 
because this much cross-validation can actually cause the model to become overfit. So, there is a limit to how much 
cross-validation you should to.


'''

############################

# Cross-validation scoring
'''Now, you should use cross-validation scoring with cross_val_score() to check the overall performance.

This is exercise presents an excellent opportunity to test out the use of the hyperparameters learning_rate and max_depth. Remember, hyperparameters are like settings which can help create optimum performance.

The data sets cr_loan_prep, X_train, and y_train have already been loaded in the workspace.

Instructions
100 XP
Create a gradient boosted tree with a learning rate of 0.1 and a max depth of 7. Store the model as gbt.
Calculate the cross validation scores against the X_train and y_train data sets with 4 folds. Store the results as cv_scores.
Print the cross validation scores.
Print the average accuracy score and standard deviation with formatting.
'''


# Create a gradient boosted tree model using two hyperparameters
gbt = xgb.XGBClassifier(learning_rate = 0.1, max_depth = 7)

# Calculate the cross validation scores for 4 folds
cv_scores = cross_val_score(gbt, X_train, np.ravel(y_train), cv = 4)

# Print the cross validation scores
print(cv_scores)

# Print the average accuracy and standard deviation of the scores
print("Average accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(),
                                              cv_scores.std() * 2))
'''
Your average cv_score for this course is getting higher! With only a couple of hyperparameters and cross-validation, we can get 
the average accuracy up to 93%. This is a great way to validate how robust the model is.


'''
############

### Undersampling training data
'''
X_y_train, count_nondefault, and count_default are already loaded in the workspace. They have been created using the following code:

X_y_train = pd.concat([X_train.reset_index(drop = True),
                       y_train.reset_index(drop = True)], axis = 1)
count_nondefault, count_default = X_y_train['loan_status'].value_counts()
The .value_counts() for the original training data will print automatically.

Instructions
100 XP
Create data sets of non-defaults and defaults stored as nondefaults and defaults.
Sample the nondefaults to the same number as count_default and store it as nondefaults_under.
Concatenate nondefaults and defaults using .concat() and store it as X_y_train_under.
Print the .value_counts() of loan status for the new data set.
'''
# Create data sets for defaults and non-defaults
nondefaults = X_y_train[X_y_train['loan_status'] == 0]
defaults = X_y_train[X_y_train['loan_status'] == 1]

# Undersample the non-defaults
nondefaults_under = nondefaults.sample(count_default)

# Concatenate the undersampled nondefaults with defaults
X_y_train_under = pd.concat([nondefaults_under.reset_index(drop = True),
                             defaults.reset_index(drop = True)], axis = 0)

# Print the value counts for loan status
print(X_y_train_under['loan_status'].value_counts())

'''
Piece of cake! Now, our training set has an even number of defaults and non-defaults. Let's test out some machine learning models
 on this new undersampled data set and compare their performance to the models trained on the regular data set.



'''
####################################################

### Undersampled tree performance
''''You've undersampled the training set and trained a model on the undersampled set.

The performance of the model's predictions not only impact the probability of default on the test set, 
but also on the scoring of new loan applications as they come in. You also now know that it is even more 
important that the recall of defaults be high, because a default predicted as non-default is more costly.

The next crucial step is to compare the new model's performance to the original model. The original 
predictions are stored as gbt_preds and the new model's predictions stored as gbt2_preds.

The model predictions gbt_preds and gbt2_preds are already stored in the workspace in addition to y_test.

Instructions 1/3
35 XP
1 Print the classification_report() for both the old model and new model.

2Print a confusion_matrix() of the old and new model predictions.

3Print the roc_auc_score of the new model and old model.

'''

# Check the classification reports
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, gbt_preds, target_names=target_names))
print(classification_report(y_test, gbt2_preds, target_names=target_names))

# Print the confusion matrix for both old and new models
print(confusion_matrix(y_test,gbt_preds))
print(confusion_matrix(y_test,gbt2_preds))



# Print and compare the AUC scores of the old and new models
print(roc_auc_score(y_test, gbt_preds))
print(roc_auc_score(y_test, gbt2_preds))


'''
Undersampling intuition
Intuition check again! Now you've seen the effects of undersampling the training set to improve default 
prediction. You undersampled the training data set X_train, and it had a positive impact on the new 
model's AUC score and recall for defaults. The training data had class imbalance which is normal for 
most credit loan data.

You did not undersample the test data X_test. Why not undersample the test set as well?

You should not undersample the test set because it will make the test set unrealistic.
'''

'''
Looks like this is classified as a success! Undersampling the training data results in more false 
positives, but the recall for defaults and the AUC score are both higher than the original model. 
This means overall it predicts defaults much more accurately.


'''
###############################################################################
###############################################################################
###############################################################################

# 4. Model Evaluation and Implementation

###############################################################################
###############################################################################
###############################################################################

# Comparing model reports
'''
You've used logistic regression models and gradient boosted trees. It's time to compare these two to see 
which model will be used to make the final predictions.

One of the easiest first steps for comparing different models' ability to predict the probability of default
is to look at their metrics from the classification_report(). With this, you can see many different scoring
metrics side-by-side for each model. Because the data and models are normally unbalanced with few defaults,
focus on the metrics for defaults for now.

The trained models clf_logistic and clf_gbt have been loaded into the workspace along with their predictions
preds_df_lr and preds_df_gbt. A cutoff of 0.4 was used for each. 
The test set y_test is also available.

Instructions
100 XP
Print the classification_report() for the logistic regression predictions.
Print the classification_report() for the gradient boosted tree predictions.
Print the macro average of the F-1 Score for the logistic regression using precision_recall_fscore_support().
Print the macro average of the F-1 Score for the gradient boosted tree using precision_recall_fscore_support().
'''

# Print the logistic regression classification report
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, preds_df_lr['loan_status'], target_names=target_names))

# Print the gradient boosted tree classification report
print(classification_report(y_test, preds_df_gbt['loan_status'], target_names=target_names))

# Print the default F-1 scores for the logistic regression
print(precision_recall_fscore_support(y_test,preds_df_lr['loan_status'], average = 'macro')[2])

# Print the default F-1 scores for the gradient boosted tree
print(precision_recall_fscore_support(y_test,preds_df_gbt['loan_status'], average = 'macro')[2])

'''
Great! There is a noticeable difference between these two models. 
Do you see that the scores from the classification_report() are all higher for the gradient boosted tree? 
This means the tree model is better in all of these aspects. 
Let's check the ROC curve.


'''
#######################################

'''
Comparing with ROCs You should use ROC charts and AUC scores to compare the two models. 
Sometimes, visuals can really help you and potential business users understand the differences 
between the various models under consideration.

With the graph in mind, you will be more equipped to make a decision. 
The lift is how far the curve is  from the random prediction. 
The AUC is the area between the curve and the random prediction. 
The model with more lift, and a higher AUC, is the one that's better at making predictions accurately.

The trained models clf_logistic and clf_gbt have been loaded into the workspace. 
The predictions for the probability of default clf_logistic_preds and clf_gbt_preds have been loaded as 
well.

Instructions 1/2
50 XP
1 
Calculate the fallout, sensitivity, and thresholds for the logistic regression and gradient boosted tree.
Plot the ROC chart for the lr then gbt using the fallout on the x-axis and sensitivity on the y-axis for each model.
2
Print the AUC for the logistic regression.
Print the AUC for the gradient boosted tree.



'''


# ROC chart components
fallout_lr, sensitivity_lr, thresholds_lr = roc_curve(y_test, clf_logistic_preds)
fallout_gbt, sensitivity_gbt, thresholds_gbt = roc_curve(y_test, clf_gbt_preds)

# ROC Chart with both
plt.plot(fallout_lr, sensitivity_lr, color = 'blue', label='%s' % 'Logistic Regression')
plt.plot(fallout_gbt, sensitivity_gbt, color = 'green', label='%s' % 'GBT')
plt.plot([0, 1], [0, 1], linestyle='--', label='%s' % 'Random Prediction')
plt.title("ROC Chart for LR and GBT on the Probability of Default")
plt.xlabel('Fall-out')
plt.ylabel('Sensitivity')
plt.legend()
plt.show()

################################
# Print the logistic regression AUC with formatting
print("Logistic Regression AUC Score: %0.2f" % roc_auc_score(y_test, clf_logistic_preds))

# Print the gradient boosted tree AUC with formatting
print("Gradient Boosted Tree AUC Score: %0.2f" % roc_auc_score(y_test, clf_gbt_preds))
############
'''
Wow! Look at the ROC curve for the gradient boosted tree. Not only is the lift much higher, the calculated
AUC score is also quite a bit higher. It's beginning to look like the gradient boosted tree is best. 
Let's check the calibration to be sure.
'''

###############################################################################

'''
Calibration curves You now know that the gradient boosted tree clf_gbt has the best overall performance. 
You need to check the calibration of the two models to see how stable the default prediction performance is
 across probabilities. 
You can use a chart of each model's calibration to check this by calling the calibration_curve() function.

Calibration curves can require many lines of code in python, so you will go through each step slowly to add
 the different components.

The two sets of predictions clf_logistic_preds and clf_gbt_preds have already been loaded into the 
workspace. Also, the output from calibration_curve() for each model has been loaded as: 
 frac_of_pos_lr, mean_pred_val_lr, frac_of_pos_gbt, and mean_pred_val_gbt.

Instructions 1/3
35 XP
1 Create a calibration curve plot() by starting with the perfect calibration guideline and label it 
'Perfectly calibrated'. Then add the labels for the y-axis and x-axis in order.

2 Add a plot of the mean predicted values on the x-axis and fraction of positives on the y-axis for the 
logistic regression model to the plot of the guideline. Label this 'Logistic Regression'.

3 Finally, add a plot of the mean predicted values on the x-axis and fraction of positives on the 
y-axis for the gradient boosted tree to the plot. Label this 'Gradient Boosted tree'.


'''
# Create the calibration curve plot with the guideline
plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')    
plt.ylabel('Fraction of positives')
plt.xlabel('Average Predicted Probability')
plt.legend()
plt.title('Calibration Curve')
plt.show()

###############
# Add the calibration curve for the logistic regression to the plot
plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')    
plt.plot(mean_pred_val_lr, frac_of_pos_lr,
         's-', label='%s' % 'Logistic Regression')
plt.ylabel('Fraction of positives')
plt.xlabel('Average Predicted Probability')
plt.legend()
plt.title('Calibration Curve')
plt.show()

#############

# Add the calibration curve for the gradient boosted tree
plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')    
plt.plot(mean_pred_val_lr, frac_of_pos_lr,
         's-', label='%s' % 'Logistic Regression')
plt.plot(mean_pred_val_gbt, frac_of_pos_gbt,
         's-', label='%s' % 'Gradient Boosted Tree')
plt.ylabel('Fraction of positives')
plt.xlabel('Average Predicted Probability')
plt.legend()
plt.title('Calibration Curve')
plt.show()
####################

'''
Great job! 
Expand the plot window and take a good look at this. Notice that for the logistic regression, the 
calibration for probabilities starts off great but then gets more erratic as it the average probability 
approaches 0.4. Something similar happens to the gradient boosted tree around 0.5, 
but the model eventually stabilizes. We will be focusing on only the gbt model  from now on.


'''

#################################################################################
# Acceptance rates
'''
Setting an acceptance rate and calculating the threshold for that rate can be used to set the percentage 
of new loans you want to accept. For this exercise, assume the test data is a fresh batch of new loans. 
You will need to use the quantile() function from numpy to calculate the threshold.

The threshold should be used to assign new loan_status values. Does the number of defaults and non-defaults
 in the data change?

The trained model clf_gbt and the data frame of it's predictions, test_pred_df, are available.

Instructions
100 XP
Print the summary statistics of prob_default within the data frame of predictions using .describe().
Calculate the threshold for a 85% acceptance rate using quantile() and store it as threshold_85.
Create a new column called pred_loan_status based on threshold_85.
Print the value counts of the new values in pred_loan_status.
'''
# Check the statistics of the probabilities of default
print(test_pred_df['prob_default'].describe())

# Calculate the threshold for a 85% acceptance rate
threshold_85 = np.quantile(test_pred_df['prob_default'], 0.85)

# Apply acceptance rate threshold
test_pred_df['pred_loan_status'] = test_pred_df['prob_default'].apply(lambda x: 1 if x > threshold_85 else 0)

# Print the counts of loan status after the threshold
print(test_pred_df['pred_loan_status'].value_counts())


'''
Excellent! 

In the results of .describe() do you see how it's not until 75% that you start to see double-digit numbers? 
That's because the majority of our test set is non-default loans. Next let's look at how the acceptance rate and threshold split 
up the data.


'''

###############################################################################
'''Visualizing quantiles of acceptance
You know how quantile() works to compute a threshold, and you've seen an example of what it does to split the loans into accepted
 and rejected. What does this threshold look like for the test set, and how can you visualize it?

To check this, you can create a histogram of the probabilities and add a reference line for the threshold. With this, you can 
visually show where the threshold exists in the distribution.

The model predictions clf_gbt_preds have been loaded into the workspace.

Instructions
100 XP
Create a histogram of the predicted probabilities clf_gbt_preds.
Calculate the threshold for an acceptance rate of 85% using quantile(). Store this value as threshold.
Plot the histogram again, except this time add a reference line using .axvline().

'''


# Plot the predicted probabilities of default
plt.hist(clf_gbt_preds, color = 'blue', bins = 40)

# Calculate the threshold with quantile
threshold = np.quantile(clf_gbt_preds, 0.85)

# Add a reference line to the plot for the threshold
plt.axvline(x = threshold, color = 'red')
plt.show()

'''
Very nice! Here, you can see where the threshold is on the range of predicted probabilities. Not only can you see how many loans 
will be accepted (left side), but also how many loans will be rejected (right side). I recommend that you re-run this code with 
different threshold values to better understand how this affects the acceptance rate.


'''

#################################################################################


# Bad rates
''' With acceptance rate in mind, you can now analyze the bad rate within the accepted loans. This way you will be able to see the percentage of defaults that have been accepted.

Think about the impact of the acceptance rate and bad rate. We set an acceptance rate to have fewer defaults in the portfolio because defaults are more costly. Will the bad rate be less than the percentage of defaults in the test data?

The predictions data frame test_pred_df has been loaded into the workspace.

Instructions
100 XP
Print the first five rows of the predictions data frame.
Create a subset called accepted_loans which only contains loans where the predicted loan status is 0.
Calculate the bad rate based on true_loan_status of the subset using sum() and .count().

'''

# Print the top 5 rows of the new data frame
print(test_pred_df.head())

# Create a subset of only accepted loans
accepted_loans = test_pred_df[test_pred_df['pred_loan_status'] == 0]

# Calculate the bad rate
print(np.sum(accepted_loans['true_loan_status']) / accepted_loans['true_loan_status'].count())

'''
This bad rate doesn't look half bad! The bad rate with the threshold set by the 85% quantile() is about 8%. This means that of all
 the loans we've decided to accept from the test set, only 8% were actual defaults! If we accepted all loans, the percentage of 
 defaults would be around 22%.


'''
################################################################################

# Acceptance rate impact
''' Now, look at the loan_amnt of each loan to understand the impact on the portfolio for the acceptance rates. You can use cross
 tables with calculated values, like the average loan amount, of the new set of loans X_test. For this, you will multiply the 
 number of each with an average loan_amnt value.

When printing these values, try formatting them as currency so that the numbers look more realistic. After all, credit risk is 
all about money. This is accomplished with the following code:

pd.options.display.float_format = '${:,.2f}'.format
The predictions data frame test_pred_df, which now includes the loan_amnt column from X_test, has been loaded in the workspace.

Instructions
100 XP
Print the summary statistics of the loan_amnt column using .describe().
Calculate the average value of loan_amnt and store it as avg_loan.
Set the formatting for pandas to '${:,.2f}'
Print the cross table of the true loan status and predicted loan status multiplying each by avg_loan.
'''
# Print the statistics of the loan amount column
print(test_pred_df['loan_amnt'].describe())

# Store the average loan amount
avg_loan = np.mean(test_pred_df['loan_amnt'])

# Set the formatting for currency, and print the cross tab
pd.options.display.float_format = '${:,.2f}'.format
print(pd.crosstab(test_pred_df['true_loan_status'],
                 test_pred_df['pred_loan_status_15']).apply(lambda x: x * avg_loan, axis = 0))

##########################################

'''Nice! With this, we can see that our bad rate of about 8% represents an estimated loan value of about 7.9 million dollars. 
This may seem like a lot at first, but compare it to the total value of non-default loans! With this, we are ready to start 
talking about our acceptance strategy going forward.

'''
###############################################################################

## Making the strategy table
''' Before you implement a strategy, you should first create a strategy table containing all the possible acceptance rates you wish to look at along with their associated bad rates and threshold values. This way, you can begin to see each part of your strategy and how it affects your portfolio.

Automatically calculating all of these values only requires a for loop, but requires many lines of python code. Don't worry, most of the code is already there. Remember the calculations for threshold and bad rate.

The array accept_rates has already been populated and loaded into the workspace along with the data frames preds_df_gbt and test_pred_df. The arrays thresholds and bad_rates have not been populated.

Instructions 1/3
35 XP
1 Print the contents of accept_rates.

2 Populate the arrays thresholds and bad_rates using a for loop. Calculate the threshold thresh, and store it in thresholds. Then reassign the loan_status values using thresh. After that, Create accepted_loans where loan_status is 0.

3 Create the strategy table as a data frame and call it strat_df.
Print the contents of strat_df.


''' 

# Print accept rates
print(accept_rates)
################

# Populate the arrays for the strategy table with a for loop
for rate in accept_rates:
  	# Calculate the threshold for the acceptance rate
    thresh = np.quantile(preds_df_gbt['prob_default'], rate).round(3)
    # Add the threshold value to the list of thresholds
    thresholds.append(np.quantile(preds_df_gbt['prob_default'], rate).round(3))
    # Reassign the loan_status value using the threshold
    test_pred_df['pred_loan_status'] = test_pred_df['prob_default'].apply(lambda x: 1 if x > thresh else 0)
    # Create a set of accepted loans using this acceptance rate
    accepted_loans = test_pred_df[test_pred_df['pred_loan_status'] == 0]
    # Calculate and append the bad rate using the acceptance rate
    bad_rates.append(np.sum((accepted_loans['true_loan_status']) / len(accepted_loans['true_loan_status'])).round(3))
#################

# Create a data frame of the strategy table
strat_df = pd.DataFrame(zip(accept_rates, thresholds, bad_rates),
                        columns = ['Acceptance Rate','Threshold','Bad Rate'])

# Print the entire table
print(strat_df)
########################

'''
Excellent! That for loop was a lot of code, but look at this sweet strategy table we have now. This uses our specific predictions
 on the credit data, and can be used to see the acceptance rates, bad rates, and financial impact all at once. One of these values
 has the highest estimated value.


'''
###############################################################################

# Visualizing the strategy
'''Now you have the extended strategy table strat_df. The table is not so big that it's difficult to analyze, but visuals can help you see the overview all at once.

You should check at the distribution of each column with a box plot. If the distribution of Acceptance Rate looks the same as the Bad Rate column, that could be a problem. That means that the model's calibration is likely much worse than you thought.

You can also visualize the strategy curve with a line plot. The Acceptance Rate would be the independent variable with the Bad Rate as the dependent variable.

The strategy table strat_df has been loaded in the workspace.

Instructions 1/2
50 XP
1 Create a simple boxplot of the values within strat_df using the pandas boxplot method.

2 Create a line plot of the acceptance rates on the x-axis and bad rates on the y-axis with a title(), xlabel(), and ylabel().

'''


# Visualize the distributions in the strategy table with a boxplot
strat_df.boxplot()
plt.show()
###########################

# Plot the strategy curve
plt.plot(strat_df['Acceptance Rate'], strat_df['Bad Rate'])
plt.xlabel('Acceptance Rate')
plt.ylabel('Bad Rate')
plt.title('Acceptance and Bad Rates')
plt.axes().yaxis.grid()
plt.axes().xaxis.grid()
plt.show()
'''
Good work! The boxplot shows us the distribution for each column. Look at the strategy curve. The bad rates are very low up until
 the acceptance rate 0.6 where they suddenly increase. This suggests that many of the accepted defaults may have 
 a prob_default value between 0.6 and 0.8.


'''
###############################################################################

# Estimated value profiling
''' The strategy table, strat_df, can be used to maximize the estimated portfolio value and minimize expected loss. Extending this table and creating some plots can be very helpful to this end.

The strat_df data frame is loaded and has been enhanced already with the following columns:

Column	               Description
Num Accepted Loans	   The number of accepted loans based on the threshold
Avg Loan Amnt	       The average loan amount of the entire test set
Estimated value	       The estimated net value of non-defaults minus defaults

Instructions 
1 Check the contents of the new strat_df by printing the entire data frame.
2 Create a line plot of the acceptance rate on the x-axis and estimated value from strat_df on the y-axis with axis labels for both x and y.

3 Print the row with the highest 'Estimated Value' from strat_df.


'''
# Print the contents of the strategy df
print(strat_df)
##############
# Create a line plot of estimated value
plt.plot(strat_df['Acceptance Rate'],strat_df['Estimated Value'])
plt.title('Estimated Value by Acceptance Rate')
plt.xlabel('Acceptance Rate')
plt.ylabel('Estimated Value')
plt.axes().yaxis.grid()
plt.show()

############

# Print the row with the max estimated value
print(strat_df.loc[strat_df['Estimated Value'] == np.max(strat_df['Estimated Value'])])
########

'''
Interesting! With our credit data and our estimated averag loan value, we clearly see that the acceptance rate 0.85 has the highest potential estimated value. Normally, the allowable bad rate is set, but we can use analyses like this to explore other options.



'''

################################################################################

# Total expected loss
''' It's time to estimate the total expected loss given all your decisions. The data frame test_pred_df has the probability of default for each loan and that loan's value. Use these two values to calculate the expected loss for each loan. Then, you can sum those values and get the total expected loss.



For this exercise, you will assume that the exposure is the full value of the loan, and the loss given default is 100%. This means that a default on each the loan is a loss of the entire amount.

The data frame test_pred_df has been loaded into the workspace.

Instructions
100 XP
Print the top five rows of test_pred_df.
Create a new column expected_loss for each loan by using the formula above.
Calculate the total expected loss of the entire portfolio, rounded to two decimal places, and store it as tot_exp_loss.
Print the total expected loss.
'''
# Print the first five rows of the data frame
print(test_pred_df.head())

# Calculate the bank's expected loss and assign it to a new column
test_pred_df['expected_loss'] = test_pred_df['prob_default'] * test_pred_df['loan_amnt'] * test_pred_df['loss_given_default']

# Calculate the total expected loss to two decimal places
tot_exp_loss = round(np.sum(test_pred_df['expected_loss']),2)

# Print the total expected loss
print('Total expected loss: ', '${:,.2f}'.format(tot_exp_loss))

'''
Very nice! This is the total expected loss for the entire portfolio using the gradient boosted tree. $27 million may seem like 
a lot, but the total expected loss would have been over $28 million with the logistic regression. 
Some losses are unavoidable, but your work here might have saved the company a million dollars!


'''


# 2- Projekt 2

Hi
