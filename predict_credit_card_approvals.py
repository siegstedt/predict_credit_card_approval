
# coding: utf-8

# ## 1. Credit card applications
# <p>Commercial banks receive <em>a lot</em> of applications for credit cards. Many of them get rejected for many reasons, like high loan balances, low income levels, or too many inquiries on an individual's credit report, for example. Manually analyzing these applications is mundane, error-prone, and time-consuming (and time is money!). Luckily, this task can be automated with the power of machine learning and pretty much every commercial bank does so nowadays. In this notebook, we will build an automatic credit card approval predictor using machine learning techniques, just like the real banks do.</p>
# <p><img src="https://assets.datacamp.com/production/project_558/img/credit_card.jpg" alt="Credit card being held in hand"></p>
# <p>We'll use the <a href="http://archive.ics.uci.edu/ml/datasets/credit+approval">Credit Card Approval dataset</a> from the UCI Machine Learning Repository. The structure of this notebook is as follows:</p>
# <ul>
# <li>First, we will start off by loading and viewing the dataset.</li>
# <li>We will see that the dataset has a mixture of both numerical and non-numerical features, that it contains values from different ranges, plus that it contains a number of missing entries.</li>
# <li>We will have to preprocess the dataset to ensure the machine learning model we choose can make good predictions.</li>
# <li>After our data is in good shape, we will do some exploratory data analysis to build our intuitions.</li>
# <li>Finally, we will build a machine learning model that can predict if an individual's application for a credit card will be accepted.</li>
# </ul>
# <p>First, loading and viewing the dataset. We find that since this data is confidential, the contributor of the dataset has anonymized the feature names.</p>

# In[1]:


# Import pandas
import pandas as pd

# explore working directory
get_ipython().system('ls')
get_ipython().system('ls datasets')
get_ipython().system('head -n 5 datasets/cc_approvals.data')

# Load dataset
cc_apps = pd.read_csv("datasets/cc_approvals.data", header=None)

# Inspect data
cc_apps.head()


# In[184]:


get_ipython().run_cell_magic('nose', '', 'import pandas as pd\n\ndef test_cc_apps_exists():\n    assert "cc_apps" in globals(), \\\n        "The variable cc_apps should be defined."\n        \ndef test_cc_apps_correctly_loaded():\n    correct_cc_apps = pd.read_csv("datasets/cc_approvals.data", header=None)\n    try:\n        pd.testing.assert_frame_equal(cc_apps, correct_cc_apps)\n    except AssertionError:\n        assert False, "The variable cc_apps should contain the data as present in datasets/cc_approvals.data."')


# ## 2. Inspecting the applications
# <p>The output may appear a bit confusing at its first sight, but let's try to figure out the most important features of a credit card application. The features of this dataset have been anonymized to protect the privacy, but <a href="http://rstudio-pubs-static.s3.amazonaws.com/73039_9946de135c0a49daa7a0a9eda4a67a72.html">this blog</a> gives us a pretty good overview of the probable features. The probable features in a typical credit card application are <code>Gender</code>, <code>Age</code>, <code>Debt</code>, <code>Married</code>, <code>BankCustomer</code>, <code>EducationLevel</code>, <code>Ethnicity</code>, <code>YearsEmployed</code>, <code>PriorDefault</code>, <code>Employed</code>, <code>CreditScore</code>, <code>DriversLicense</code>, <code>Citizen</code>, <code>ZipCode</code>, <code>Income</code> and finally the <code>ApprovalStatus</code>. This gives us a pretty good starting point, and we can map these features with respect to the columns in the output.   </p>
# <p>As we can see from our first glance at the data, the dataset has a mixture of numerical and non-numerical features. This can be fixed with some preprocessing, but before we do that, let's learn about the dataset a bit more to see if there are other dataset issues that need to be fixed.</p>

# In[2]:


# Print summary statistics
cc_apps_description = cc_apps.describe()
print(cc_apps_description)

print("\n")

# Print DataFrame information
cc_apps_info = cc_apps.info()
print(cc_apps_info)

print("\n")

# Inspect missing values in the dataset
cc_apps_tail = cc_apps.tail(17)
print(cc_apps_tail)


# In[186]:


get_ipython().run_cell_magic('nose', '', '\ndef test_cc_apps_description_exists():\n    assert "cc_apps_description" in globals(), \\\n        "The variable cc_apps_description should be defined."\n\ndef test_cc_apps_description_correctly_done():\n    correct_cc_apps_description = cc_apps.describe()\n    assert str(correct_cc_apps_description) == str(cc_apps_description), \\\n        "cc_apps_description should contain the output of cc_apps.describe()."\n    \ndef test_cc_apps_info_exists():\n    assert "cc_apps_info" in globals(), \\\n        "The variable cc_apps_info should be defined."\n\ndef test_cc_apps_info_correctly_done():\n    correct_cc_apps_info = cc_apps.info()\n    assert str(correct_cc_apps_info) == str(cc_apps_info), \\\n        "cc_apps_info should contain the output of cc_apps.info()."')


# ## 3. Handling the missing values (part i)
# <p>We've uncovered some issues that will affect the performance of our machine learning model(s) if they go unchanged:</p>
# <ul>
# <li>Our dataset contains both numeric and non-numeric data (specifically data that are of <code>float64</code>, <code>int64</code> and <code>object</code> types). Specifically, the features 2, 7, 10 and 14 contain numeric values (of types float64, float64, int64 and int64 respectively) and all the other features contain non-numeric values.</li>
# <li>The dataset also contains values from several ranges. Some features have a value range of 0 - 28, some have a range of 2 - 67, and some have a range of 1017 - 100000. Apart from these, we can get useful statistical information (like <code>mean</code>, <code>max</code>, and <code>min</code>) about the features that have numerical values. </li>
# <li>Finally, the dataset has missing values, which we'll take care of in this task. The missing values in the dataset are labeled with '?', which can be seen in the last cell's output.</li>
# </ul>
# <p>Now, let's temporarily replace these missing value question marks with NaN.</p>

# In[3]:


# Import numpy
import numpy as np

# Inspect missing values in the dataset
print(cc_apps_tail)

# Replace the '?'s with NaN
cc_apps = cc_apps.replace(to_replace='?', value=np.nan)

# Inspect the missing values again
print(cc_apps.tail(17))


# In[188]:


get_ipython().run_cell_magic('nose', '', '\n# def test_cc_apps_assigned():\n#     assert "cc_apps" in globals(), \\\n#         "After the NaN replacement, it should be assigned to the same variable cc_apps only."\n\ndef test_cc_apps_correctly_replaced():\n    cc_apps_fresh = pd.read_csv("datasets/cc_approvals.data", header=None)\n    correct_cc_apps_replacement = cc_apps_fresh.replace(\'?\', np.NaN)\n    string_cc_apps_replacement = cc_apps_fresh.replace(\'?\', "NaN")\n#     assert cc_apps.to_string() == correct_cc_apps_replacement.to_string(), \\\n#         "The code that replaces question marks with NaNs doesn\'t appear to be correct."\n    try:\n        pd.testing.assert_frame_equal(cc_apps, correct_cc_apps_replacement)\n    except AssertionError:\n        if string_cc_apps_replacement.equals(cc_apps):\n            assert False, "It looks like the question marks were replaced by the string \\"NaN\\". Missing values should be represented by `np.nan`."\n        else:\n            assert False, "The variable cc_apps should contain the data in datasets/cc_approvals.data."')


# ## 4. Handling the missing values (part ii)
# <p>We replaced all the question marks with NaNs. This is going to help us in the next missing value treatment that we are going to perform.</p>
# <p>An important question that gets raised here is <em>why are we giving so much importance to missing values</em>? Can't they be just ignored? Ignoring missing values can affect the performance of a machine learning model heavily. While ignoring the missing values our machine learning model may miss out on information about the dataset that may be useful for its training. Then, there are many models which cannot handle missing values implicitly such as LDA. </p>
# <p>So, to avoid this problem, we are going to impute the missing values with a strategy called mean imputation.</p>

# In[4]:


# Impute the missing values with mean imputation
cc_apps.fillna(cc_apps.mean(), inplace=True)

# Count the number of NaNs in the dataset to verify
cc_apps.isnull().sum()


# In[190]:


get_ipython().run_cell_magic('nose', '', '\ndef test_cc_apps_correctly_imputed():\n    assert cc_apps.isnull().values.sum() == 67, \\\n        "There should be 67 null values after your code is run, but there aren\'t."')


# ## 5. Handling the missing values (part iii)
# <p>We have successfully taken care of the missing values present in the numeric columns. There are still some missing values to be imputed for columns 0, 1, 3, 4, 5, 6 and 13. All of these columns contain non-numeric data and this why the mean imputation strategy would not work here. This needs a different treatment. </p>
# <p>We are going to impute these missing values with the most frequent values as present in the respective columns. This is <a href="https://www.datacamp.com/community/tutorials/categorical-data">good practice</a> when it comes to imputing missing values for categorical data in general.</p>

# In[5]:


# Iterate over each column of cc_apps
for col in cc_apps:
    # Check if the column is of object type
    if cc_apps[col].dtypes == 'object':
        # Impute with the most frequent value
        cc_apps = cc_apps.fillna(cc_apps[col].value_counts().index[0])

# Count the number of NaNs in the dataset and print the counts to verify
cc_apps.isnull().sum()


# In[192]:


get_ipython().run_cell_magic('nose', '', '\ndef test_cc_apps_correctly_imputed():\n    assert cc_apps.isnull().values.sum() == 0, \\\n        "There should be 0 null values after your code is run, but there isn\'t."')


# ## 6. Preprocessing the data (part i)
# <p>The missing values are now successfully handled.</p>
# <p>There is still some minor but essential data preprocessing needed before we proceed towards building our machine learning model. We are going to divide these remaining preprocessing steps into three main tasks:</p>
# <ol>
# <li>Convert the non-numeric data into numeric.</li>
# <li>Split the data into train and test sets. </li>
# <li>Scale the feature values to a uniform range.</li>
# </ol>
# <p>First, we will be converting all the non-numeric values into numeric ones. We do this because not only it results in a faster computation but also many machine learning models (like XGBoost) (and especially the ones developed using scikit-learn) require the data to be in a strictly numeric format. We will do this by using a technique called <a href="http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html">label encoding</a>.</p>

# In[6]:


# Import LabelEncoder
from sklearn import preprocessing

# Instantiate LabelEncoder
le = preprocessing.LabelEncoder()

# Iterate over all the values of each column and extract their dtypes
for col in cc_apps:
    # Compare if the dtype is object
    if cc_apps[col].dtypes=='object':
    # Use LabelEncoder to do the numeric transformation
        le.fit(cc_apps[col])
        cc_apps[col]=le.transform(cc_apps[col])
        
# Check for data types
cc_apps.info()


# In[194]:


get_ipython().run_cell_magic('nose', '', '\ndef test_le_exists():\n    assert "le" in globals(), \\\n        "The variable le should be defined."\n\ndef test_label_encoding_done_correctly():\n    for cols in cc_apps.columns:\n        if np.issubdtype(cc_apps[col].dtype, np.number) != True:\n            assert "It doesn\'t appear that all of the non-numeric columns were converted to numeric using fit_transform."')


# ## 7. Splitting the dataset into train and test sets
# <p>We have successfully converted all the non-numeric values to numeric ones.</p>
# <p>Now, we will split our data into train set and test set to prepare our data for two different phases of machine learning modeling: training and testing. Ideally, no information from the test data should be used to scale the training data or should be used to direct the training process of a machine learning model. Hence, we first split the data and then apply the scaling.</p>
# <p>Also, features like <code>DriversLicense</code> and <code>ZipCode</code> are not as important as the other features in the dataset for predicting credit card approvals. We should drop them to design our machine learning model with the best set of features. In Data Science literature, this is often referred to as <em>feature selection</em>. </p>

# In[7]:


# Import train_test_split
from sklearn.model_selection import train_test_split

# Drop the features 11 and 13 and convert the DataFrame to a NumPy array
cc_apps = cc_apps.drop(columns=[11, 13], axis=1)
cc_apps = cc_apps.values

# Segregate features and labels into separate variables
X,y = cc_apps[:,0:12] , cc_apps[:,13]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                y,
                                test_size=0.33,
                                random_state=42)


# In[196]:


get_ipython().run_cell_magic('nose', '', '\ndef test_columns_dropped_correctly():\n    assert cc_apps.shape == (690,14), \\\n        "The shape of the DataFrame isn\'t correct. Did you drop the two columns?"\n\ndef test_data_split_correctly():\n    X_train_correct, X_test_correct, y_train_correct, y_test_correct = train_test_split(X, y, \\\n                                                                                   test_size=0.33, random_state=42)\n    assert X_train_correct.all() == X_train.all() and X_test_correct.all() == X_test.all() and \\\n            y_train_correct.all() == y_train.all() and y_test_correct.all() == y_test.all(), \\\n                "It doesn\'t appear that the data splitting was done correctly."')


# ## 8. Preprocessing the data (part ii)
# <p>The data is now split into two separate sets - train and test sets respectively. We are only left with one final preprocessing step of scaling before we can fit a machine learning model to the data. </p>
# <p>Now, let's try to understand what these scaled values mean in the real world. Let's use <code>CreditScore</code> as an example. The credit score of a person is their creditworthiness based on their credit history. The higher this number, the more financially trustworthy a person is considered to be. So, a <code>CreditScore</code> of 1 is the highest since we're rescaling all the values to the range of 0-1.</p>

# In[8]:


# Import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# Instantiate MinMaxScaler and use it to rescale X_train and X_test
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.fit_transform(X_test)


# In[198]:


get_ipython().run_cell_magic('nose', '', '\ndef test_features_range_set_correctly():\n    min_value_in_rescaledX_train = np.amin(rescaledX_train)\n    max_value_in_rescaledX_train = np.amax(rescaledX_train)\n    min_value_in_rescaledX_test = np.amin(rescaledX_test)\n    max_value_in_rescaledX_test = np.amax(rescaledX_test)\n    assert min_value_in_rescaledX_train == 0.0 and max_value_in_rescaledX_train == 1.0 and \\\n        min_value_in_rescaledX_test == 0.0 and max_value_in_rescaledX_test == 1.0, \\\n        "It doesn\'t appear that the value range was scaled to a minimum of 0 and a maximum of 1."')


# ## 9. Fitting a logistic regression model to the train set
# <p>Essentially, predicting if a credit card application will be approved or not is a <a href="https://en.wikipedia.org/wiki/Statistical_classification">classification</a> task. <a href="http://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.names">According to UCI</a>, our dataset contains more instances that correspond to "Denied" status than instances corresponding to "Approved" status. Specifically, out of 690 instances, there are 383 (55.5%) applications that got denied and 307 (44.5%) applications that got approved. </p>
# <p>This gives us a benchmark. A good machine learning model should be able to accurately predict the status of the applications with respect to these statistics.</p>
# <p>Which model should we pick? A question to ask is: <em>are the features that affect the credit card approval decision process correlated with each other?</em> Although we can measure correlation, that is outside the scope of this notebook, so we'll rely on our intuition that they indeed are correlated for now. Because of this correlation, we'll take advantage of the fact that generalized linear models perform well in these cases. Let's start our machine learning modeling with a Logistic Regression model (a generalized linear model).</p>

# In[9]:


# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

# Instantiate a LogisticRegression classifier with default parameter values
logreg = LogisticRegression()

# Fit logreg to the train set
logreg.fit(rescaledX_train, y_train)


# In[200]:


get_ipython().run_cell_magic('nose', '', '\ndef test_logreg_defined():\n    assert "logreg" in globals(),\\\n        "Did you instantiate LogisticRegression in the logreg variable?"\n\ndef test_logreg_defined_correctly():\n    logreg_correct = LogisticRegression()\n    assert str(logreg_correct) == str(logreg), \\\n        "The logreg variable should be defined with LogisticRegression() only."')


# ## 10. Making predictions and evaluating performance
# <p>But how well does our model perform? </p>
# <p>We will now evaluate our model on the test set with respect to <a href="https://developers.google.com/machine-learning/crash-course/classification/accuracy">classification accuracy</a>. But we will also take a look the model's <a href="http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/">confusion matrix</a>. In the case of predicting credit card applications, it is equally important to see if our machine learning model is able to predict the approval status of the applications as denied that originally got denied. If our model is not performing well in this aspect, then it might end up approving the application that should have been approved. The confusion matrix helps us to view our model's performance from these aspects.  </p>

# In[10]:


# Import confusion_matrix
from sklearn.metrics import confusion_matrix

# Use logreg to predict instances from the test set and store it
y_pred = logreg.predict(rescaledX_test)

# Get the accuracy score of logreg model and print it
print("Accuracy of logistic regression classifier: ", logreg.score(rescaledX_test, y_test).round(4))

# Print the confusion matrix of the logreg model
confusion_matrix(y_test, y_pred)


# In[202]:


get_ipython().run_cell_magic('nose', '', '\ndef test_ypred_defined():\n    assert "y_pred" in globals(),\\\n        "The variable y_pred should be defined."\n\ndef test_ypred_defined_correctly():\n    correct_y_pred = logreg.predict(rescaledX_test)\n    assert str(correct_y_pred) == str(y_pred),\\\n        "The y_pred variable should contain the predictions as made by LogisticRegression on rescaledX_test."')


# ## 11. Grid searching and making the model perform better
# <p>Our model was pretty good! It was able to yield an accuracy score of almost 84%.</p>
# <p>For the confusion matrix, the first element of the of the first row of the confusion matrix denotes the true negatives meaning the number of negative instances (denied applications) predicted by the model correctly. And the last element of the second row of the confusion matrix denotes the true positives meaning the number of positive instances (approved applications) predicted by the model correctly.</p>
# <p>Let's see if we can do better. We can perform a <a href="https://machinelearningmastery.com/how-to-tune-algorithm-parameters-with-scikit-learn/">grid search</a> of the model parameters to improve the model's ability to predict credit card approvals.</p>
# <p><a href="http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html">scikit-learn's implementation of logistic regression</a> consists of different hyperparameters but we will grid search over the following two:</p>
# <ul>
# <li>tol</li>
# <li>max_iter</li>
# </ul>

# In[11]:


# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Define the grid of values for tol and max_iter
tol = [0.01, 0.001 ,0.0001]
max_iter = [100, 150, 200]

# Create a dictionary where tol and max_iter are keys and the lists of their values are corresponding values
param_grid = dict(tol=tol, max_iter=max_iter)


# In[204]:


get_ipython().run_cell_magic('nose', '', '\ndef test_tol_defined():\n    assert "tol" in globals(),\\\n        "The variable tol should be defined."\n\ndef test_max_iter_defined():\n    assert "max_iter" in globals(),\\\n        "The variable max_iter should be defined."\n    \ndef test_tol_defined_correctly():\n    correct_tol = [0.01, 0.001 ,0.0001]\n    assert correct_tol == tol, \\\n        "It looks like the tol variable is not defined with the list of correct values."\n    \ndef test_max_iter_defined_correctly():\n    correct_max_iter = [100, 150, 200]\n    assert correct_max_iter == max_iter, \\\n        "It looks like the max_iter variable is not defined with a list of correct values."    \n  \ndef test_param_grid_defined():\n    assert "param_grid" in globals(),\\\n        "The variable param_grid should be defined."\n\ndef test_param_grid_defined_correctly():\n    correct_param_grid = dict(tol=tol, max_iter=max_iter)\n    assert str(correct_param_grid) == str(param_grid),\\\n        "It looks like the param_grid variable is not defined properly."')


# ## 12. Finding the best performing model
# <p>We have defined the grid of hyperparameter values and converted them into a single dictionary format which <code>GridSearchCV()</code> expects as one of its parameters. Now, we will begin the grid search to see which values perform best.</p>
# <p>We will instantiate <code>GridSearchCV()</code> with our earlier <code>logreg</code> model with all the data we have. Instead of passing train and test sets separately, we will supply <code>X</code> (scaled version) and <code>y</code>. We will also instruct <code>GridSearchCV()</code> to perform a <a href="https://www.dataschool.io/machine-learning-with-scikit-learn/">cross-validation</a> of five folds.</p>
# <p>We'll end the notebook by storing the best-achieved score and the respective best parameters.</p>
# <p>While building this credit card predictor, we tackled some of the most widely-known preprocessing steps such as <strong>scaling</strong>, <strong>label encoding</strong>, and <strong>missing value imputation</strong>. We finished with some <strong>machine learning</strong> to predict if a person's application for a credit card would get approved or not given some information about that person.</p>

# In[12]:


# Instantiate GridSearchCV with the required parameters
grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)

# Use scaler to rescale X and assign it to rescaledX
rescaledX = scaler.fit_transform(X)

# Fit data to grid_model
grid_model_result = grid_model.fit(rescaledX, y)

# Summarize results
best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
print("Best: %f using %s" % (best_score, best_params))


# In[206]:


get_ipython().run_cell_magic('nose', '', '\ncorrect_grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)\ncorrect_grid_model_result = correct_grid_model.fit(rescaledX, y)\n\ndef test_grid_model_defined():\n    assert "grid_model" in globals(),\\\n        "The variable grid_model should be defined."\n\ndef test_grid_model_defined_correctly():\n    #correct_grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)\n    assert str(correct_grid_model) == str(grid_model),\\\n        "It doesn\'t appear that grid_model was defined correctly."\n\ndef test_features_range_set_correctly():\n    min_value_in_rescaledX = np.amin(rescaledX)\n    max_value_in_rescaledX = np.amax(rescaledX)\n    assert min_value_in_rescaledX == 0.0 and max_value_in_rescaledX == 1.0, \\\n        "It doesn\'t appear that the X was scaled to a minimum of 0 and a maximum of 1."    \n    \ndef test_grid_model_results_defined():\n    assert "grid_model_result" in globals(),\\\n        "The variable grid_model_result should be defined."\n    \ndef test_grid_model_result_defined_correctly():\n#     correct_grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)\n#     correct_grid_model_result = correct_grid_model.fit(rescaledX, y)\n    assert str(correct_grid_model_result) == str(grid_model_result), \\\n        "It doesn\'t appear that grid_model_result was defined correctly."\n    \ndef test_best_score_defined_correctly():\n#     correct_grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)\n#     correct_grid_model_result = correct_grid_model.fit(rescaledX, y)\n    correct_best_score = correct_grid_model_result.best_score_\n    assert correct_best_score == best_score,\\\n        "It looks like the variable best_score is not defined correctly."\n    \ndef test_best_params_defined_correctly():\n#     correct_grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)\n#     correct_grid_model_result = correct_grid_model.fit(rescaledX, y)\n    correct_best_params = correct_grid_model_result.best_params_\n    assert correct_best_params == best_params,\\\n        "It looks like the variable best_params is not defined correctly."')

