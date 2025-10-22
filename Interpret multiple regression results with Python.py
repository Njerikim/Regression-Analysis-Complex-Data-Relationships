#!/usr/bin/env python
# coding: utf-8

# # Multiple linear regression 

# In[2]:


# Importing packages
import pandas as pd
import seaborn as sns


# In[3]:


# Loading dataset
penguins = sns.load_dataset("penguins", cache=False)

# Examining first 5 rows of dataset
penguins.head()


# ## Data cleaning

# In[4]:


# Subset data
penguins = penguins[["body_mass_g", "bill_length_mm", "sex", "species"]]

# Renaming columns
penguins.columns = ["body_mass_g", "bill_length_mm", "gender", "species"]

# Droping rows with missing values
penguins.dropna(inplace=True)

# Reseting index
penguins.reset_index(inplace=True, drop=True)


# In[5]:


# Examining first 5 rows of data
penguins.head()


# ## Creating holdout sample

# In[10]:


# Subset X and y variables
penguins_X = penguins[["bill_length_mm", "gender", "species"]]
penguins_y = penguins[["body_mass_g"]]


# In[11]:


# Importing train-test-split function from sci-kit learn
from sklearn.model_selection import train_test_split


# In[12]:


# Creating training data sets and holdout (testing) data sets
X_train, X_test, y_train, y_test = train_test_split(penguins_X, penguins_y, 
                                                    test_size = 0.3, random_state = 42)


# ## Model construction

# In[16]:


# Write out OLS formula as a string
ols_formula = "body_mass_g ~ bill_length_mm + C(gender) + C(species)"


# In[15]:


# Importing ols() function from statsmodels package
from statsmodels.formula.api import ols


# In[17]:


# Creating OLS dataframe
ols_data = pd.concat([X_train, y_train], axis = 1)

# Creating OLS object and fit the model
OLS = ols(formula = ols_formula, data = ols_data)
model = OLS.fit()


# ## Model evaluation and interpretation

# In[18]:


# Getting model results
model.summary()


# We can then interpret each of the beta coefficients for each X variable.
# 
# ### C(gender) - Male
# Given the name of the variable, we know that the variable was encoded as `Male = 1`, `Female = 0`. This means that female penguins are the reference point. If all other variables are constant, then we would expect a male penguin's body mass to be about 528.95 grams more than a female penguin's body mass.
# 
# ### C(species) - Chinstrap and Gentoo
# Given the names of these two variables, we know that Adelie penguins are the reference point. So, if we compare an Adelie penguin and a Chinstrap penguin, who have the same characteristics except their species, we would expect the Chinstrap penguin to have a body mass of about 285.39 grams less than the Adelie penguin. If we compare an Adelie penguin and a Gentoo penguin, who have the same characteristics except their species, we would expect the Gentoo penguin to have a body mass of about 1,081.62 grams more than the Adelie penguin.
# 
# ### Bill length (mm)
# Lastly, bill length (mm) is a continuous variable, so if we compare two penguins who have the same characteristics, except one penguin's bill is 1 millimeter longer, we would expect the penguin with the longer bill to have 35.55 grams more body mass than the penguin with the shorter bill.
