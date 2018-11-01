# TODO: Add import statements
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Assign the dataframe to this variable.
# TODO: Load the data
bmi_life_data = pd.read_csv("bmi_and_life_expectancy.csv")

# Make and fit the linear regression model
#TODO: Fit the model and Assign it to bmi_life_model
print(len(bmi_life_data['Life expectancy'].values), len(bmi_life_data['BMI'].values))
bmi_life_model = LinearRegression()
bmi_life_model.fit(bmi_life_data['BMI'].values.reshape(len(bmi_life_data['BMI'].values), 1), 
            bmi_life_data['Life expectancy'].values.reshape(len(bmi_life_data['Life expectancy'].values), 1))

# Make a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict([[21.07931]])

print("Life exp for BMI of 21.07931 : ", laos_life_exp)
