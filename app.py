import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV

import joblib

ridgeCVregressor = joblib.load(open('RidgeCV_Model.pkl', 'rb'))
preprocessor = joblib.load(open('preprocessor.pkl', 'rb'))

test_input = [[0,0,0,0,0,0,0,0,0,0,0]]

print(test_input)

import streamlit as st

st.title('Used Car Price Predictor')

test_input = pd.DataFrame(test_input, columns=['Location','Fuel_Type','Transmission','Owner_Type','Brand','Year','Kilometers_Driven','Mileage','Engine','Power','Seats'])

# Fill empty or missing values with default values
# test_input.fillna(value={['Location'][0]: 'unknown', ['Fuel_Type'][0]: 'unknown', ['Transmission'][0]: 'unknown', ['Owner_Type'][0]: 'unknown', ['Brand'][0]: 'unknown', ['Year'][0]: 0, ['Kilometers_Driven'][0]: 0, ['Mileage'][0]: 0, ['Engine']:[0] 0, ['Power'][0]: 0, ['Seats'][0]: 0}, inplace=True)

test_input['Location'][0] = 'abc'
test_input['Fuel_Type'][0] = 'abc'
test_input['Transmission'][0] = 'abc'
test_input['Owner_Type'][0] = 'abc'
test_input['Brand'][0] = 'abc'
test_input['Year'][0] = 0
test_input['Kilometers_Driven'][0] = 0
test_input['Mileage'][0] = 0
test_input['Engine'][0] = 0
test_input['Power'][0] = 0
test_input['Seats'][0] = 0

with st.sidebar:
    st.header('Enter the input values: ')
    test_input['Location'][0] = st.text_input("\nEnter location: ", value='Mumbai')
    test_input['Fuel_Type'][0] = st.text_input("\nEnter fuel type: ",value='Petrol')
    test_input['Transmission'][0] = st.text_input("\nEnter transmission type: ",value='Automatic')
    test_input['Owner_Type'][0] = st.text_input("\nEnter owner type: ",value='First')
    test_input['Brand'][0] = st.text_input("\nEnter brand name: ",value='Honda')
    test_input['Year'][0] = st.number_input("\nEnter year of manufacture: ")
    test_input['Kilometers_Driven'][0] = st.number_input("\nEnter kilometers driven: ")
    test_input['Mileage'][0] = st.number_input("\nEnter mileage: ")
    test_input['Engine'][0] = st.number_input("\nEnter engine: ")
    test_input['Power'][0] = st.number_input("\nEnter power: ")
    test_input['Seats'][0] = st.number_input("\nEnter number of seats: ")

print(test_input)
test_input_transformed = preprocessor.transform(test_input)
test_pred = ridgeCVregressor.predict(test_input_transformed)

st.write("The predicted price of the used car is: ",test_pred)

