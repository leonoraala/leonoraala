#LEONORA ALA
#105038131
#COMPLETED: APRIL 4,2021
#the goal of this project is to predict the temperature of this upcoming july month based off of the 5 julys prior 

import pandas as pd
from pandas import ExcelFile 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

#here we are reading the excel files that has 4 years of data 
#the dataset consists of the max temp, avg temp, and min temp of each day of the month of July per year
#where the year range is 2016-2020
#data has been taken from www.wunderground.com/history/monthly/ca/windsor/CYQG/date/2020-7 (note: the date changes for the end of the web address)
df=pd.read_excel('Desktop/weatherData/dataset.xlsx')


#target vector is what we want to predict 
target = 'tempavg'
Y = df[target]


#feature matrix are all of the columns 
X = df[["dates","tempmax", "tempavg", "tempmin"]]
#we drop the dates column since it does not have any value in predicting the temp for july
X=X.drop('dates', axis=1)
#now we are going to train test our x and y 
#I have split the datasets into 75% training and 25% testing
#the random state refers to getting a random selection of data, 
X_train, X_val, Y_train, Y_val = train_test_split(X,Y,test_size=0.25, random_state=46)
#printing just to check that it works, which it does yay!
print(X_train.shape)
print(X_val.shape)
print(Y_train.shape)
print(Y_val.shape)

#the next step is to establish a baseline mean absolute error
#if our model does not improve beyond this baseline then it is a failure, and another model must be implemented
#the baseline chosen is the average temperature for the month of july
#we must beat the avg error that is computed here 
y_prediction =[Y_train.mean()] * len(Y_train)
print('Baseline (aka the mean absolute error):', round(mean_absolute_error(Y_train, y_prediction),3))


#I decided to use Random Forest Regressor model  (RFGM)
#RFGM uses multiple decision trees as the base learning models for the ai 
#it performs row and feature sampling from the dataset, then forming sample datasets for every model 
forestModel = make_pipeline(
    SelectKBest(k='all'), #scores the features using an internal function, here we chose to score all the features 
    StandardScaler(),
    RandomForestRegressor(
        n_estimators=100, #this represents the number of trees in this random forest 
        max_depth=50,     #reps. the depth of each tree in the forest; the deeper the tree, the more splits it has, and the more info about the data it captures
        random_state=77,  
        n_jobs=-1)        #this is the number of cores the regressor will use; -1 means it will use all cores available to run the random forest regressor
    )
#this is having our training data fit into our pipelines
forestModel.fit(X_train, Y_train)
#and here we are making predictions based on the rnadom forest regressor model 
#note: MAE = mean absolute error 
#we use MAE because we want to see how far away our average prediction is from the actual value, so we take the absolute value
print("Random Forest Regressor Model Training MAE:", mean_absolute_error(Y_train, forestModel.predict(X_train)))
print("Random Forest Regressor Model Validation MAE:", mean_absolute_error(Y_val, forestModel.predict(X_val)))
print("The Median Absolute Error: %.2f degrees fahrenheit" % median_absolute_error(Y_val, forestModel.predict(X_val)))

#in conclusion, what we notice when we run the regressor is that we have an average temperature predicition esitmate of 0.09 for our 
#validation of the forest MAE, this is about a 3.29 average improvement of our baseline which we determined to be 3.397
#of course, changing the value of our random states and how much % we chose our data to be in testing and trainging, we will get different values for our baseline and MAE 

#what I have been able to do is demonstrate how to use the Random Forest Regressor Model in order to predict future mean weather temperature of the month of July 
#using this model I could predict the expected values that are based off of the inputs from the testing subset of data, and then be able to evaluate the accuracy of this prediction 
#for fun I added the median absolute error to see the error in degrees fahrenheit 