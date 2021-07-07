#LEONORA ALA
#105038131
#Here I used a correlation matrix to show how each column is related to one another 
#this helps to visualize how our predictions will be 
#NOTE: in this visual graph we can ignore the dates column, as in our code we drop it since it has no real value 
#towards our prediction

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas import ExcelFile 

#reading the dataset
df=pd.read_excel('Desktop/weatherData/dataset.xlsx')

corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()