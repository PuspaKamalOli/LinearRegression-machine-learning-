import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import math

df = pd.read_csv('USA_Housing.csv')
df.head()

# visualizing data of Housing.csv
sns.pairplot(df)

x = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
        'Avg. Area Number of Bedrooms', 'Area Population']]
y = df['Price']

# splitting data into test and train set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=101)
lm = LinearRegression()

# training data
lm.fit(x_train, y_train)

# making a dataframe to know check coeff required for data analysis
cdf = pd.DataFrame(lm.coef_, x.columns, columns=['coeff'])
print(cdf)

# predicting price of house
predictions = lm.predict(x_test)

# checking accuracy of model from variance score ,value near 1 is best
metrics.explained_variance_score(y_test, predictions)

# checking through scatter plot if straight line appears means model is very good
plt.scatter(y_test, predictions, c='blue', alpha=0.6)
