#!/usr/bin/env python
# coding: utf-8

# ### Time Series Analysis and Forecasting of superstore sales (posted in https://community.tableau.com/docs/DOC-1236).

# In[16]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import itertools
import warnings
warnings.filterwarnings("ignore")
import statsmodels.api as sm


# In[2]:


matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'


# ### Import and Visualize dataset

# In[17]:


df = pd.read_excel("Superstore.xls")


# In[18]:


df.columns


# In[19]:


# df.dtypes
df.isnull().sum()


# In[9]:


print(df)


# In[ ]:


#### We are forecasting sales of furniture first, so lets create a dataset of furniture. To create the furniture dataset, 
#### we need to drop 'Office Supplies' and 'Technology' from Category Column.


# In[20]:


df = df[df.Category != 'Office Supplies']
df = df[df.Category != 'Technology']


# In[21]:


# First order the dataset and then group based on the 'Order Date'
df = df.sort_values('Order Date')
df = df.groupby('Order Date').sum()


# In[22]:


## our dataset contains daily values, lets try to use average monthly values based on the daily values.
## change of daily values to the monthly values is based on the resample function. 
## The 'MS' string groups the data in buckets by start of the month
y = df['Sales'].resample('MS').mean()


# In[23]:


# Checking dataset
y.head(5)


# In[16]:


# Visualizing the time series of the dataset
y.plot(figsize=(15,6))
plt.show()


# In[17]:


## The time series plot shows variation with time showing clear indication of the patterns.
## sales are lower at the beginning of each year and higher close to the end of the year.
## which is an indication of seasonality.Trend is not clear in this dataset.

## In time-series analysis, three components (i.e., trend, seasonality, and residuals)
## are analyzed. the three components can be plotted/visualized through 'time-series decomposition'.
## The decomposition can be achieved by seasonal_decompose function of the statsmodels.

from pylab import rcParams
rcParams['figure.figsize'] = 18,8

decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()


# In[ ]:


# Above plot shows clear seasonality pattern. Trend is not clear. plot shows large residuals.


# ### Forecasting with ARIMA

# ##### Parameter Selection for ARIMA

# In[18]:


## Use grid searach to iteratively explore different combinations of parameters. 
## For each combination of parameters, fit a new seasonal ARIMA model with the SARIMAX()
## function from the 'statsmodels' module and assess its overall quality. 
## The optimal set of parameters yields the best performance for our criteria of interest.

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# In[19]:


## use the triplets of parameters defined above to automate the process of training and
## evaluating ARIMA models on different combinations, the process is called grid search
## (or hyperparameter optimization) for model selection.
## Use AIC to measure how well a model fits data with consideration of the overall complexity of the model.


warnings.filterwarnings("ignore") # specify to ignore warning messages

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


# In[ ]:


## ARIMA (1,1,1)x(1,1,0,12) yields the lowest AIC value of 297.78. Therefore, we should 
## consider this to be optimal option.


# In[20]:


### Fitting the ARIMA model
## Lets put the optimal parameter values, found from gird search, into a new SARIMAX model:

mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])


# In[ ]:


## Above summary attribute tables shows ar.L1 and ar.S.L12 are statistically insiginificant.
## the coef column shows the weight (i.e., importance) of each feature and how each one
## impacts the time series. 


# In[21]:


## Lets run model diagnostics to investigate any unusual behavior than suggested 
## by the calculated ARIMA model. We can use 'plot_diagnostics' object to quickly generate
## model diagnostics and investigate for any unusual behavior.

results.plot_diagnostics(figsize=(16, 8))
plt.show()


# In[ ]:


## most important point from the diagram is to check if the residuals of the model 
## are uncorrelated and normally distributed with zero-mean. 

## the histogram is not perfect, however, is close to the normal distribution.
## the Q-Q plot shows the ordered distribution of residuals (blue dots) follows the
## linear trend of the samples taken from a standard normal distribution;
## which indicates residuals are normally distributed.

## The residuals over time (top left plot) shows the seasonality is week or absent, 
## which is confirmed by the autocorrelation (i.e. correlogram) plot on the bottom right,
## which shows that the time series residuals have low correlation with lagged versions of itself.

## With above results, we can say our model is white noise.


# ### Validating Forecasts

# In[22]:


## Now, we have a model which we can use to forecasts. To understand the accuracy of 
## the forecasts, lets compare the predicted values to real values of the time series.
## The get_prediction() and conf_int() attributes gives the values and associated 
## confidence intervals for the forecast.

pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
pred_ci = pred.conf_int()
## Above code gives forecasts starting at January 1998.
## The 'dynamic=False' argument is for one-step ahead forecasts, i.e., forecasts at
## each point are generated using the full history up to that point.

ax = y['2014':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
plt.legend()
plt.show()


# In[ ]:


## our forecasts show, in overall, close pattern which was present before. it has
## captured the seasonality from the beginning to the end of the year.


# In[23]:


## Lets quantify the accuracy of our forecasts. lets use MSE (Mean Squared Error and 
## Root Mean Squared Error (RMSE) to find the error of the forecasts)

y_forecasted = pred.predicted_mean
y_truth = y['2017-01-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))


# In[ ]:


## MSE is 22993.58 and RMSE is 151.64. MSE is very high.
## the furniture prices range from 400 to 1200, and the 151.64 RMSE may not be bad.
## So, our model can be considered as representative forecasting model for our data.


# ### Producing and Visualizing forecasts

# In[99]:


# get_forecasts (steps=100) computing forecasted values for 100 steps ahead in future
pred_uc = results.get_forecast(steps=100)

# getting confidence intervals of forecasts
pred_ci = pred_uc.conf_int()

## use the confidence intervals output from above code to plot the time series
## and forecasts of its future values.

ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
plt.legend()
plt.show()


# In[ ]:


## all of the forecasts are with similar trend as before, but the confidence intervals
## grow larger with further in steps or in time.

## Lets compare this result with other categories and see their relation with eachother
## over time. Lets compare time series of furniture and office supplier.


# ### Time Series comparison furniture sales and Office Supplies

# In[14]:


#### the number of sales from Office Supplies is higher than from Furniture 

df = pd.read_excel("Superstore.xls")

furniture = df.loc[df['Category'] == 'Furniture']
office = df.loc[df['Category'] == 'Office Supplies']
furniture.shape, office.shape


# In[ ]:


### Data Exploration
## We are going to compare two categories’ sales in the same time period. 
## This means combine two data frames into one and plot these two categories’ 
## time series into one plot.


# In[15]:


cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit']
furniture.drop(cols, axis=1, inplace=True)
office.drop(cols, axis=1, inplace=True)
furniture = furniture.sort_values('Order Date')
office = office.sort_values('Order Date')
furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()
office = office.groupby('Order Date')['Sales'].sum().reset_index()
furniture = furniture.set_index('Order Date')
office = office.set_index('Order Date')
y_furniture = furniture['Sales'].resample('MS').mean()
y_office = office['Sales'].resample('MS').mean()
furniture = pd.DataFrame({'Order Date':y_furniture.index, 'Sales':y_furniture.values})
office = pd.DataFrame({'Order Date': y_office.index, 'Sales': y_office.values})
store = furniture.merge(office, how='inner', on='Order Date')
store.rename(columns={'Sales_x': 'furniture_sales', 'Sales_y': 'office_sales'}, inplace=True)
store.head()


# In[104]:


plt.figure(figsize=(20, 8))
plt.plot(store['Order Date'], store['furniture_sales'], 'b-', label = 'furniture')
plt.plot(store['Order Date'], store['office_sales'], 'r-', label = 'office supplies')
plt.xlabel('Date'); plt.ylabel('Sales'); plt.title('Sales of Furniture and Office Supplies')
plt.legend();


# In[ ]:


### the two plots show similar pattern, lower in the beginning of the year and higher 
## towards the end of the year. By, the end of the year, sales of both category falls.
## The sales of the furniture is higher than that of office supplies. Occasionally, 
## there are some times, when office supplies is higher than that of furniture, 
## Which is checked in below codes.


# In[105]:


first_date = store.ix[np.min(list(np.where(store['office_sales'] > store['furniture_sales'])[0])), 'Order Date']

print("Office supplies first time produced higher sales than furniture is {}.".format(first_date.date()))


# In[ ]:


## References:
# https://www.digitalocean.com/community/tutorials/a-guide-to-time-series-forecasting-with-arima-in-python-3

