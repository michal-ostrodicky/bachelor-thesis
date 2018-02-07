import numpy as np
import pandas as pd
import pywt
import xlrd
import pyflux as pf
from arch import arch_model
import seaborn as sns
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime


# PREPARING DATA
data = xlrd.open_workbook("elspot-prices_2017_hourly_sek.xls")

data_xls = pd.read_excel('elspot-prices_2017_hourly_sek.xls', 'elspot-prices_2017_hourly_sek', index_col=None)
data_xls.to_csv('prices.csv', encoding='utf-8')


data_csv = pd.read_csv("prices.csv")

# Pridanie hodiny k datumu
dates = data_csv[data_csv.columns[0:2]]
dates.columns = ['Day', 'Hour']
dates['Hour'] = dates['Hour'].map(lambda x: str(x)[:2])

df = pd.DataFrame(dates)
df['Period'] = df.Day.astype(str).str.cat(df.Hour.astype(str), sep=' ')
df['Period'] = pd.to_datetime(df["Period"])

data_csv['Hours'] = df['Period']
data_csv = data_csv.drop(data_csv.columns[[0]], axis=1)


# VYBER STLPCA, pre ktory chceme robit predikciu
bergen_data = data_csv[['Hours','Bergen']]


# Plotts
# bergen_data.index = data_csv['Hours'].values
# plt.figure(figsize=(15,5))
# plt.plot(bergen_data.index,bergen_data['Bergen'])
# plt.ylabel('Price')
# plt.xlabel('Date')
# plt.title('Price of electricity');
# plt.show()


print(bergen_data['Bergen'])


X = bergen_data['Bergen'].values
size = int(len(X) * 0.66)

print('Dlzka dat je ', len(X))
print('Size je ', size)

train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
history = np.array(history)
print(history)


for t in range(len(test)):
    model = pf.ARIMA(data=history, ar=3, ma=1, integ=0, family=pf.Normal())
    x = model.fit("MLE")
    # print(x.summary())
    radcA = model.predict(h=1, intervals=False)
    #print(radcA)
    yhat = radcA['Series'].values
    predictions.append(yhat)
    obs = test[t]
    history = np.append(history,obs)
    #print('predicted=%f, expected=%f' % (yhat, obs))


error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)




