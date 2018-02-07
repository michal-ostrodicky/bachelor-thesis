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
bergen_data = data_csv[['Hours','FI']]


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



# WAVELET TRANSFORMATION
w = pywt.Wavelet('db4')
# print("Coefs length> " ,pywt.dwt_coeff_len(X.__len__(),w,mode='symmetric'))
# print(pywt.dwt_max_level(data_len=bergen_data.__len__(), filter_len=w.dec_len))


# coefs = pywt.wavedec(X, w) multilevel decomposition
# cA6, cD1, cD2, cD3, cD4, cD5, cD6 = coefs  # potreba spravit generalne riesenie pre N cD radov

cA, cD = pywt.dwt(X,w) # single level decomposition

# print(cA.shape)
# print(type(cA))
# print(cA)

## GARCH
model = pf.GARCH(cD,p=0,q=1)
x = model.fit()
print(x.summary())
radCD1 = model.predict(h=len(X)-size+1)
#
# print(radCD1.shape)
# print(type(radCD1))
# print(radCD1)

novy_CD1 = radCD1.as_matrix()
novy_CD1 = novy_CD1[:,0]

#
# print(novy_CD1.shape)
# print(type(novy_CD1))
# print(novy_CD1)


## ARIMA

model = pf.ARIMA(data=cA, ar=1, ma=1,integ=0, family=pf.Normal())
x = model.fit("MLE")
print(x.summary())
radcA = model.predict(h = len(X)-size+1, intervals=False)

novy_cA = radcA.as_matrix()
novy_cA = novy_cA[:,0]


# print(novy_cA.shape)
# print(type(novy_cA))
# print(novy_cA)


# Wavelet reconstruction
# coefs = cA6, volaco1, volaco2, volaco3,cD4, cD5, cD6 # multidimensional approach
# print('Po ', pywt.waverec(coefs,'db4'))

prediction = pywt.idwt(novy_cA, novy_CD1, 'db4', 'smooth')
print('Po ',prediction)

for t in range(len(test)):
    yhat = prediction[t]
    obs = test[t]
    predictions.append(yhat)
    print('predicted=%f, expected=%f' % (yhat, obs))

error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

