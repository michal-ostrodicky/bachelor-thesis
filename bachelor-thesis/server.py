from flask import Flask,render_template, request
from werkzeug.utils import secure_filename
from os.path import join, dirname, abspath
import os
import math
import numpy as np
import pandas as pd
from ARIMA import wavelet_smooth,prediction_arima_flask
from neural_network import prediction_neural_network_flask
import matplotlib as plt
UPLOAD_FOLDER = join(dirname(abspath(__file__)), 'static/uploads')

ALLOWED_EXTENSIONS = set(['csv'])
data = None
market = None
g_filename = None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)
@app.route("/")
def hello():
    return render_template('index.html')

def upload_file():
    if request.method == 'POST':
        if 'dataset' not in request.files:
            print("Something went wrong")
        file = request.files['dataset']

        if file.filename == '':
            return render_template('index.html' ,empty_upload=True)

        if file and allowed_file(file.filename) :
            global g_filename
            g_filename = secure_filename(file.filename)
            filename = secure_filename(file.filename)
            file.save('/var/www/ostrodickyApp/ostrodickyApp/static/uploads/'+g_filename)
            data_csv = pd.read_csv('/var/www/ostrodickyApp/ostrodickyApp/static/uploads/'+g_filename)

            column_names = list(data_csv.columns.values)
            global data,info_data,end_date

            data = data_csv
            info_data = [0, 0, '', '']
            info_data[0] = data.shape[0]
            info_data[1] = len(column_names[2:])
            start_date = data.iat[0, 0]
            end_date = data.iat[data.shape[0] - 1, 0]
            info_data[2] = data.iat[0, 0]  # start date
            info_data[3] = data.iat[data.shape[0] - 1, 0]  # end date
            return render_template('upload.html', info =info_data, labels = column_names[2:])
        else:
            return render_template('index.html', invalid_extension=True)

@app.route('/choose', methods=['POST'])
def choose_label():
    if request.method == 'POST':
        global market
        market = request.form["label_to_predict"]
        fancy_data = data[market].describe()
        column_names = list(data.columns.values)
        fancy_data = np.round(fancy_data, 2)
        return render_template('upload.html', info=info_data, statistics = fancy_data ,to_predict= market, labels = column_names[2:], uploded_file = True)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
            something = request.values
            data[market].fillna((data[market].mean()), inplace=True)
            size = int(len(data[market].values) * 0.66)
            data[market] = wavelet_smooth(data[market].values)
            model_fit,mape_statistical,result_arima = prediction_arima_flask(data[market].values,market,size)


            datumy = [None] * 25
            td = np.timedelta64(1, 'h')
            last_date = pd.to_datetime(end_date)
            datumy[0] = last_date + 24*td

            for i in range(24):
                datumy[i + 1] = datumy[i] + td

            mape_statistical = np.round(mape_statistical, 2)
            output =np.round(result_arima, 2)
            result_arima = [datumy, output]
            model_neural,mape_network,result_network = prediction_neural_network_flask(data,market)
            mape_network = np.round(mape_network, 2)
            output = np.round(result_network, 2)
            result_network = [datumy, output]

            return render_template('prediction.html', mape_statistical=mape_statistical,
                                   mape_network=mape_network, result_network=result_network,
                                   result_arima=result_arima, market=market)


if __name__ == "__main__":
    app.run()





					

