import os
from os.path import join, dirname, realpath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from neural_network import prediction_neural_network_flask

UPLOAD_FOLDER = join(dirname(realpath(__file__)), 'static/uploads')


ALLOWED_EXTENSIONS = set(['xls', 'csv'])
global data,market
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':

        # check if the post request has the file part
        if 'dataset' not in request.files:
            print("Something went wrong")
            # return redirect(request.url)
        file = request.files['dataset']
        # if user does not select file, browser also
        # submit a empty part without filename

        if file.filename == '':
            return render_template('index.html' ,empty_upload=True)

        if file and allowed_file(file.filename) :
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            filename_without_extension, file_extension = os.path.splitext(filename)
            path_saved_file = os.path.join(app.config['UPLOAD_FOLDER']) + "/" + filename

            ## otvorenie file podla toho, aky extension ma file

            if(file_extension == '.xls' ):
                print(path_saved_file)
                data_xls = pd.read_excel(path_saved_file, filename_without_extension, index_col=None)
                data_xls.to_csv(os.path.join(app.config['UPLOAD_FOLDER']) + "/" + filename_without_extension + '.csv', encoding='utf-8')
                data_csv = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER']) + "/" + filename_without_extension + '.csv')

            elif(file_extension == '.csv'):
                data_csv = pd.read_csv(path_saved_file)

            column_names = list(data_csv.columns.values)
            global data,info_data,end_date


            data = data_csv
            print(data['Hours'])
            #print(column_names)
            info_data = [0,0,'','']
            info_data[0] = data.shape[0]
            info_data[1] = len(column_names[2:])
            start_date = data.iat[0,0]
            end_date = data.iat[data.shape[0]-1,0]
            info_data[2]= data.iat[0,0] # start date
            info_data[3] = data.iat[data.shape[0]-1,0] # end date

            return render_template('upload.html', info =info_data, labels = column_names[2:])
        else:
            return render_template('index.html', invalid_extension = True)


@app.route('/choose', methods=['POST'])
def choose_label():
    if request.method == 'POST':
        global market
        market = request.form["label_to_predict"]
        full_filename = 'chart.png'
        fancy_data = data[market].describe()
        column_names = list(data.columns.values)
        fancy_data = np.round(fancy_data, 2)
        return render_template('upload.html', info= info_data, statistics = fancy_data ,to_predict= market, labels = column_names[2:], uploded_file = True)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        bullshit = request.values
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


        return render_template('prediction.html', mape_statistical = mape_statistical,
                               mape_network = mape_network, result_network = result_network,
                               result_arima = result_arima, market = market)

# def plot_label(data_csv, market):
# # Pridanie hodiny k datumu
#     dates = data_csv[data_csv.columns[0:2]]
#     dates.columns = ['Day', 'Hour']
#     dates['Hour'] = dates['Hour'].map(lambda x: str(x)[2:])
#
#     df = pd.DataFrame(dates)
#     df['Period'] = df.Day.astype(str).str.cat(df.Hour.astype(str), sep=' ')
#     df['Period'] = pd.to_datetime(df["Period"])
#
#     data_csv['Hours'] = df['Period']
#     data_csv = data_csv.drop(data_csv.columns[[0]], axis=1)
#     data.index = data_csv['Hours'].values
#
#     sns_plot = sns.tsplot(data=data[market])
#     fig = sns_plot.gcf()
#     fig.savefig('output.png')

def plot_label(data_csv,market):
    # Pridanie hodiny k datumu
    dates = data_csv[data_csv.columns[0:2]]
    dates.columns = ['Day', 'Hour']
    dates['Hour'] = dates['Hour'].map(lambda x: str(x)[2:])

    df = pd.DataFrame(dates)
    df['Period'] = df.Day.astype(str).str.cat(df.Hour.astype(str), sep=' ')
    df['Period'] = pd.to_datetime(df["Period"])

    data_csv['Hours'] = df['Period']
    data_csv = data_csv.drop(data_csv.columns[[0]], axis=1)


    # Plots
    data.index = data_csv['Hours'].values
    plt.figure(figsize=(18, 9))
    plt.plot(data.index, data[market])
    plt.legend(loc='upper right')
    plt.ylabel('Price')
    plt.xlabel('Date')
    plt.title('Price of electricity');
    plt.legend(loc='upper right')
    plt.grid(which='major', axis='both', linestyle='--')
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER']+ "/" + 'chart.png'))
#     # path_saved_file = os.path.join(app.config['UPLOAD_FOLDER']) + "/" + filename


if __name__ == '__main__':
    app.run(debug = True)


