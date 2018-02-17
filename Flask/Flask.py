from flask import Flask,render_template,request
app = Flask(__name__)
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import send_from_directory
from os.path import join, dirname, realpath
import os
import xlrd
import numpy as np
import pandas as pd



UPLOAD_FOLDER = join(dirname(realpath(__file__)), 'static/uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


ALLOWED_EXTENSIONS = set(['xls', 'csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')



def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


#
# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'],
#                                filename)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'dataset' not in request.files:
            # flash('No file part')
            print("Something went wrong")
            # return redirect(request.url)
        file = request.files['dataset']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            # flash('No selected file')
            empty_upload = True
            print("No file")
            return render_template('index.html' ,empty_upload=True)

        if file and allowed_file(file.filename) :
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # return redirect(url_for('uploaded_file',
                      #              filename=filename))

            filename_without_extension, file_extension = os.path.splitext(filename)
            path_saved_file = os.path.join(app.config['UPLOAD_FOLDER']) + "/" + filename

            ## otvorenie file podla toho, aky extension ma file

            if(file_extension == '.xls' ):
                data_xls = pd.read_excel(path_saved_file, filename_without_extension, index_col=None)
                data_xls.to_csv('prices.csv', encoding='utf-8')
                data_csv = pd.read_csv("prices.csv")

            elif(file_extension == '.csv'):
                data_csv = pd.read_csv(path_saved_file)

            column_names = list(data_csv.columns.values)

            #print(column_names)
            prediction_arima(data_csv)

            return render_template('upload.html', labels = column_names)
        else:
            invalid_extension = True
            return render_template('index.html', invalid_extension = True)

@app.route('/choose', methods=['POST'])
def predict_label():
    if request.method == 'POST':
        print("Vybral som label + ", request.values)
        return render_template('blank.html')



def prediction_arima(data_csv):
    pass


if __name__ == '__main__':
    app.run(debug = True)
