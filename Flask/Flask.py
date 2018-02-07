from flask import Flask,render_template,request
app = Flask(__name__)
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import send_from_directory
from os.path import join, dirname, realpath
import os


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
            print("dsadsadsadsadsa")
            return redirect(request.url)
        file = request.files['dataset']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            # flash('No selected file')
            print("No file")
            return redirect(request.url)
        if file and allowed_file(file.filename) :
            print("succsess ty kokot")
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # return redirect(url_for('uploaded_file',
                      #              filename=filename))

            return render_template('upload.html')



if __name__ == '__main__':
    app.run(debug = True)
