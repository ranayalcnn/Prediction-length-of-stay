from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import webbrowser  # ✅ Otomatik açmak için eklendi
from threading import Timer  # ✅ Web'i gecikmeyle açmak için

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'hea', 'dat'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        files = request.files.getlist("files")
        filenames = []

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                filenames.append(filename)

        if len(filenames) >= 2:
            prediction = "Tahmini ICU süresi: 3.8 gün"  # Dummy
        else:
            prediction = "Lütfen .hea ve .dat dosyalarını yükleyin."

    return render_template("index.html", prediction=prediction)

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == "__main__":
    Timer(1, open_browser).start()  # ✅ Uygulama başlamadan 1 saniye sonra tarayıcı aç
    app.run(debug=True)
