from flask import Flask, request, render_template
from predict import predict

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Buraya yüklenen dosya ile predict() fonksiyonunu entegre edebilirsin
        return "Tahmin tamamlandı"
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
