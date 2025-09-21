import flask

app = flask.Flask(__name__)


@app.route("/")
def home():
    return flask.render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if flask.request.method == "POST":
        pass
    else:
        return "Method Not Allowed", 445
    return None


if __name__ == "__main__":
    app.run()
