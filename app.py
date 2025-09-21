import flask
import os
from asl_util.ai import main

app = flask.Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return flask.render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in flask.request.files:
        return flask.jsonify({"error": "No file part"}), 400

    file = flask.request.files["file"]
    if file.filename == "":
        return flask.jsonify({"error": "No selected file"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    main(filepath)

    # run ai
    return flask.jsonify({"message": "File uploaded successfully", "filename": file.filename})


if __name__ == "__main__":
    app.run()
