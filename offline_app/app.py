from flask import Flask, request, jsonify, render_template_string
from ai import generate_text

app = Flask(__name__)

HTML = """
<!doctype html>
<title>Offline AI</title>
<h1>Offline AI Generator</h1>
<form method=post action="/generate">
  <textarea name=prompt rows=6 cols=60 placeholder="Enter prompt..."></textarea><br>
  <input type=submit value=Generate>
</form>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML)


@app.route("/generate", methods=["POST"])
def generate():
    if request.is_json:
        data = request.get_json()
        prompt = data.get("prompt", "")
    else:
        prompt = request.form.get("prompt", "")

    if not prompt:
        return jsonify({"error": "prompt required"}), 400

    try:
        out = generate_text(prompt, max_length=128)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"response": out})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
