"""Flask web application for emotion detection."""

from flask import Flask, render_template, request

from EmotionDetection import emotion_detector

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    """Render the application home page."""
    return render_template("index.html")


@app.route("/emotionDetector", methods=["GET"])
def detect_emotion():
    """Analyze text supplied through the textToAnalyze query parameter."""
    text_to_analyze = request.args.get("textToAnalyze", "")
    if not text_to_analyze.strip():
        return "Please enter some text to analyze.", 400

    result = emotion_detector(text_to_analyze)
    if result["dominant_emotion"] is None:
        return "Unable to analyze the supplied text. Please try again.", 400

    return (
        "For the given statement, the following emotions were detected: "
        f"anger: {result['anger']}, disgust: {result['disgust']}, "
        f"fear: {result['fear']}, joy: {result['joy']} and "
        f"sadness: {result['sadness']}. "
        f"The dominant emotion is {result['dominant_emotion']}."
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
