"""Emotion detection using the IBM Watson NLP emotion service."""

import requests


EMOTION_URL = (
    "https://sn-watson-emotion.labs.skills.network/"
    "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
MODEL_ID = "emotion_aggregated-workflow_lang_en_stock"


def emotion_detector(text_to_analyze):
    """Return emotion scores and the dominant emotion for input text.

    If the service rejects the input with HTTP 400, return None values.
    """
    if not isinstance(text_to_analyze, str) or not text_to_analyze.strip():
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }

    headers = {
        "grpc-metadata-mm-model-id": MODEL_ID,
        "Content-Type": "application/json",
    }
    payload = {"raw_document": {"text": text_to_analyze}}

    try:
        response = requests.post(
            EMOTION_URL, headers=headers, json=payload, timeout=30
        )
        if response.status_code == 400:
            return {
                "anger": None,
                "disgust": None,
                "fear": None,
                "joy": None,
                "sadness": None,
                "dominant_emotion": None,
            }
        response.raise_for_status()
        emotion_scores = response.json()["emotionPredictions"][0]["emotion"]
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        return {
            "anger": emotion_scores.get("anger"),
            "disgust": emotion_scores.get("disgust"),
            "fear": emotion_scores.get("fear"),
            "joy": emotion_scores.get("joy"),
            "sadness": emotion_scores.get("sadness"),
            "dominant_emotion": dominant_emotion,
        }
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError):
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }
