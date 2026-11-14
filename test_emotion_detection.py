"""Unit tests for the emotion detection application."""

import unittest
from unittest.mock import Mock, patch

from emotion_detection import emotion_detector


class TestEmotionDetector(unittest.TestCase):
    """Test response formatting and error handling."""

    @patch("emotion_detection.requests.post")
    def test_returns_formatted_emotions(self, mock_post):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "emotionPredictions": [
                {
                    "emotion": {
                        "anger": 0.01,
                        "disgust": 0.02,
                        "fear": 0.03,
                        "joy": 0.90,
                        "sadness": 0.04,
                    }
                }
            ]
        }
        mock_post.return_value = response
        result = emotion_detector("I am very happy today")
        self.assertEqual(result["dominant_emotion"], "joy")
        self.assertEqual(result["joy"], 0.90)
        self.assertEqual(set(result), {
            "anger", "disgust", "fear", "joy", "sadness", "dominant_emotion"
        })

    @patch("emotion_detection.requests.post")
    def test_bad_request_returns_none_values(self, mock_post):
        response = Mock()
        response.status_code = 400
        mock_post.return_value = response
        result = emotion_detector("invalid request")
        self.assertIsNone(result["dominant_emotion"])

    def test_blank_input_returns_none_values(self):
        result = emotion_detector("   ")
        self.assertIsNone(result["dominant_emotion"])


if __name__ == "__main__":
    unittest.main()
