import re
from os import path

import nltk
from optimum.onnxruntime import ORTModelForSequenceClassification
from scipy.special import softmax
from textblob import TextBlob
from transformers import AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from model_download import download

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt_tab")
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

try:
    download()
except Exception as exc:
    print("Something went wrong while downloading the model. Error:", exc)
    raise


def standard_score_to_sentiment(score):
    """Sentiment score to sentiment text"""
    if score >= 0.5:
        sentiment = "positive"
    elif score >= -0.5:
        sentiment = "neutral"
    else:
        sentiment = "negative"
    return sentiment


def rating_to_sentiment(rating):
    """Rating (1 to 5) to sentiment text conversion"""
    rating = int(rating)
    if rating > 3:
        return "positive"
    elif rating > 2:
        return "neutral"
    else:
        return "negative"


class VaderTextBlob:
    """Class for predicting sentiment using vader and textblob"""

    def __init__(self, title, text) -> None:
        self.tb_results = self.get_textblob_sentiment(title, text)
        self.vader_results = self.get_vader_sentiment(title, text)

    def eval_and_return_result(self, method, title, text):
        """Alogrithm for calculating sentiment"""
        title_score = None
        if re.search("[a-zA-Z]{2,}", title):
            title_score, title_sentiment = method(title)
        text_score, text_sentiment = method(text.strip())
        if not title_score:
            return text_score, text_sentiment
        if title_sentiment == text_sentiment:
            return round((title_score + text_score) / 2), text_sentiment
        if title_score and title_sentiment != "neutral":
            return title_score, title_sentiment
        return round(text_score, 2), text_sentiment

    def get_vader_sentiment_score(self, text):
        sia = SentimentIntensityAnalyzer()
        if text is None:
            return 0.0
        else:
            compound = float(sia.polarity_scores(str(text))["compound"])
            if compound >= 0.05:
                score = 0.5 + ((compound - 0.05) * (0.5 / 0.95))
            elif compound <= -0.05:
                score = -0.5 + ((compound + 0.05) * (0.5 / 0.95))
            else:
                score = compound * (10)
        return round(score, 2)

    def get_vader_score_and_label(self, text):
        score = self.get_vader_sentiment_score(text)
        return score, standard_score_to_sentiment(score)

    def get_vader_sentiment(self, title, text):
        return self.eval_and_return_result(method=self.get_vader_score_and_label, title=title, text=text)

    def get_textblob_sentiment_score(self, text):
        tb = TextBlob(text)
        return self.convert_textblob_polarity(tb.sentiment.polarity)

    def get_textblob_score_and_label(self, text):
        score = self.get_textblob_sentiment_score(text)
        return score, standard_score_to_sentiment(score)

    def convert_textblob_polarity(self, original_polarity):
        """
        Converts textblob sentiment score to standardized sentiment score

        .1 to 1 -> positive range
        -.1 to .1 -> neutral range
        -.1 to -1 -> negative range
        """
        if original_polarity >= 0.1:
            return 0.5 + ((original_polarity - 0.1) * (0.5 / 0.9))
        if original_polarity <= -0.1:
            return -0.5 + ((original_polarity + 0.1) * (0.5 / 0.9))
        return 0

    def get_textblob_sentiment(self, title, text):
        return self.eval_and_return_result(method=self.get_textblob_score_and_label, title=title, text=text)


class RobustSentimentAnalysis:
    # Load model and tokenizer
    model_name = "robust-sentiment-analysis"
    model_path = path.join(path.expanduser("~"), ".models", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, file_name="model.onnx")
    model = ORTModelForSequenceClassification.from_pretrained(model_path, file_name="model.onnx")

    # Function to predict sentiment
    def predict_sentiment(self, text, show_probabilities=True):
        inputs = self.tokenizer(text.lower(), return_tensors="pt", truncation=True, padding=True, max_length=512)

        inputs["token_type_ids"] = inputs["attention_mask"]  # For fixing NoneType error in onnx, this has no impact on the results
        outputs = self.model.forward(**inputs)
        # onnx output is still a tensor, we need to convert it to numpy for finding the softmax
        probabilities = softmax(outputs.logits.numpy())
        sentiment_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}

        # compute weighted score
        weights = {"Very Negative": -1, "Negative": -0.75, "Neutral": 0, "Positive": 0.75, "Very Positive": 1}
        weighted_score = 0
        if show_probabilities:
            for sentiment, prob in zip(sentiment_map.values(), probabilities[0]):
                weighted_score += prob.item() * weights[sentiment]

        sentiment = standard_score_to_sentiment(weighted_score)

        return weighted_score, sentiment


class ReviewSentiment:
    model = RobustSentimentAnalysis()

    def get_review_sentiment(self, rating, title, text):
        """Extracts review sentiment using conditional logic combining multiple sentiment extraction methods"""
        title = title if title else ""
        text = text if text else ""

        if not title and not text:
            return 0.0

        if text and text[0] == "[":
            text = text.replace("[This review was collected as part of a promotion.]", "", 1)
        rating_sentiment = rating_to_sentiment(rating)
        vader_textblob = VaderTextBlob(title=title, text=text)
        tb_score, tb_sentiment = vader_textblob.tb_results
        va_score, va_sentiment = vader_textblob.vader_results
        if rating_sentiment == va_sentiment:
            return round(va_score, 2)
        if rating_sentiment == tb_sentiment:
            return round(tb_score, 2)
        dscore, dsentiment = self.model.predict_sentiment((title if title else "") + "\n" + text)
        return round(dscore, 2)


# if __name__ == "__main__":
#     rs = ReviewSentiment()
#     rating = 1
#     title = "LOVE"
#     text = "This stroller & car seat set is so amazing! Not big and bulky it's just the right size! It makes traveling so easy for me & my family. Also comes with a cup holder !"
#     print(rs.get_review_sentiment(rating=rating, title=title, text=text))
