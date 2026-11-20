import string
# import nltk
# from nltk import find
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class Normalizer:
    def __init__(self):
        # TODO: This doesnt need to be downloaded on each init
        # nltk.download("wordnet")
        # nltk.download("omw-1.4")
        self.lemmatizer = WordNetLemmatizer()

    def tokenize_and_lemmatize(self, text: str) -> str:
        if not isinstance(text, str):
            return "None"
            # Process the text with Stanza
        translator = str.maketrans("", "", string.punctuation)
        tokens = word_tokenize(text)
        tokens = [token.lower().translate(translator) for token in tokens]
        lemmatized_words = [self.lemmatizer.lemmatize(token) for token in tokens]
        # Return the lemmatized text as a single string
        return " ".join(lemmatized_words)
