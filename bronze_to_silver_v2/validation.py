from logging import getLogger

logger = getLogger()


SENTIMENT_VALUES = ["negative", "positive", "neutral"]
PLUTCHIK_EMOTIONS = ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 'Disgust', 'Anger', 'Anticipation']

def is_numeric_and_in_range(value):
    try:
        num = float(value)
        return -1 <= num <= 1
    except (ValueError, TypeError):
        return False

def map_sentiment(score):
    score = float(score)
    if score >= 0.5:
        return "Positive"
    if score <= -0.5:
        return "Negative"
    return "Neutral"

def is_invalid_string(string_value):
    return string_value is None or string_value.lower() in ["null", "none"]

def fix_emotion(emotion):
    return "Sadness" if emotion.lower() == "disappointment" else emotion.title()

def is_invalid_emotion(emotion):
    return is_invalid_string(emotion) or fix_emotion(emotion) not in PLUTCHIK_EMOTIONS


def validate_or_fix_theme_response(review_text, review_title, theme_response):
    try:
        if not theme_response["themes"]:
            logger.info(f"themes is Invalid: themes: {theme_data['themes']}")
            raise ValueError("Invalid themes")

        for idx, theme_data in enumerate(theme_response["themes"]):
            if is_invalid_string(theme_data["theme_name"]):
                logger.info(f"theme_name is Invalid: theme_name: {theme_data['theme_name']}")
                raise ValueError("Invalid theme_name")

            theme_response["themes"][idx]["theme_name"] = theme_response["themes"][idx]["theme_name"].title().strip()
            if not is_numeric_and_in_range(theme_data["theme_sentiment_score"]):
                logger.info(f"theme_sentiment_score is Invalid: theme_sentiment_score: {theme_data['theme_sentiment_score']}")
                raise ValueError("Invalid theme_sentiment_score")

            theme_response["themes"][idx]["theme_sentiment_score"] = float(theme_data["theme_sentiment_score"])
            theme_response["themes"][idx]["theme_sentiment"] = map_sentiment(theme_data["theme_sentiment_score"])
            for aidx, associated_dimension in enumerate(theme_data["associated_dimensions"]):
                if is_invalid_string(associated_dimension["dimension_name"]):
                    logger.info(f"dimension_name is Invalid: dimension_name: {associated_dimension['dimension_name']}")
                    raise ValueError("Invalid dimension_name")

                theme_response["themes"][idx]["associated_dimensions"][aidx]["dimension_name"] = associated_dimension["dimension_name"].title().strip()

                if not associated_dimension["sentiment"] or associated_dimension["sentiment"].lower() not in SENTIMENT_VALUES:
                    logger.info(f"sentiment is Invalid: sentiment: {associated_dimension['sentiment']}")
                    raise ValueError("Sub theme sentiment is invalid")

                theme_response["themes"][idx]["associated_dimensions"][aidx]["sentiment"] = associated_dimension["sentiment"].title().strip()
                if associated_dimension["primary_physical_part"] is not None and associated_dimension["primary_physical_part"].lower() in ["null", "none", ""]:
                    associated_dimension["primary_physical_part"] ='General'

                theme_response["themes"][idx]["associated_dimensions"][aidx]["primary_physical_part"] = associated_dimension["primary_physical_part"].title().strip()

                associated_dimension["mention"] = associated_dimension["mention"].strip(".") if associated_dimension["mention"] else None
                if associated_dimension["mention"] and associated_dimension["mention"] in review_title:
                    logger.info(f"mention is present in Review Title: mention: {associated_dimension['mention']}")
                    continue
                if not associated_dimension["mention"] or (associated_dimension["mention"] and associated_dimension["mention"] not in review_text):
                    logger.info(f"mention is Invalid: mention: {associated_dimension['mention']}")
                    raise ValueError("Returned theme excerpt is not present in the review")

    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        print(exc)
        raise Exception("Invalid_Response")
    except Exception as exc:
        print(f"Caught {exc} while processing theme response")
        raise exc
    return theme_response


def validate_or_fix_emotion_response(review_text, emotion_response):
    if not is_numeric_and_in_range(emotion_response["sentiment"]):
        logger.info(f"sentiment is Invalid: sentiment: {emotion_response['sentiment']}")
        raise ValueError("sentiment is invalid")

    emotion_response["sentiment"] = float(emotion_response["sentiment"])
    emotion_response["sentiment_text"] = map_sentiment(emotion_response["sentiment"])
    if 'dominant_plutchik_emotions' not in emotion_response:
        emotion_response['dominant_plutchik_emotions'] = None

    if 'plutchik_emotions' not in emotion_response:
        emotion_response['plutchik_emotions'] = None

    if not emotion_response["relevant_excerpts"]:
        logger.info(f"relevant_excerpts is Invalid: relevant_excerpts: {emotion_response['relevant_excerpts']}")
        raise ValueError("relevant_excerpts is invalid")

    for emotion_group, emotion_group_list in list(emotion_response["relevant_excerpts"].items()):
        for idx, emotion_group_dict in enumerate(emotion_group_list):
            emotion_group_dict["text"] = emotion_group_dict["text"].strip(".") if emotion_group_dict["text"] else None
            if not emotion_group_dict["text"] or (emotion_group_dict["text"] and emotion_group_dict["text"] not in review_text):
                logger.info(f"text is Invalid: text: {emotion_group_dict['text']}")
                raise ValueError("Returned emotion excerpt is not present in the review")

            if is_invalid_emotion(emotion_group_dict["emotion"]) and is_invalid_emotion(emotion_group_dict["primary_emotion"]):
                logger.info(f"emotion is Invalid: emotion: {emotion_group_dict['emotion']}")
                raise ValueError("emotion or primary_emotion is invalid")

            if is_invalid_emotion(emotion_group_dict["emotion"]):
                emotion_group_dict["primary_emotion"] =  fix_emotion(emotion_group_dict["primary_emotion"])
                if emotion_group_dict["primary_emotion"] not in PLUTCHIK_EMOTIONS:
                    logger.info(f"primary_emotion is Invalid: primary_emotion: {emotion_group_dict['primary_emotion']}")
                    raise ValueError("primary_emotion is invalid")

                emotion_group_dict["emotion"] =  emotion_group_dict["primary_emotion"]
            elif is_invalid_emotion(emotion_group_dict["primary_emotion"]):
                emotion_group_dict["emotion"] = fix_emotion(emotion_group_dict["emotion"])
                if emotion_group_dict["emotion"] not in PLUTCHIK_EMOTIONS:
                    logger.info(f"emotion is Invalid: emotion: {emotion_group_dict['emotion']}")
                    raise ValueError(f"emotion {emotion_group_dict['emotion']} is invalid")

                emotion_group_dict["primary_emotion"] =  emotion_group_dict["emotion"]
            emotion_group_dict["primary_emotion"] =  fix_emotion(emotion_group_dict["primary_emotion"])
            emotion_group_dict["emotion"] =  emotion_group_dict["primary_emotion"]
            emotion_group_dict["display_name"] = emotion_group_dict["emotion"]
            if not emotion_group_dict["polarity"] or emotion_group_dict["polarity"].lower() not in SENTIMENT_VALUES:
                logger.info(f"polarity is Invalid: polarity: {emotion_group_dict['polarity']}")
                raise ValueError("Emotion sentiment is invalid")

            emotion_group_dict["polarity"] = emotion_group_dict["polarity"].strip().title()
            emotion_response["relevant_excerpts"][emotion_group][idx] = emotion_group_dict

    relevant_excerpts = emotion_response['relevant_excerpts']
    list_of_data = [item for value in relevant_excerpts.values() for item in value]
    emotion_response['relevant_excerpts'] = list_of_data

    return emotion_response
