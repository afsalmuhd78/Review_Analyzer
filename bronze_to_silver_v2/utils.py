import os
from itertools import groupby
from logging import getLogger

import nltk
import pandas as pd
import requests
import yaml
from nltk import find
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from settings import MC_PROJECT_META_API

logger = getLogger()

from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
    BooleanType,
    ArrayType,
    MapType,
    DateType,
)

try:
    import spacy

    if spacy.util.is_package("en_core_web_sm"):
        pass
    else:
        raise ImportError
except ImportError:
    os.system(f"python -m spacy download en_core_web_sm")

nltk_resources = [
    'corpora/stopwords.zip',
    'corpora/wordnet.zip',
    'taggers/averaged_perceptron_tagger.zip',
    'corpora/omw-1.4.zip',
    'sentiment/vader_lexicon.zip'
]


def ensure_resources_downloaded(nltk_resources):
    for resource in nltk_resources:
        try:
            find(resource)
        except LookupError:
            nltk.download(resource.split('/')[-1].strip(".zip"))


def get_sentiment_score(text):
    sia = SentimentIntensityAnalyzer()
    if text is None:
        return 0.0
    else:
        compound = float(sia.polarity_scores(str(text))['compound'])
        if compound >= 0.05:
            score = 0.5 + ((compound - 0.05) * (0.5 / 0.95))
        elif compound <= -0.05:
            score = -0.5 + ((compound + 0.05) * (0.5 / 0.95))
        else:
            score = compound * (10)
    return round(score, 2)


def load_config():
    """
    :return:
    """
    try:
        spark_home_directory = os.path.expanduser("~/spark-pipelines")
        # Load YAML config file
        with open(f"{spark_home_directory}/mc-reviews/conf.yaml", 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as err:
        raise err


def send_request(url: str, request_type='api', headers=None):
    """
    Send Request To Defined URL
    """
    response = None
    logger.warning(f'Msg: Sending Request To: {url}')
    try:
        response_obj = requests.get(url=url, headers=headers)
        if response_obj.status_code == 200:
            if request_type == 'api':
                response = response_obj.json()
            else:
                response = response_obj
    except Exception as ex:
        logger.info((
            f'Msg: Exception From Url: {url} | Exception : {ex}'
            ' | Action: Return None Response Object'))
    return response


def is_advanced_project(project):
    api_response = send_request(MC_PROJECT_META_API + f"?project={project}")
    if api_response and api_response[0]["advanced"]:
        return True
    else:
        return False


def get_reviews_silver_table_schema():
    """
    :return:
    """
    videos_type = ArrayType(
        StructType(
            [
                StructField("poster_url", StringType(), True),
                StructField("video_url", StringType(), True),
            ]
        )
    )

    aspects_schema = ArrayType(
        StructType(
            [
                StructField("aspect_name", StringType(), True),
                StructField("aspect_sentiment", StringType(), True),
            ]
        )
    )

    exrepects = ArrayType(
        StructType(
            [
                StructField("theme_group", StringType(), True),
                StructField("theme", StringType(), True),
                StructField("sub_theme", StringType(), True),
                StructField("attribute", StringType(), True),
                StructField("part", StringType(), True),
                StructField("theme_sentiment", StringType(), True),
                StructField("theme_sentiment_score", DoubleType(), True),
                StructField("sub_theme_sentiment", StringType(), True),
                StructField("llm_theme", StringType(), True),
                StructField("llm_attribute", StringType(), True),
                StructField("llm_part_name", StringType(), True),
                StructField("excerpt", StringType(), True),
            ]
        )
    )

    emotion_schema = StructType([
        StructField("sentiment", DoubleType(), True),
        StructField("sentiment_text", StringType(), True),
        StructField("plutchik_emotions", ArrayType(StringType()), True),
        StructField("dominant_plutchik_emotions", ArrayType(StringType()), True),
        StructField("relevant_excerpts", ArrayType(
            StructType([
                StructField("emotion", StringType(), True),
                StructField("display_name", StringType(), True),
                StructField("text", StringType(), True),
                StructField("polarity", StringType(), True),
                StructField("primary_emotion", StringType(), True)
            ])
        ), True)
    ])

    # Define schema
    schema = StructType(
        [
            StructField("organization", IntegerType(), True),
            StructField("project", IntegerType(), True),
            StructField("author_badges", ArrayType(StringType()), True),
            StructField("author_name", StringType(), True),
            StructField("downvote_count", IntegerType(), True),
            StructField("images", ArrayType(StringType()), True),
            StructField("is_incentivized", BooleanType(), True),
            StructField("is_recommended_by_author", BooleanType(), True),
            StructField("is_verified_purchase", BooleanType(), True),
            StructField("mc_productonretaileroforg_id", IntegerType(), True),
            StructField("product_aspects", aspects_schema, True),
            StructField("review_source", StringType(), True),
            StructField("rating", IntegerType(), True),
            StructField("retailer", StringType(), True),
            StructField("retailer_product_id", StringType(), True),
            StructField("retailer_review_summary", StringType(), True),
            StructField("review_aspects", aspects_schema, True),
            StructField("review_date", DateType(), True),
            StructField("review_id", StringType(), True),
            StructField("review_text", StringType(), True),
            StructField("review_title", StringType(), True),
            StructField("scrape_date", DateType(), True),
            StructField("source", StringType(), True),
            StructField("upvote_count", IntegerType(), True),
            StructField("variant", MapType(StringType(), StringType()), True),
            StructField("videos", videos_type, True),
            StructField("_extras", StringType(), True),
            StructField("qa_timestamp", StringType(), True),
            StructField("run_id", IntegerType(), True),
            StructField("spider_id", IntegerType(), True),
            StructField("product_name", StringType(), True),
            StructField("review_sentiment_score", DoubleType(), True),
            StructField("ingested_timestamp", StringType(), True),
            StructField("excerpts", exrepects, True),
            StructField("emotions", emotion_schema, True),
            StructField("dag_run_id", IntegerType(), True),
            StructField("brand", StringType(), True),
            StructField("product_image", StringType(), True),
            StructField("product_url", StringType(), True)
        ]
    )

    return schema


def sample_records(input_records, max_samples=300):
    df = pd.DataFrame(input_records)
    df['review_text_length'] = df['review_text'].fillna('').str.len() + df['review_title'].fillna('').str.len()
    df = df.sort_values(by='review_text_length', ascending=False).reset_index(drop=True)

    # Add a row number per group to enable round-robin sampling with group-order
    df['group_order'] = df.groupby('retailer_product_id').cumcount()
    df = df.sort_values(by=['group_order', 'review_text_length'], ascending=[True, False])
    sampled_df = df.head(max_samples).drop(columns=['review_text_length', 'group_order'])
    sampled_records = sampled_df.to_dict(orient='records')
    return sampled_records


ensure_resources_downloaded(nltk_resources)
