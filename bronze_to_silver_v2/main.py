#!/usr/bin/python3
import argparse
import logging
import os
import json
import traceback
import psycopg2
from os import environ
from datetime import datetime

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, col

from utils import get_reviews_silver_table_schema, is_advanced_project
from review_analyzer import ReviewAnalyzer
from sentiment import ReviewSentiment
from lakehouse_utils.traces import Trace

trace = Trace()
# postgres connection
PG_DB_NAME = environ['DB_NAME']
PG_USER = environ['DB_USER']
PG_PASS = environ['DB_PASS']
PG_HOST = environ['DB_HOST']

ONTOLOGY_UPDATE = False

# Postgres connection
conn = psycopg2.connect(dbname=PG_DB_NAME, user=PG_USER, password=PG_PASS, host=PG_HOST)

# Configure logging
FORMAT = "%(asctime)s %(levelname)s %(filename)s %(funcName)s(), line %(lineno)s *--------* : %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt="%y/%m/%d %H:%M:%S")
logger = logging.getLogger(__name__)

# load secret configurations
warehouse_conf = {
    "bronze": {
        "nessie": os.getenv("BRONZE_NESSIE_URI"),
        "s3-endpoint": os.getenv("BRONZE_S3_ENDPOINT"),
        "warehouse": os.getenv("BRONZE_WAREHOUSE"),
        "access-key": os.getenv("BRONZE_AWS_ACCESS_KEY"),
        "secret-key": os.getenv("BRONZE_AWS_SECRET_KEY")
    },
    "silver": {
        "nessie": os.getenv("SILVER_NESSIE_URI"),
        "s3-endpoint": os.getenv("SILVER_S3_ENDPOINT"),
        "warehouse": os.getenv("SILVER_WAREHOUSE"),
        "access-key": os.getenv("SILVER_AWS_ACCESS_KEY"),
        "secret-key": os.getenv("SILVER_AWS_SECRET_KEY")
    }
}


def create_ontology(cursor, project_id):
    """
    Create an ontology based on theme_group, theme, sub_theme, and attributes.
    
    Args:
        cursor: Database cursor to execute SQL queries.
        project_id: Integer representing the project ID for which the ontology is built.
    
    Returns:
        A dictionary representing the ontology.
    """
    # Execute the SQL query using the correct column names
    query = """
    SELECT 
        tg.name AS theme_group_name, 
        t.name AS theme_name,         
        st.name AS sub_theme_name,    
        a.name AS attribute_name     
    FROM 
        theme_group tg
    JOIN 
        theme t ON t.theme_group = tg.id
    LEFT JOIN 
        Subtheme st ON st.theme = t.id
    LEFT JOIN 
        attribute a ON a.Subtheme = st.id
    WHERE 
        tg.project = %s;
    """
    cursor.execute(query, (project_id,))
    rows = cursor.fetchall()

    ontology = {}
    try:
        # Build the ontology structure
        for row in rows:
            theme_group = row[0]
            theme = row[1]
            sub_theme = row[2]
            attribute = row[3]
            
            if theme_group not in ontology:
                ontology[theme_group] = {}

            if theme not in ontology[theme_group]:
                ontology[theme_group][theme] = {}

            if sub_theme not in ontology[theme_group][theme]:
                ontology[theme_group][theme][sub_theme] = []

            # Add attribute to the sub_theme list if it exists
            if attribute and attribute not in ontology[theme_group][theme][sub_theme]:
                ontology[theme_group][theme][sub_theme].append(attribute)
    except Exception as err:
        logger.warning("Ontology creation failed Error:")
        logger.exception(err)

    return ontology


def read_args():
    parser = argparse.ArgumentParser(description="PySpark job run")
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        dest="run_id",
        help="The run id of a spider to query the dataset from bronze.reviews",
    )
    args = parser.parse_args()
    logger.info(f"Called with arguments: {args}")
    return args


def load_spark_session(collection, app_name=None):
    if app_name:
        spark_conf = pyspark.SparkConf().setAppName(app_name)
    else:
        spark_conf = pyspark.SparkConf()
    spark_conf = (spark_conf
                  .set("spark.sql.catalog.nessie", "org.apache.iceberg.spark.SparkCatalog")
                  .set("spark.sql.catalog.nessie.uri", warehouse_conf.get(collection).get("nessie"))
                  .set("spark.sql.catalog.nessie.ref", "main")
                  .set("spark.sql.catalog.nessie.authentication.type", "NONE")
                  .set("spark.sql.catalog.nessie.catalog-impl", "org.apache.iceberg.nessie.NessieCatalog")
                  .set("spark.sql.catalog.nessie.s3.endpoint", warehouse_conf.get(collection).get("s3-endpoint"))
                  .set("spark.sql.catalog.nessie.warehouse", warehouse_conf.get(collection).get("warehouse"))
                  .set("spark.sql.catalog.nessie.io-impl", "org.apache.iceberg.aws.s3.S3FileIO")
                  .set("spark.hadoop.fs.s3a.access.key", warehouse_conf.get(collection).get("access-key"))
                  .set("spark.hadoop.fs.s3a.secret.key", warehouse_conf.get(collection).get("secret-key"))
                  )
    spark_session = (
        SparkSession.builder
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
        .config(conf=spark_conf)
        .getOrCreate()
    )
    return spark_session


def extract(spark_session, run_id, project=None):
    if not project:
        sp_df = spark_session.sql(f"""SELECT * FROM nessie.{SOURCE_DB}.reviews WHERE run_id = {run_id}""")
        logger.info(f"Spark Dataframe created for run_id : {run_id}")
    else:
        sp_df = spark_session.sql(f"""SELECT * FROM nessie.{SOURCE_DB}.reviews WHERE project = {project}""")
        logger.info(f"Spark Dataframe created for Project : {project}")
    logger.info(f"Spark df count: {sp_df.count()}")
    return [row.asDict() for row in sp_df.collect()]


def load_dataset(sdf):
    logger.info("Iceberg Write Started...")
    try:
        sdf.write.format("iceberg").mode("append").save(f"nessie.{SINK_DB}.review")
    except Exception as e:
        logger.warning(f"Sample Transformed DataRow: {sdf.first().asDict()}")
        raise Exception(traceback.print_exc())
    logger.info("Iceberg Write Completed.")


""" ---------------------------------------------------------------------------------------------------------------"""


if __name__ == "__main__":
    # Iceberg details
    SOURCE_DB = "bronze"
    SINK_DB = "silver"
    TABLE_NAME = "reviews"

    # load secret configurations
    args = read_args()
    run_id = args.run_id
    logger.info(f"Input RunId: {run_id}.")

    # Define Spark Session
    ss_read = load_spark_session(collection=SOURCE_DB,
                                 app_name=f"mc-reviews-pipeline-bronze2silver-Read-{run_id}-v2")
    logger.info(f"SparkSession created for Warehouse {SOURCE_DB}.")

    # Extract
    review_data = extract(spark_session=ss_read, run_id=run_id)
    logger.info(f"Extracted rows from Iceberg: {len(review_data)}")

    if len(review_data) == 0:
        logger.error("No Rows extracted from Iceberg!")
        # sys.exit(1)

    project_id = review_data[0]['project']
    logger.warning(f"Project_id {project_id} | run_id: {run_id}")

    #trace
    trace.set_trace(stage="bronze_to_silver_v2", event="data", event_phase="start", project=project_id, spider_run_id=run_id, meta={
        "type": "batch",
        "source": "archive.bronze.reviews",
        "count": len(review_data)
    })

    # collecting project data count
    total_count_df = ss_read.sql(f"""SELECT COUNT(*) AS total FROM nessie.{SOURCE_DB}.reviews WHERE project = {int(project_id)}""")
    project_data_count = int(total_count_df.collect()[0]["total"])
    logger.info(f"Processing review analyzer for project: {project_id}")
    sentiment_obj = ReviewSentiment()

    # for advance project False case
    if not is_advanced_project(project_id):
        # review_data = extract_product_data(review_data)
        final_data = [
            {
                **row,
                "excerpts": None,
                "emotions": None,
                "review_sentiment_score": float(sentiment_obj.get_review_sentiment(
                        row["rating"], row["review_title"] ,row['review_text']
                )),
                "ingested_timestamp": datetime.now().isoformat(),
            } for row in review_data
        ]
    else:
        if project_data_count >= 100 and project_data_count - len(review_data) < 100:
            review_data = extract(spark_session=ss_read, run_id=run_id, project=project_id)

            logger.info(f"Combined data from Bronze table for Project: {project_id}, Count: {len(review_data)}")
            existing_ontology = {}
            ONTOLOGY_UPDATE = True
        else:
            cursor = conn.cursor()
            existing_ontology = create_ontology(cursor, project_id)
            # pg con`nection close
            cursor.close()
            conn.close()
        logger.info(f"Existing ontology for Project: {project_id} is : {existing_ontology}")

        #Extract product columns from _extras
        # review_data = extract_product_data(review_data)

        # review analyzer v2
        review_analyzer = ReviewAnalyzer(existing_ontology=existing_ontology)
        final_data = review_analyzer.analyze_review(review_data, project_id, run_id)

    if len(final_data) == 0:
        logger.error("No Rows available after transforming!")

    # Session Recreate
    ss_read.stop()
    ss_write = load_spark_session(collection=SINK_DB,
                                  app_name=f"mc-reviews-pipeline-bronze2silver-Write-{run_id}-v2")
    logger.info(f"New Session created for mc-production")

    logger.info("Final Dataset to Pyspark DF conversion Initiated...")
    sdf = ss_write.createDataFrame(final_data, schema=get_reviews_silver_table_schema())
    logger.info("Final Dataset to Pyspark DF conversion Completed")

    logger.info("IceBerg Compatibility: str ingested_timestamp to_timestamp Started...")
    sdf = sdf.withColumn("ingested_timestamp", to_timestamp(col("ingested_timestamp"), "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"))
    logger.info("IceBerg Compatibility: str ingested_timestamp to_timestamp Completed")
    sdf = sdf.withColumn("qa_timestamp", to_timestamp(col("qa_timestamp"), "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"))

    try:
        # delete and insert updated project data
        if ONTOLOGY_UPDATE:
            ss_write.sql(f"""DELETE FROM nessie.{SINK_DB}.review WHERE project = {project_id};""")
        # Load
        load_dataset(sdf)

        # trace - Silver Write
        trace.set_trace(stage="bronze_to_silver_v2", event="data", event_phase="end", project=project_id, spider_run_id=run_id, meta={
            "type": "batch",
            "count": len(final_data),
            "destination": "production.silver.review",
        })

    except (Exception, KeyboardInterrupt, SystemExit) as expt:
        logger.error(traceback.print_exc())
        logger.warning(f"Executing DELETE Query for record under run_id: {run_id}")
        ss_write.sql(f"""DELETE FROM nessie.{SINK_DB}.review WHERE run_id = {run_id};""")
        logger.warning("DELETE Query executed successfully!")
        raise Exception(f"Deleting Data under RunID: {run_id} because of {type(expt)}!")

    finally:
        logger.info("Stopping the session..........")
        ss_write.stop()
        logger.info("Session Stopped Successfully!")

    logger.info("Completed Bronze to Silver Pipeline")
