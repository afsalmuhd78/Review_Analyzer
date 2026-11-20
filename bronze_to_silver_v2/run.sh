#!/bin/bash

# Load environment variables
if [ -f .env ]; then
    set -a && source .env && set +a
fi

# Ensure Environment variables are in the .env file
# List of environment variables to check
mandatory_vars=("BRONZE_AWS_REGION" "BRONZE_AWS_ACCESS_KEY" "BRONZE_AWS_SECRET_KEY" "BRONZE_WAREHOUSE"
"BRONZE_NESSIE_URI" "BRONZE_S3_ENDPOINT" "MASTER_URL" "NUM_EXECUTORS" "EXECUTOR_CORES" "TOTAL_EXECUTOR_CORES"
"EXECUTOR_MEMORY" "DRIVER_MEMORY" "REDIS_HOST" "ENVIRONMENT" "SILVER_NESSIE_URI" "SILVER_AWS_REGION" "SILVER_WAREHOUSE"
"SILVER_S3_ENDPOINT" "SILVER_AWS_ACCESS_KEY" "SILVER_AWS_SECRET_KEY" "LANGFUSE_HOST" "LANGFUSE_SECRET_KEY"
"LANGFUSE_PUBLIC_KEY" "MISTRAL_API_KEY" "DB_NAME" "DB_USER" "DB_PASS" "DB_HOST" "ADVANCED_PROJECTS_API" "GITLAB_MODEL_TOKEN")

# Loop through each variable and check if it is set
for var in "${mandatory_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "Environment variable $var is not set. Please check your .env file."
        exit 1
    fi
done

BASEDIR=$(pwd)
export PYSPARK_PYTHON="$BASEDIR/venv/bin/python3"
export PYSPARK_DRIVER_PYTHON="$BASEDIR/venv/bin/python3"
export SPARK_HOME="/opt/spark"

# SparkSubmit command
spark_command=$(cat <<'EOF'
spark-submit \
--master $MASTER_URL \
--num-executors $NUM_EXECUTORS \
--executor-cores $EXECUTOR_CORES \
--total-executor-cores $TOTAL_EXECUTOR_CORES \
--executor-memory $EXECUTOR_MEMORY \
--driver-memory $DRIVER_MEMORY \
--conf spark.streaming.stopGracefullyOnShutdown=true \
--conf "spark.driver.extraJavaOptions=-Daws.region=$BRONZE_AWS_REGION -Daws.accessKeyId=$BRONZE_AWS_ACCESS_KEY -Daws.secretAccessKey=$BRONZE_AWS_SECRET_KEY" \
--conf "spark.executor.extraJavaOptions=-Daws.region=$BRONZE_AWS_REGION -Daws.accessKeyId=$BRONZE_AWS_ACCESS_KEY -Daws.secretAccessKey=$BRONZE_AWS_SECRET_KEY" \
--packages org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.4.3,\
org.apache.iceberg:iceberg-spark-extensions-3.4_2.13:1.4.3,\
org.projectnessie.nessie-integrations:nessie-spark-extensions-3.3_2.12:0.67.0,\
software.amazon.awssdk:bundle:2.20.18 \
--conf spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions,\
org.projectnessie.spark.extensions.NessieSparkSessionExtensions \
--py-files attribute_clusterer.py,base_clusterer.py,fit_and_merge.py,llm_analyzer.py,normalizer.py,\
ontology_manager.py,review_analyzer.py,settings.py,theme_clusterer.py,utils.py,sentiment.py,model_download.py main.py --run_id $1
EOF)


echo "$spark_command"
eval "$spark_command"
