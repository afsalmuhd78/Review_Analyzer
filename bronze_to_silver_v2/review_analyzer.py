# import json
# import os
from typing import Dict, List
import time
from logging import getLogger
# import nltk
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import json
from redis import Redis, ConnectionPool
from datetime import datetime
from base_clusterer import BaseClusterer
from fit_and_merge import FitAndMerge
from llm_analyzer import LLMReviewTagger
from normalizer import Normalizer
from ontology_manager import OntologyManager
from theme_clusterer import ThemeClusterer
# from utils import get_sentiment_score
from sentiment import ReviewSentiment
from parts_processor import PartNameProcessor
from settings import CACHE_HOST
from lakehouse_utils.traces import Trace

trace = Trace()

# Redis connection
CONN_POOL = ConnectionPool(host=CACHE_HOST, port=6379, db=0)
REDIS_OBJ = Redis(connection_pool=CONN_POOL, health_check_interval=30)

# logger
logger = getLogger()


class ReviewAnalyzer:
    def __init__(self, existing_ontology):
        self.existing_ontology = existing_ontology
        self.clusterer = BaseClusterer()
        self.ontology_manager = OntologyManager(ontology=existing_ontology)
        self.llm_tagger = LLMReviewTagger()
        self.normalizer = Normalizer()
        self.sentiment_score = ReviewSentiment()

    def map_part_processor(self, raw_data, project_id):
        """
        Process and map parts from raw data, updating Redis with new mappings.
        """
        # Redis key for storing processed parts
        redis_key = f"mc_review_part_processor_{project_id}"
        existing_parts_map = REDIS_OBJ.get(redis_key)
        existing_parts_map = json.loads(existing_parts_map) if existing_parts_map else {}

        # Retrieve already processed parts
        existing_parts = list(existing_parts_map.keys())

        # Initialize the list for dimensions
        dimension_list = []

        # Extract dimensions from themes and aspects
        for row in raw_data:
            if row["themes_and_aspects"]:
                for themes in row.get("themes_and_aspects", {}).get("themes", []):
                    for dimention in themes.get("associated_dimensions", []):
                        if dimention:
                            dimension_list.append(dimention)

        # Create DataFrame for extracted dimensions
        parts_df = pd.DataFrame(dimension_list)

        # Normalize and clean the 'primary_physical_part' column
        try:
            parts_df['part'] = parts_df['primary_physical_part'].map(lambda x: 'general' if x == 'null' else x.lower())
            parts_df = parts_df.drop_duplicates(subset=['part'])

            print(f"Unique parts -------------- {set(parts_df['part'])}")
            parts_df = parts_df[~parts_df["part"].isin(existing_parts)]

            # Process parts using PartNameProcessor
            processor = PartNameProcessor(distance_threshold=0.5, min_bigram_freq=2)
            processor.load_data(parts_df, part_column="part")
            processor.process()
            mappings = processor.get_mappings()

            print(f"First mapping result: {mappings}")

            # Reverse mappings to update Redis
            new_mappings = {}
            for key, values in mappings.items():
                for value in values:
                    new_mappings[value] = key

            # Update the existing parts map with new mappings
            existing_parts_map.update(new_mappings)
            REDIS_OBJ.set(redis_key, json.dumps(existing_parts_map))
        except Exception as err:
            print(f"Error processing parts: {err}")

        return existing_parts_map

    def analyze_review(self, reviews: List[Dict[str, str]], project_id: int, run_id: int):
        """
        Take reviews input data as list of dict, process ontology and excerpts data
        """
        start_time = time.perf_counter()

        if not reviews:
            return []

        # Step 1: Get identified themes from existing ontology if available
        step_start = time.perf_counter()
        if self.existing_ontology:
            identified_themes = self.ontology_manager.get_themes_in_ontology()
        else:
            identified_themes = []
        logger.info(f"Time taken to get identified themes: {((time.perf_counter() - step_start) / 60):.4f} minutes")

        # Step 2: Process data using LLM tagger
        step_start = time.perf_counter()

        #trace - llm calls
        trace.set_trace(
            stage="bronze_to_silver_v2", event="process", event_phase="start", project=project_id, spider_run_id=run_id, notes="Theme & Emotion LLM Calls"
        )
        processed_data = self.llm_tagger.process_dataframe(reviews, identified_themes=identified_themes)
        logger.info(f"Time taken to process data with LLM tagger: {((time.perf_counter() - step_start) / 60):.4f} minutes")
        trace.set_trace(
            stage="bronze_to_silver_v2", event="process", event_phase="end", project=project_id, spider_run_id=run_id, notes="Theme & Emotion LLM Calls"
        )

        #trace - Parts process
        trace.set_trace(
            stage="bronze_to_silver_v2", event="process", event_phase="start", project=project_id, spider_run_id=run_id, notes="Process Parts Data"
        )
        parts_processor_map = self.map_part_processor(processed_data, project_id)
        logger.info("---------------------------------------------------")
        logger.warning(f" Parts clustering Result: {parts_processor_map}")
        trace.set_trace(
            stage="bronze_to_silver_v2", event="process", event_phase="end", project=project_id, spider_run_id=run_id, notes="Process Parts Data"
        )

        # Step 3: Create ontology if it doesn’t exist
        step_start = time.perf_counter()
        if not self.existing_ontology:

            #trace - Ontology Creation
            trace.set_trace(
                stage="bronze_to_silver_v2", event="process", event_phase="start", project=project_id, spider_run_id=run_id, notes="Create Ontology"
            )
            logger.info("Ontology Creation started --------")
            self.existing_ontology = self.ontology_manager.create_ontology(processed_data)
            logger.info(f"Ontology created: {self.existing_ontology}")
            trace.set_trace(
                stage="bronze_to_silver_v2", event="process", event_phase="end", project=project_id, spider_run_id=run_id, notes="Create Ontology"
            )
        logger.info(f"Time taken to create ontology (if applicable): {((time.perf_counter() - step_start) / 60):.4f} minutes")

        # Step 4: Save the processed review data
        step_start = time.perf_counter()

        #trace - Finalize excerpts
        trace.set_trace(
            stage="bronze_to_silver_v2", event="process", event_phase="start", project=project_id, spider_run_id=run_id, notes="Finalize Excerpts"
        )
        final_dataset = self.save_review(processed_data=processed_data, parts_processor_map=parts_processor_map)
        logger.info(f"Time taken to save processed review data: {((time.perf_counter() - step_start) / 60):.4f} minutes")
        trace.set_trace(
            stage="bronze_to_silver_v2", event="process", event_phase="end", project=project_id, spider_run_id=run_id, notes="Finalize Excerpts"
        )

        total_time = (time.perf_counter() - start_time) / 60
        logger.info(f"Total time taken for analyze_review: {total_time:.4f} minutes")

        return final_dataset

    def save_review(self, processed_data, parts_processor_map):
        """
        Taking input as themes_and_aspects processed review data.
        update ontology for processed reviews themes_and_aspects data
        create excerpts column using themes_and_aspects
        """
        fitter = FitAndMerge(existing_ontology=self.existing_ontology)
        # caching KMean cluster object for default theme_clusters
        self.clusterer.cached_kmeans(theme_clusters=fitter.theme_mapping)

        final_outputs = []

        def process_record(record):
            review_excerpts = record.pop("themes_and_aspects")
            # `fitter` and `self.finalize_excerpts` would need to be accessible within this function
            fitted_results, _ = self.finalize_excerpts(
                review_excerpts=review_excerpts, review=record, fitter=fitter, parts_processor_map=parts_processor_map
            )
            # updating `ingested_timestamp` and `review_sentiment_score`
            fitted_results["ingested_timestamp"] = datetime.now().isoformat()
            fitted_results["review_sentiment_score"] = float(self.sentiment_score.get_review_sentiment(
                fitted_results["rating"],fitted_results['review_title'], fitted_results["review_text"]
            ))
            return fitted_results

        with ThreadPoolExecutor(max_workers=16) as executor:  # Adjust max_workers as needed
            # Submit each record for processing
            futures = [executor.submit(process_record, row) for row in processed_data]

            # Collect results as they complete
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing rows"):
                final_outputs.append(future.result())

        return final_outputs

    def finalize_excerpts(self, review_excerpts, review, fitter, parts_processor_map):
        """
        Generating excerpts from the themes_and_aspects column and ontology update
        Returns: Final_result: {"review": review, "excerpts": [excerpts]}
        Returns: unfittable_results
        """
        review["excerpts"] = []
        unfittable_results = {
            "review": review,
            "missing_themes": [],
            "missing_attributes": {},
            "missing_excerpts": [],
        }
        generated_themes = review_excerpts.get("themes", [])
        if not generated_themes:
            review["excerpts"] = None
            return review, unfittable_results

        for theme in generated_themes:
            theme_name = theme.get("theme_name")
            theme_sentiment = theme.get("theme_sentiment")
            theme_sentiment_score = theme.get("theme_sentiment_score")
            for dimension in theme.get("associated_dimensions", []):
                skip_processing = False
                mention = dimension.get("mention")
                attribute = dimension.get("dimension_name", theme_name)
                dimension_sentiment = dimension.get("sentiment")
                theme_group, sub_theme = fitter.fit(theme_name, attribute)
                if not theme_group and theme_name not in unfittable_results["missing_themes"]:
                    try:
                        theme_group, prob, all_probs = self.clusterer.assign_label_to_new_element(
                            theme_name, fitter.theme_mapping, threshold=0.1
                        )
                    except:
                        theme_group = "Miscellaneous"
                    if theme_group == "Miscellaneous":
                        theme_clusterer = ThemeClusterer()
                        theme_group = theme_clusterer.assign_theme_to_theme_group(
                            theme_name, theme_groups=self.ontology_manager.get_existing_theme_groups()
                        )
                if not sub_theme:
                    try:
                        sub_theme, prob, all_probs = self.clusterer.assign_label_to_new_element(
                            attribute, self.fitter.attribute_mapping.get(theme_name, {}), threshold=0.1
                        )
                    except:
                        sub_theme = "Miscellaneous"

                unprocessed_part = dimension.get("primary_physical_part")
                part_name = parts_processor_map.get(unprocessed_part.lower())
                part_name = part_name if part_name else unprocessed_part

                if skip_processing:
                    unfittable_results["missing_excerpts"].append(theme)
                    continue
                try:
                    excerpt = {
                        "theme_group": self.normalizer.tokenize_and_lemmatize(theme_group).title(),
                        "theme": self.normalizer.tokenize_and_lemmatize(theme_name).title(),
                        "sub_theme": self.normalizer.tokenize_and_lemmatize(sub_theme).title(),
                        "attribute": self.normalizer.tokenize_and_lemmatize(attribute).title(),
                        "part": self.normalizer.tokenize_and_lemmatize(part_name).title(),
                        "theme_sentiment": theme_sentiment.title(),
                        "theme_sentiment_score": float(theme_sentiment_score) if theme_sentiment_score else 0.0,
                        "sub_theme_sentiment": dimension_sentiment.title(),
                        "llm_theme": theme_name,
                        "llm_attribute": attribute,
                        "llm_part_name": dimension.get("primary_physical_part"),
                        "excerpt": mention,
                    }
                    self.ontology_manager.add_theme_to_theme_group(
                        theme_name=excerpt["theme"], theme_group=excerpt["theme_group"]
                    )
                    self.ontology_manager.add_attribute_to_sub_theme(
                        theme=excerpt["theme"], sub_theme=excerpt["sub_theme"], attribute=excerpt["attribute"]
                    )
                    review["excerpts"].append(excerpt)
                except Exception as err:
                    print(f"Error while convering Excerpt: {err}")
    
        return review, unfittable_results
