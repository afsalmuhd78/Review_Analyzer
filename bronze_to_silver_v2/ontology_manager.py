import copy
import logging
from typing import Dict

from attribute_clusterer import AttributeClusterer
from normalizer import Normalizer
from theme_clusterer import ThemeClusterer
# from concurrent.futures import ThreadPoolExecutor
from utils import sample_records

# Configure logging
FORMAT = "%(asctime)s %(levelname)s %(filename)s %(funcName)s(), line %(lineno)s *--------* : %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt="%y/%m/%d %H:%M:%S")
logger = logging.getLogger(__name__)


class OntologyManager:
    def __init__(self, ontology: Dict):
        self.ontology = ontology
        self.theme_mapping = {}
        self.attribute_mapping = {}

    def get_themes_in_ontology(self):
        all_themes = []
        for theme_group, themes in self.ontology.items():
            all_themes.extend(list(set(themes.keys())))
        return all_themes

    def add_theme_to_theme_group(self, theme_group, theme_name):
        if theme_group not in self.ontology:
            self.ontology[theme_group] = {}
        if theme_name not in self.ontology[theme_group]:
            self.ontology[theme_group][theme_name] = {}

    def add_attribute_to_sub_theme(self, theme: str, attribute: str, sub_theme: str):
        for theme_group, themes in self.ontology.items():
            sub_themes_and_attributes = themes.get(theme, {})
            if sub_themes_and_attributes:
                if sub_theme in sub_themes_and_attributes and attribute not in sub_themes_and_attributes[sub_theme]:
                    sub_themes_and_attributes[sub_theme].append(attribute)
                else:
                    sub_themes_and_attributes[sub_theme] = []
                    sub_themes_and_attributes[sub_theme].append(attribute)

    def get_existing_theme_groups(self):
        return {theme_group: list(themes.keys()) for theme_group, themes in self.ontology.items()}

    def create_ontology(self, processed_data):
        """
         Creating ontology with Sample processed data.
        """
        max_samples = 300  # Update here for the number of samples needed
        input_data_len = len(processed_data)
        logger.info(f"Ontology creation Initiated with {input_data_len} records..")
        if input_data_len > max_samples:
            processed_data = sample_records(input_records=processed_data, max_samples=max_samples)
            logger.warning(f"Sampling enabled in dataset with {len(processed_data)} records..")
        logger.info("Themes and Attributes generation initiated..")
        themes, themes_and_attributes = self.get_themes_and_attributes(processed_data=processed_data)
        logger.info("Themes and Attributes generated")
        theme_to_theme_group_mapping = self.cluster_themes(themes=themes)
        logger.info("ThemesGroup generated")
        theme_to_sub_theme, sub_theme_to_attributes = self.cluster_attributes(
            themes_and_attributes=themes_and_attributes
        )
        logger.info("Themes to Subtheme and Subtheme to Attribute generated")
        # Create the ontology JSON after parallel tasks complete
        final_ontology = self.create_ontology_json(
            processed_data, theme_to_theme_group_mapping, theme_to_sub_theme, sub_theme_to_attributes
        )
        logger.info("Ontology Json generated")
        self.ontology = final_ontology
        return copy.deepcopy(final_ontology)

    def create_ontology_json(
            self, processed_data: list, theme_to_theme_group_mapping, theme_to_sub_theme, sub_theme_to_attributes
    ):
        ontology = {}
        normalizer = Normalizer()
        for row in processed_data:
            themes_and_aspects = row["themes_and_aspects"]
            if not themes_and_aspects:
                continue
            for theme in themes_and_aspects["themes"]:
                theme_group = None
                theme_name = theme["theme_name"]
                og_theme = theme_name
                for key, value in theme_to_theme_group_mapping.items():
                    if theme_name in value:
                        theme_group = key
                    if not theme_group:
                        theme_group = "Miscellaneous"
                theme_group = normalizer.tokenize_and_lemmatize(theme_group).title()
                theme_name = normalizer.tokenize_and_lemmatize(theme_name).title()
                associated_dimensions = theme.get("associated_dimensions", [])
                for dimension in associated_dimensions:
                    attribute = dimension.get("dimension_name")
                    subtheme_name = None
                    if attribute in theme_to_sub_theme.get(og_theme, []):
                        subtheme_name = attribute
                    if not subtheme_name:
                        for key, value in sub_theme_to_attributes.get(og_theme, {}).items():
                            if attribute in value:
                                subtheme_name = key
                    if not subtheme_name or subtheme_name == "None":
                        subtheme_name = og_theme
                    subtheme_name = normalizer.tokenize_and_lemmatize(subtheme_name).title()
                    if theme_group not in ontology:
                        ontology[theme_group] = {}
                    if theme_name not in ontology[theme_group]:
                        ontology[theme_group][theme_name] = {}
                    if subtheme_name not in ontology[theme_group][theme_name]:
                        ontology[theme_group][theme_name][subtheme_name] = []
                    if attribute not in ontology[theme_group][theme_name][subtheme_name]:
                        ontology[theme_group][theme_name][subtheme_name].append(attribute)
        return ontology

    @staticmethod
    def get_themes_and_attributes(processed_data: list):
        """
        Returns themes and themes_and_attributes
        """
        themes = set()
        themes_and_attributes = {}
        for row in processed_data:
            themes_and_aspects = row['themes_and_aspects']
            for theme in themes_and_aspects.get("themes", []):
                themes.add(theme["theme_name"])
                associated_dimensions = theme.get("associated_dimensions", [])
                for dimension in associated_dimensions:
                    attribute = dimension.get("dimension_name", theme["theme_name"])
                    if not attribute:
                        continue
                    if theme["theme_name"] not in themes_and_attributes:
                        themes_and_attributes[theme["theme_name"]] = []
                    if attribute not in themes_and_attributes[theme["theme_name"]]:
                        themes_and_attributes[theme["theme_name"]].append(attribute)
        return list(themes), themes_and_attributes

    def cluster_themes(self, themes):
        """
        Clustering themes found in processed data
        """
        core_theme_groups = [
            "Packaging & Delivery",
            "Customer Support",
            "Pricing",
            "Design & Color",
            "Performance & Functionality",
            "Health & Safety",
            "Quality",
            "Weight & Size",
        ]
        theme_clusterer = ThemeClusterer()
        cluster_mappings = theme_clusterer.cluster_themes(
            themes_to_be_clustered=themes, theme_groups=core_theme_groups, limit_cluster_size=True
        )
        finalized_theme_mappings = theme_clusterer.validate_and_label_clusters(cluster_mappings=cluster_mappings)
        theme_groups_with_themes = {data["label"]: data["themes"] for data in finalized_theme_mappings}
        classified_themes = []
        for key, values in theme_groups_with_themes.items():
            classified_themes.extend([value for value in values])
        loop_ = True
        while loop_:
            theme_groups_with_themes = {data["label"]: data["themes"] for data in finalized_theme_mappings}
            classified_themes = []
            for key, values in theme_groups_with_themes.items():
                classified_themes.extend([value for value in values])
            outliers = list(set(themes) - set(classified_themes))
            if outliers:
                finalized_theme_mappings = theme_clusterer.process_themes_in_batches(
                    theme_groups_with_themes, themes_to_classify=outliers, labelled_clusters=finalized_theme_mappings
                )
            else:
                loop_ = False
        theme_groups_with_themes = {data["label"]: data["themes"] for data in finalized_theme_mappings}
        return theme_groups_with_themes

    def cluster_attributes(self, themes_and_attributes):
        """
        Clustering attributes
        """
        attribute_clusterer = AttributeClusterer()
        cluster_mappings = attribute_clusterer.cluster_attributes(themes_and_attributes=themes_and_attributes)
        finalized_attribute_mappings = attribute_clusterer.validate_and_label_clusters(cluster_mappings)
        to_add = []
        for index, data in enumerate(finalized_attribute_mappings):
            if data.get("outliers"):
                new_data = {
                    "original_theme": data["original_theme"],
                    "subtheme_name": data.get("new_group", "Miscalleneous"),
                    "themes": [],
                }
                for outlier in data["outliers"]:
                    if outlier in data["themes"]:
                        print(data["themes"].remove(outlier))
                    new_data["themes"].append(outlier)
                to_add.append(new_data)
        finalized_attribute_mappings.extend(to_add)
        valid = {}
        sub_theme_data = {}
        for data in finalized_attribute_mappings:
            if data["original_theme"] not in valid:
                valid[data["original_theme"]] = []
            if data["subtheme_name"] not in valid[data["original_theme"]]:
                valid[data["original_theme"]].append(data["subtheme_name"])
            if data["original_theme"] not in sub_theme_data:
                sub_theme_data[data["original_theme"]] = {}
            if data["subtheme_name"] not in sub_theme_data[data["original_theme"]]:
                sub_theme_data[data["original_theme"]][data["subtheme_name"]] = []
            sub_theme_data[data["original_theme"]][data["subtheme_name"]].extend(data["themes"])
        return valid, sub_theme_data
