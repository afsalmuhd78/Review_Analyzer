import concurrent
import json
from typing import Dict, List
import litellm
from tqdm import tqdm

from base_clusterer import BaseClusterer
from settings import LLM_MODEL

litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]


class AttributeClusterer(BaseClusterer):
    def __init__(self):
        super().__init__()
        self.cluster_labeling_prompt = """
                You are analyzing a single cluster derived from e-commerce product reviews using the k-means clustering algorithm. This cluster represents sub-aspects of a specific quality dimension related to customer feedback. Your tasks are:

                1. **Validate the Cluster**:
                   - **Objective**: Assess the cluster's validity using two key metrics: the silhouette score and the Davies-Bouldin Index.
                   - **Output**: Mark the cluster as valid if it has a high silhouette score and a low Davies-Bouldin Index, indicating good separation and cohesion.

                2. **Label the Cluster**:
                   - **Objective**: Assign a succinct, meaningful subtheme name that reflects the core aspect of the quality dimensions within this cluster. The subtheme should resonate with customer feedback and accurately encapsulate the cluster’s main aspect. It should be a noun or noun phrase of up to two words without using phrases like "ease of," "issues," or specific product details, and should not use the exact theme name.
                   - **Rules for Naming Subthemes**:
                     - **Conciseness**: Keep the subtheme name short, using 1-2 words.
                     - **Actionable Focus**: Reflect an observable or measurable product feature directly related to the quality dimensions.
                     - **Relevance**: Ensure the subtheme accurately captures the shared aspect of the quality dimensions.
                     - **Neutral Terminology**: Use descriptive language without emotional or subjective terms.
                     - **Distinct and Non-Overlapping**: The subtheme should uniquely represent a specific product aspect without overlapping with other subthemes or themes.
                     - **Avoid Phrases**: Exclude terms like "ease of," "issues," or the exact theme name.
                     - **Industry Appropriateness**: Tailor the subtheme to the product’s context while remaining understandable to a general audience.
                   - **Process**: Review the common patterns, key terms, and the overall tone of the quality dimensions. Choose a subtheme name that encapsulates the cluster's core aspect, providing actionable insights for improving customer experience.

                3. **Identify Outliers**:
                   - **Objective**: Detect and segregate outlier dimensions that do not align with the cluster's central aspect.
                   - **Process**: Evaluate each dimension in the cluster. Any dimension that significantly deviates from the main context should be classified as an outlier.

                4. **Suggest a New Group for Outliers**:
                   - **Objective**: Propose a suitable new group for the outliers based on their shared characteristics or distinct features.
                   - **Process**: Analyze the outlier dimensions and suggest a concise, appropriate group name that accurately reflects the common aspect of these outliers.

                5. **Provide a Comprehensive Explanation**:
                   - **Objective**: Explain the decisions made during the validation, subtheme naming, outlier identification, and new group suggestion processes.
                   - **Process**: Include details on:
                     - The cluster’s validity based on the calculated metrics.
                     - The rationale behind the chosen subtheme name.
                     - The reasoning for identifying specific dimensions as outliers.
                     - The justification for the suggested new group for the outliers.

                6. **Output the Results**:
                   - **Format**: Return a structured dictionary in JSON format containing:
                     - \\"subtheme_name\\": The subtheme name assigned to the cluster.
                     - \\"validity\\": A Boolean indicating if the cluster is valid.
                     - \\"explanation\\": A detailed summary covering validation, subtheme naming, outliers, and suggested new group.
                     - \\"outliers\\": A list of dimensions classified as outliers.
                     - \\"new_group\\": The suggested new group name for the outliers.

                **For Context**:
                Theme: {theme}
                Quality Dimensions: {quality_dimensions}
        """

    def cluster_attributes(self, themes_and_attributes: Dict[str, List]):
        grouped_attributes = {}
        for original_theme, attributes in themes_and_attributes.items():
            grouped_attributes[original_theme] = {}
            if len(attributes) > 10:
                cluster_mappings = self.cluster_phrases(
                    phrases_to_be_clustered=attributes, phrase_groups=[original_theme]
                )
                for cluster_num, cluster_values in cluster_mappings.items():
                    grouped_attributes[original_theme][cluster_num] = list(set(cluster_values))
            else:
                grouped_attributes[original_theme][0] = [
                    self.normalizer.tokenize_and_lemmatize(attribute) for attribute in set(attributes)
                ]
        return grouped_attributes

    def validate_and_label_clusters(self, cluster_mappings: Dict[str, Dict[str, List[str]]]):
        final_data = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for theme, to_be_grouped in cluster_mappings.items():
                for cluster_num, attributes in to_be_grouped.items():
                    futures.append(
                        executor.submit(self.process_attribute_cluster, cluster_num, list(attributes), theme)
                    )
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                final_response = future.result()
                final_data.append(final_response)
        finalized_groups = self.handle_outliers(final_data)
        return finalized_groups

    def process_attribute_cluster(self, idx, sub_themes, original_theme):
        """
        Attribute cluster with llm call
        """
        # themes_in_cluster = [themes[i] for i in cluster_indices]
        text = f"{','.join(sub_themes)}"

        # llm call
        response = litellm.completion(
            model= LLM_MODEL,
            messages=[
                {"role": "user", "content": self.cluster_labeling_prompt.format(theme=original_theme, quality_dimensions=text)},
            ],
            metadata={
                "generation_name": "attribute_clustrer",
                "trace_name": "process_attribute_cluster",
            },
            response_format={"type": "json_object"}
        )
        final_response = response.choices[0].message.content
        final_response = json.loads(final_response)

        final_response["themes"] = sub_themes
        final_response["original_theme"] = original_theme
        return final_response

    def handle_outliers(self, labelled_clusters: List[Dict]):
        to_add = []
        for data in labelled_clusters:
            if data.get("outliers"):
                new_data = {
                    "original_theme": data["original_theme"],
                    "subtheme_name": data.get("new_group", "Miscellaneous"),
                    "themes": [],
                }
                for outlier in data["outliers"]:
                    if outlier in data["themes"]:
                        data["themes"].remove(outlier)
                    new_data["themes"].append(outlier)
                to_add.append(new_data)
        labelled_clusters.extend(to_add)
        return labelled_clusters
