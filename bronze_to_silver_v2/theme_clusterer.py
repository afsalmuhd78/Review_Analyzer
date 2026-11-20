import json
from typing import Dict, List
import litellm

from settings import LLM_MODEL
from base_clusterer import BaseClusterer

litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]


class ThemeClusterer(BaseClusterer):
    def __init__(self):
        super().__init__()
        self.cluster_labeling_prompt = """
        You are analyzing a single cluster derived from e-commerce product reviews using the k-means clustering algorithm. This cluster represents groups of themes related to customer feedback. Your tasks are:

        1. **Validate the Cluster**:
           - **Objective**: Assess the cluster's validity using two key metrics: the silhouette score and the Davies-Bouldin Index.
           - **Output**: Mark the cluster as valid if it has a high silhouette score and a low Davies-Bouldin Index, indicating good separation and cohesion.

        2. **Label the Cluster**:
           - **Objective**: Assign a succinct, meaningful label that reflects the core theme of the customer feedback within this group of themes. The label should resonate with customer experience terminology and be actionable, offering clear insights into customer sentiments or areas of focus. It should be a noun or noun phrase of up to two words that encapsulates the cluster's main theme without using phrases like "ease of," "issues," or specific product details.
           - **Rule for Label Creation**:
             - **Critical Incident Focus**: The label should capture the critical incidents that most significantly impact customer satisfaction or dissatisfaction, reflecting common patterns and key aspects of feedback. This aligns with the Critical Incident Technique, emphasizing events or features that drive customer perceptions of quality.
             - **Conciseness and Actionability**: The label should be concise, descriptive, and offer actionable insights, summarizing the cluster’s primary aspect in a way that is easily understandable and relevant to improving customer experience.
             - **Avoid Over-Specificity**: The label should avoid being too narrow; it must encompass the overall idea of the themes within the cluster, reflecting broader customer concerns or areas of focus.
             - **Process**: Consider the emotional tone, common patterns, and key terms in the feedback. Choose a label that a customer experience analyst might use to categorize and communicate key insights, such as "Navigation," "Aesthetics," or "Support Quality." The label should facilitate actionable insights for improving customer experience while being concise and precise.

        3. **Identify Outliers**:
           - **Objective**: Detect and segregate outlier themes that do not align with the cluster's central theme.
           - **Process**: Evaluate each theme in the cluster. Any theme that significantly deviates from the main context should be classified as an outlier and moved to a "Miscellaneous" category.

        4. **Provide a Comprehensive Explanation**:
           - **Objective**: Explain the decisions made during the validation, labeling, and outlier identification processes.
           - **Process**: Include details on:
             - The cluster’s validity based on the calculated metrics.
             - The rationale behind the chosen label, emphasizing how it reflects critical incidents impacting customer perceptions.
             - The reasoning for identifying specific themes as outliers.

        5. **Output the Results**:
           - **Format**: Return a structured dictionary in JSON format containing:
             - `"label"`: The label assigned to the cluster.
             - `"validity"`: A Boolean indicating if the cluster is valid.
             - `"explanation"`: A detailed summary covering validation, labeling, and outliers.
             - `"outliers"`: A list of themes classified as outliers.

        ### Example JSON Output:
        ```json
        {
          "label": "Aesthetics",
          "validity": true,
          "explanation": "The cluster shows a silhouette score of 0.85 and a Davies-Bouldin Index of 0.40, indicating strong cohesion and separation. Key terms like 'style,' 'color,' and 'appearance' suggest the label 'Aesthetics,' reflecting critical incidents related to visual appeal. The theme 'Durability' was identified as an outlier and placed in 'Miscellaneous' due to its lack of alignment with the aesthetic focus.",
          "outliers": ["Durability"]
        }
        ```
        """

        self.grouping_outliers_prompt = """
        You will be provided with a dictionary where the keys represent theme group names and the values are lists of themes already classified under those groups. You will also receive a separate list of themes that need to be classified into one of these groups.

        Your task is to:
        1. For each theme in the list that needs classification, go through each theme list in the provided theme groups to identify the group where the theme fits best based on similarity.
        2. Classify each theme into the most appropriate group. If a theme strongly resembles themes in more than one group, choose the group that makes the most sense contextually.
        3. If a theme does not fit well into any existing group, create a new group if it resembles other ungrouped themes, or classify it under 'Other' if it cannot be grouped.

        Return the output as a JSON object with the following structure:
        - "classified_themes": An object where each key is a theme group and the value is an array of themes that were newly classified into that group.
        - "new_groups": An object where each key is a newly created theme group (if any) and the value is an array of themes that were grouped under this new group.
        - "other": An array of themes that could not be grouped.

        Input Example:
        {
          "theme_groups": {
            "Marketing": ["SEO Optimization", "Content Strategy"],
            "Technology": ["AI", "Blockchain"],
            "Finance": ["Taxation", "Investment"]
          },
          "themes_to_classify": ["Social Media Marketing", "Machine Learning", "Tax Planning", "Augmented Reality", "Digital Transformation"]
        }

        Output Example:
        {
          "classified_themes": {
            "Marketing": ["Social Media Marketing"],
            "Technology": ["Machine Learning", "Augmented Reality", "Digital Transformation"],
            "Finance": ["Tax Planning"]
          },
          "new_groups": {},
          "other": []
        }

        Instructions:
        1. For each theme in "themes_to_classify", compare it against the existing themes in each theme group provided in the "theme_groups" dictionary.
        2. Assign the theme to the group it most closely aligns with based on the aspect that the themes already in that group represent.
        3. If the theme does not align with any group, create a new group if it shares similarities with other ungrouped themes. Otherwise, place it under the 'Other' category.
        4. Return the results in the specified JSON format, with themes categorized into existing groups, new groups created, or placed under 'Other' if necessary.
        """

    def cluster_themes(self, themes_to_be_clustered: List[str], theme_groups: List[str], limit_cluster_size=True):
        return self.cluster_phrases(
            phrases_to_be_clustered=themes_to_be_clustered,
            phrase_groups=theme_groups,
            limit_cluster_size=limit_cluster_size,
        )

    def handle_outliers(self, labelled_clusters):
        outliers = []
        for labelled_cluster in labelled_clusters:
            for outlier in labelled_cluster.get("outliers", []):
                if outlier in labelled_cluster["themes"]:
                    labelled_cluster["themes"].remove(outlier)
                    outliers.append(outlier)
        theme_groups = {data["label"]: data["themes"] for data in labelled_clusters}
        final_clusters = self.process_themes_in_batches(theme_groups, outliers, labelled_clusters)
        return final_clusters

    def validate_and_label_clusters(self, cluster_mappings: Dict[str, List[str]]):
        labelled_clusters = []
        for cluster_num, themes_in_cluster in cluster_mappings.items():
            user_prompt = f"{','.join(themes_in_cluster)}"

            # llm call
            response = litellm.completion(
                model= LLM_MODEL,
                messages=[
                    {"role": "system", "content": self.cluster_labeling_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                metadata={
                    "generation_name": "theme_clusterer",
                    "trace_name": "validate_and_label_clusters",
                },
                response_format={"type": "json_object"}
            )
            final_response = response.choices[0].message.content

            final_response = json.loads(final_response)
            final_response["themes"] = themes_in_cluster
            labelled_clusters.append(final_response)
        finalized_groups = self.handle_outliers(labelled_clusters=labelled_clusters)
        return finalized_groups

    def assign_theme_to_theme_group(self, new_theme, theme_groups):
        user_prompt = {"theme_groups": theme_groups, "themes_to_classify": new_theme}

        response = litellm.completion(
            model= LLM_MODEL,
            messages=[
                {"role": "system", "content": self.grouping_outliers_prompt},
                {"role": "user", "content": json.dumps(user_prompt)},
            ],
            metadata={
                "generation_name": "theme_clusterer",
                "trace_name": "assign_theme_to_theme_group",
            },
            response_format={"type": "json_object"}
        )
        final_response = response.choices[0].message.content

        final_response = json.loads(final_response)
        classified_themes = final_response["classified_themes"]
        new_groups = final_response["new_groups"]
        for key, value in new_groups.items():
            if new_theme in value:
                return key
        for c_tg, c_t in classified_themes.items():
            if new_theme in c_t:
                return c_tg
