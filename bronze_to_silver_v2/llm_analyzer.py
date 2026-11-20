import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from json import JSONDecodeError
from typing import Dict, List
import litellm
from redis import Redis, ConnectionPool
from tqdm import tqdm
from settings import LLM_MODEL, CACHE_HOST
from tenacity import retry, retry_if_exception_message, stop_after_attempt, wait_fixed
from validation import validate_or_fix_emotion_response, validate_or_fix_theme_response


litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]

# Redis connection
CONN_POOL = ConnectionPool(host=CACHE_HOST, port=6379, db=0)
REDIS_OBJ = Redis(connection_pool=CONN_POOL, health_check_interval=30)
ZSCORE_KEY = 'review_v2_llm_call_project_{}'


class LLMReviewTagger:
    def __init__(self):
        # self.llm = OpenAI(
        #     api_key='',
        #     model='gpt-4o-mini',
        #     reuse_client=False,
        #     additional_kwargs={"seed": 1424},
        # )
        self.system_prompt = """
        You are an assistant designed to extract themes and dimensions from product reviews, focusing on specific product features without including user experiences or emotions. Your primary task is to identify high-level themes related to product features and break down each theme into measurable dimensions. Additionally, identify the physical part or component being discussed in each mention, but do not include physical part names or component names in the themes or dimensions themselves. If no physical part is mentioned, return 'null' for the physical component. Reuse previously identified themes where relevant. Follow the naming rules for themes and dimensions to ensure clarity, relevance, and consistency.

        ### Rules for Naming Themes:
        1. **Conciseness**: Use 1-2 words that broadly capture the feedback aspect without unnecessary detail.
        2. **High-Level Focus**: Themes should represent broad categories of product features or attributes significant to overall user feedback.
        3. **Relevance**: Ensure the theme name directly reflects key product aspects relevant to customer satisfaction.
        4. **Avoid Physical Part Names**: Themes should not include names of parts, components, or user personas.
        5. **Neutral Terminology**: Use descriptive language that avoids emotional or subjective terms.
        6. **Distinct and Non-Overlapping**: Each theme should uniquely represent a specific product aspect without overlapping others.
        7. **Industry Relevance**: Themes should be clear within the product’s context while remaining general enough for broad feedback capture.
        8. **Avoid "Ease of"**: Theme names should not contain the phrase "ease of."
        9. **General Grouping Rules**: Group related feedback under broader themes based on the product's functional aspects:
           - **Delivery & Packaging Aspects**: Group feedback related to packaging, delivery, shipping, and handling under a broad theme such as **"Packaging"** or **"Delivery."**
           - **Performance Aspects**: Group feedback related to the functionality, efficiency, power, and operational performance of components under **"Performance."**
           - **Design Aspects**: Group feedback on aesthetics, appearance, style, and visual appeal under **"Design."**
           - **Usability Aspects**: Group feedback related to ease of use, installation, setup, and user interaction under **"Usability."**
           - **Durability Aspects**: Group feedback on material quality, build, wear, and long-term reliability under **"Durability."**
           - **Comfort Aspects**: Group feedback on user comfort, ergonomics, and fit under **"Comfort."**
           - **Safety Aspects**: Group feedback on safety features, protection, and risk prevention under **"Safety."**
           - **Maintenance Aspects**: Group feedback related to cleaning, upkeep, repair, and servicing under **"Maintenance."**
           - **Compatibility Aspects**: Group feedback on how well the product works with other items, devices, or systems under **"Compatibility."**
           - **Feature Aspects**: Group feedback on additional features, functions, or capabilities that enhance the product under **"Features."**
           - **Value Aspects**: Group feedback on price, cost-effectiveness, value for money, and affordability under **"Value."**

        ### Rules for Naming Dimensions:
        1. **Conciseness**: Keep names 1-2 words long for clarity.
        2. **Actionable Focus**: Reflect a directly measurable or observable aspect of the product.
        3. **Relevance**: Ensure dimensions are relevant to the product’s features or attributes.
        4. **Neutral Terminology**: Use unbiased language.
        5. **Avoid Duplication**: Ensure dimensions are unique and do not overlap.
        6. **Industry Appropriateness**: Tailor dimensions to fit the industry while being understandable to a general audience.
        7. **Avoid Physical Part Names**: Physical components should be identified separately, not within dimension names.
        8. **Avoid "Ease of"**: Dimension names should not contain the phrase "ease of."
        9. **Grouping Rules for Dimensions**:
           - **Performance Dimensions**: Group dimensions related to speed, efficiency, power, and reliability, e.g., **"Speed," "Power Efficiency," "Response Time."**
           - **Design Dimensions**: Include dimensions focusing on aesthetics, materials, and ergonomic factors, e.g., **"Visual Appeal," "Material Quality," "Ergonomic Fit."**
           - **Usability Dimensions**: Dimensions should relate to installation, ease of setup, and interaction, e.g., **"Installation Ease," "User Interaction."**
           - **Durability Dimensions**: Group dimensions on wear, material resilience, and overall build quality, e.g., **"Wear Resistance," "Material Strength."**
           - **Comfort Dimensions**: Focus on dimensions related to comfort, fit, and user ergonomics, e.g., **"Fit Comfort," "Ergonomic Support."**
           - **Safety Dimensions**: Group dimensions related to protection, risk mitigation, and safety features, e.g., **"Risk Prevention," "Safety Standards."**
           - **Maintenance Dimensions**: Include dimensions on cleaning, upkeep, and ease of repair, e.g., **"Cleaning Ease," "Repair Accessibility."**
           - **Compatibility Dimensions**: Dimensions related to integration with other systems, adaptability, e.g., **"System Integration," "Adaptability."**
           - **Feature Dimensions**: Group feedback on distinct or additional functionalities, e.g., **"Additional Functions," "Customization Options."**
           - **Value Dimensions**: Include dimensions related to pricing and cost-benefit analysis, e.g., **"Affordability," "Cost-Effectiveness."**

        ### Approach:
        Use the Critical Incident Technique (CIT) to identify significant product-related incidents impacting quality assessment. Themes and dimensions should align with this approach, ensuring they reflect specific product-related incidents as detailed in "Measuring Customer Satisfaction and Loyalty: Survey Design, Use, and Statistical Analysis Methods" by Bob E. Hayes.

        ### Output Structure:
        ```json
        {
          "themes": [
            {
              "theme_name": "Concise, broad aspect of product feedback",
              "theme_sentiment": "Overall sentiment towards the theme, one of positive, negative or neutral",
              "theme_sentiment_score": "Overall sentiment score for the theme, between -1 and 1, where value grater than 0.5 is positive, less than -0.5 is negative and anything in between is neutral",
              "associated_dimensions": [
                {
                  "dimension_name": "Specific product feature or attribute following rules",
                  "mention": "Exact review phrase mentioning this dimension",
                  "sentiment": "positive, negative, neutral",
                  "all_physical_parts": ["List of all physical parts or components related to this dimension"],
                  "primary_physical_part": "Primary physical part related to this dimension or 'General' if none"
                }
              ]
            }
          ]
        }

        ```
        """
        self.user_prompt = """
        Given the following details, extract themes and their associated dimensions focusing strictly on specific product features, ensuring no mention of user experiences, emotions, or physical part/component names within the theme or dimension names. Only extract themes that have direct mentions in the review text. Identify all physical parts and the primary physical part mentioned in each dimension and relate them to the appropriate theme, ensuring themes remain appropriately broad and distinct, using sub-groups where relevant. Ensure that themes and dimensions are named according to the rules provided, emphasizing features without including names of physical parts or components.

        - Product Name: "{product_name}"
        - Review Title: "{review_title}"
        - Review Text: "{review_text}"
        - Already Identified Themes: {identified_themes}

        ### Instructions:

        1. **Identify Themes:**
           - Extract themes that have direct mentions in the review and represent broad categories of product features, ensuring each theme captures the specific aspect of feedback without using names of physical parts or components.
           - Use the provided grouping guidelines and sub-groups to maintain broader themes:
             - For logistical aspects like packaging and shipping, group feedback under appropriate groups like **"Packaging"**,  **"Delivery."**etc
             - For performance-related feedback, use **"Performance"** and appropriate sub-groups like **"Speed"**, **"Power"**, and **"Reliability."**
             - Group design-related feedback under **"Design"**, with sub-groups like **"Appearance"**, **"Ergonomics"**, and **"Materials."**
             - Feedback on usability aspects should be under **"Usability"**, including sub-groups like **"Installation"** and **"User Interaction."**
             - Durability feedback falls under **"Durability"**, with sub-groups like **"Build Quality"** and **"Wear Resistance."**
             - For comfort, use **"Comfort"**, with sub-groups such as **"Fit"** and **"Feel."**
             - Safety-related feedback should be categorized under **"Safety,"** with sub-groups like **"Protection"** and **"Risk Prevention."**
             - Maintenance aspects go under **"Maintenance,"** with sub-groups like **"Cleaning"** and **"Repair."**
             - Compatibility feedback should be grouped under **"Compatibility,"** including sub-groups like **"Integration"** and **"Adaptability."**
             - For additional functions or capabilities, use **"Features"** with sub-groups like **"Additional Features"** and **"Customizability."**
             - Value-related feedback should fall under **"Value,"** with sub-groups like **"Pricing"** and **"Cost-Effectiveness."**
           - Avoid making themes overly specific or narrow; distinct features should have separate themes where necessary.

        2. **Specify Dimensions:**
           - For each identified theme, list associated dimensions with concise, actionable, relevant, neutral, and non-duplicative names.
           - Use sub-groups to maintain context and ensure dimensions reflect specific aspects of the broader theme.
           - Avoid any direct reference to physical part names or components within the dimension names.

        3. **Physical Parts Identification:**
           - Identify all physical parts mentioned in the review text and specify the primary physical part relevant to each dimension.
           - Relate the physical parts to themes and dimensions without naming them directly within these elements.

        4. **Critical Analysis:**
           - Critically evaluate the review to ensure no overlap between themes and that dimensions do not mistakenly include names of physical components.
           - Emphasize feature-specific aspects, separating distinct product features into unique themes based on general categories and sub-groups.

        5. **Mentions and Sentiments:**
           - Include any relevant mentions or sentiments associated with each theme and dimension.

        ### Output:
        - List the identified themes with direct mentions and their associated dimensions.
        - Clearly state the physical parts related to each dimension without incorporating these names into themes or dimension titles.
        """
        self.system_prompt_emotion = """
        Please analyze the following review text and provide the following information in JSON format:

        - sentiment: A float between -1 and 1 representing the overall sentiment score of the review.
        - sentiment_text: A string indicating the overall sentiment of the review (e.g., 'positive', 'negative', 'neutral').
        - plutchik_emotions: An array of strings listing all the relevant Plutchik emotions expressed in the review.
        - dominant_plutchik_emotions: An array of strings listing the dominant Plutchik emotions expressed in the review.
        - relevant_excerpts: An object containing arrays of excerpts for each identified Plutchik emotion. Each array should include objects with the following fields:
          - emotion: A string indicating the Plutchik emotion associated with the excerpt.
          - display_name: A human-readable name for the Plutchik emotion (e.g., 'Joy', 'Sadness').
          - text: The text of the excerpt.
          - polarity: A string indicating the polarity of the excerpt (e.g., 'positive', 'negative', 'neutral').
          - primary_emotion: A string indicating the primary Plutchik emotion most strongly expressed in the excerpt. This should be one of the eight primary Plutchik emotions (joy, trust, fear, surprise, sadness, disgust, anger, anticipation).
        Output only the JSON.
        """

        self.user_prompt_emotion = """
        Review: {review}
        """
        self.plutchik_primary_emotions = ['joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust', 'anger', 'anticipation']

    @retry(
        retry=retry_if_exception_message('RedisConnectionError'),
        stop=stop_after_attempt(5), wait=wait_fixed(2))
    def zscore_redis_update(self, project_id: int, process_name: str):
        zset_key = ZSCORE_KEY.format(project_id)

        try:
            REDIS_OBJ.zincrby(zset_key, 1, process_name)
        except Exception:
            raise Exception('RedisConnectionError')

    def generate_emotions(self, review_text, review_id, project):

        # no review_text return None
        if not review_text:
            return None

        review_text = review_text.replace("[This review was collected as part of a promotion.]", "").strip()
        user_prompt = self.user_prompt_emotion.format(review=review_text)

        try:
            # llm call
            response = litellm.completion(
                model= LLM_MODEL,
                messages=[
                    {"role": "system", "content": self.system_prompt_emotion},
                    {"role": "user", "content": user_prompt},
                ],
                metadata={
                    "generation_name": "llm_analyzer",
                    "trace_name": "generate_emotions",
                    "session_id": review_id
                },
                response_format={"type": "json_object"}
            )
            final_response = response.choices[0].message.content
            content = json.loads(final_response)
            content = validate_or_fix_emotion_response(review_text, content)
            self.zscore_redis_update(project_id=project, process_name="emotion_success")
            return content
        except Exception as exc:
            self.zscore_redis_update(project_id=project, process_name="emotion_failed")
            return None

    def tag_reviews(
        self, review_title: str, review_text: str, review_id: str, product_name: str, identified_themes: List, project: int
    ) -> Dict[str, str]:
        """
        Returns Excerpts Tagged with themes, attributes and parts from review in the following format
        {
          "themes": [
            {
              "theme_name": "Concise, broad aspect of product feedback",
              "associated_dimensions": [
                {
                  "dimension_name": "Specific product feature or attribute following rules",
                  "mention": "Exact review phrase mentioning this dimension",
                  "sentiment": "positive, negative, neutral",
                  "all_physical_parts": ["List of all physical parts or components related to this dimension"],
                  "primary_physical_part": "Primary physical part related to this dimension or 'General' if none"
                }
              ]
            }
          ]
        }
        """
        if not review_text:
            return {}
        user_chat_prompt = self.user_prompt.format(
            review_text=review_text,
            product_name=product_name,
            review_title=review_title,
            identified_themes=f"{identified_themes}",
        )

        try:
            # llm call
            response = litellm.completion(
                model= LLM_MODEL,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_chat_prompt},
                ],
                metadata={
                    "generation_name": "llm_analyzer",
                    "trace_name": "tag_reviews",
                    "session_id": review_id
                },
                response_format={"type": "json_object"}
            )
            final_response = response.choices[0].message.content
            content = json.loads(final_response)
            content = validate_or_fix_theme_response(review_text, review_title, content)
            self.zscore_redis_update(project_id=project, process_name="themes_success")
            return content
        except Exception as exc:
            self.zscore_redis_update(project_id=project, process_name="themes_failed")
            return  {}

    def process_dataframe(
        self,
        reviews,
        review_title_key="review_title",
        review_text_key="review_text",
        product_name_key="product_name",
        identified_themes=None,
    ):
        # Initialize identified themes as a set for faster lookups
        if identified_themes is None:
            identified_themes = []
        identified_themes = set(identified_themes)
        lock = threading.Lock()

        def process_row(row):
            max_attempts = 2
            themes_and_aspects = {}
            emotions = None

            # Retry for `tag_reviews` up to 2 times
            for attempt in range(max_attempts):
                themes_and_aspects = self.tag_reviews(
                    review_title=row[review_title_key],
                    review_text=row[review_text_key],
                    review_id=row['review_id'],
                    product_name=row.get(product_name_key, ""),
                    identified_themes=list(identified_themes),  # Pass as a list to avoid thread issues
                    project=row["project"]
                )
                if themes_and_aspects:
                    break

            # Retry for `generate_emotions` up to 2 times
            for attempt in range(max_attempts):
                emotions = self.generate_emotions(
                    review_text=row[review_text_key],
                    review_id=row['review_id'],
                    project=row["project"]
                )
                if emotions:
                    break

            # Extract themes and update the shared identified_themes set safely
            themes = themes_and_aspects.get("themes", []) if themes_and_aspects else []
            with lock:  # Acquire lock to update the shared set safely
                for theme in themes:
                    identified_themes.add(theme["theme_name"])

            # Update the row with the results
            row["themes_and_aspects"] = themes_and_aspects
            row["emotions"] = emotions
            return row

        processed_rows = []
        # Using ThreadPoolExecutor to parallelize row processing
        with ThreadPoolExecutor(max_workers=16) as executor:  # Adjust max_workers based on your system's capability
            # Submit each row for processing
            futures = [executor.submit(process_row, row) for row in reviews]

            # Collect results as they complete
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing rows"):
                processed_rows.append(future.result())

        return processed_rows
