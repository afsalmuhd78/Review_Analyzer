import os

LANGFUSE_HOST = os.getenv('LANGFUSE_HOST')
LANGFUSE_PUBLIC_KEY = os.getenv('LANGFUSE_PUBLIC_KEY')
LANGFUSE_SECRET_KEY = os.getenv('LANGFUSE_SECRET_KEY')
# MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
LLM_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o-mini"

PG_DB_NAME = os.getenv('DB_NAME')
PG_USER = os.getenv('DB_USER')
PG_PASS = os.getenv('DB_PASS')
PG_HOST = os.getenv('DB_HOST')
MC_PROJECT_META_API = os.getenv('MC_PROJECT_META_API')

REDIS_KEYS = {
    'theme': {
        'failed': 'mc_review_theme_llm_failed_project_{}',
        'success': 'mc_review_theme_llm_success_project_{}'
    },
    'emotion': {
        'failed': 'mc_review_emotion_llm_failed_project_{}',
        'success': 'mc_review_emotion_llm_success_project_{}'
    }
}

# Redis
REDIS_HOST = os.getenv("REDIS_HOST")
CACHE_HOST = os.getenv("CACHE_HOST")

system_prompt_theme = """
Analyze the provided ecommerce review, including its title, to identify and categorize the main themes and specific mentions that support each theme. Determine the sentiment and sentiment score for each theme and its associated mentions.

Guidelines for theme and sub-theme naming:

- Themes should be general nouns or noun phrases capturing overall aspects of the review, such as "quality" or "performance." Avoid using the term "product" and specific details.
- Sub-themes should be specific aspects of the themes, like "durability" or "speed," without repeating theme names or mentioning specific product details. It must be a direct sub-theme to the theme.
- Use consistent terminology across different reviews for similar themes and sub-themes.

Output Structure:

Provide a list of JSON objects, each representing a theme. Each theme object should include:

- theme: (string) General theme name in lowercase
- display_name: (string) Display name for the theme
- sentiment: (float) Overall sentiment score for the theme, ranging from -1 to 1
- sentiment_text: (string) Overall sentiment text for the theme (e.g., "positive," "negative," "neutral")
- mentions: (list of JSON objects) Each mention should include:
  - sub_theme: (string) Specific aspect of the theme
  - display_name: (string) Display name for the sub-theme
  - text: (string) Text from the review supporting the sub-theme
  - sentiment: (float) Sentiment score for the mention, ranging from -1 to 1
  - sentiment_text: (string) Overall sentiment text for the mention (e.g., "positive," "negative," "neutral")
  - confidence: (float) Confidence score for the mention, ranging from 0 to 1
Output only the JSON.
"""

user_prompt_theme = """
Review Header:
{review_header}
Review:
{review}
"""

system_prompt_emotion = """
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

user_prompt_emotion = """
Review: {review}
"""

plutchik_primary_emotions = ['joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust', 'anger', 'anticipation']
high_threshold = 0.9