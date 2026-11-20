import copy
import json
import string
from typing import Dict, List

import nltk
import numpy as np
import spacy
import litellm
from settings import LLM_MODEL
from gensim.models import FastText

# from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from tqdm import tqdm
from nltk.corpus import wordnet
from itertools import chain
from hashlib import sha1

from normalizer import Normalizer

litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]


synsets = {}
lemmas_in_wordnet = set(chain(*[x.lemma_names() for x in wordnet.all_synsets()]))

for lemma in lemmas_in_wordnet:
    synsets[lemma] = wordnet.synsets(lemma)


def get_synsets(word):
    try:
        return synsets[word.lower()]
    except Exception:
        return []


class BaseClusterer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.sbert_model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)
        self.normalizer = Normalizer()
        self.weight_sentence = 0.4
        self.weight_avg_word = 0.7
        self.weight_fasttext = 0.5
        self.weight_synset = 0.3
        # self.llm = OpenAI(
        #     model='gpt-4o-mini',
        #     reuse_client=False,
        #     temperature=0.5,
        #     additional_kwargs={"seed": 1424},
        # )
        self.cluster_labeling_prompt = None
        self.grouping_outliers_prompt = None
        self.pca_model = None
        self.KMEANS_CACHED = {}

    def cached_kmeans(self, theme_clusters: Dict[str, List[str]]):

        theme_cluster_hash = sha1(json.dumps(theme_clusters).encode("utf-8")).hexdigest()

        if self.KMEANS_CACHED.get(theme_cluster_hash) is not None:
            print(f"Returned {theme_cluster_hash} from cache ...")
            return self.KMEANS_CACHED[theme_cluster_hash]

        # Prepare data for KMeans training
        all_phrases = []
        labels = []
        label_indices = {}  # To map label indices for probability calculation
        index = 0

        # Collect phrases and map indices to labels
        for label, phrases in theme_clusters.items():
            normalized_phrases = [self.normalizer.tokenize_and_lemmatize(phrase) for phrase in phrases]
            all_phrases.extend(normalized_phrases)
            labels.extend([label] * len(phrases))
            label_indices[label] = index
            index += 1

        # Generate embeddings for all cluster phrases
        cluster_embeddings, cluster_embeddings_normalized = self.get_embeddings(all_phrases)

        # Normalize the cluster embeddings
        cluster_embeddings_normalized = normalize(cluster_embeddings, axis=1)

        # Determine the number of clusters
        n_clusters = len(theme_clusters)

        # Train the KMeans model on normalized embeddings
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(cluster_embeddings_normalized)

        self.KMEANS_CACHED[theme_cluster_hash] = kmeans.cluster_centers_
        print(f"Added {theme_cluster_hash} to cache")
        return kmeans

    def handle_outliers(self, labelled_clusters: List[Dict]):
        raise NotImplementedError

    def validate_and_label_clusters(self, cluster_mappings: Dict[str, List[str]]):
        raise NotImplementedError

    def assign_label_to_new_element(
        self, new_element: str, theme_clusters: Dict[str, List[str]], threshold: float = 0.5
    ):
        # Get the cluster centroids
        centroids = self.cached_kmeans(theme_clusters=theme_clusters)

        # Generate embeddings for the new element using the existing embedding model
        new_element_embeddings, _ = self.get_embeddings(
            [self.normalizer.tokenize_and_lemmatize(new_element)], phrase_groups=[]
        )

        # Normalize the new element embeddings
        new_element_embeddings_normalized = normalize(new_element_embeddings, axis=1)

        # Calculate distances between the new element and each cluster centroid
        distances = [np.linalg.norm(new_element_embeddings_normalized - centroid) for centroid in centroids]

        # Convert distances to probabilities (smaller distances imply higher probabilities)
        inverse_distances = np.array([1 / (1 + d) for d in distances])  # Avoid division by zero
        probability_scores = inverse_distances / inverse_distances.sum()  # Normalize to get probabilities

        # Map probabilities back to corresponding labels
        label_probabilities = {list(theme_clusters.keys())[i]: prob for i, prob in enumerate(probability_scores)}

        # Find the best label and its probability
        best_label = max(label_probabilities, key=label_probabilities.get)
        best_probability = label_probabilities[best_label]

        # If the confidence is below the threshold, label as Miscellaneous
        if best_probability < threshold:
            best_label = "Miscellaneous"

        return best_label, best_probability, label_probabilities

    def cluster_phrases(self, phrases_to_be_clustered: List[str], phrase_groups=None, limit_cluster_size=True):
        # TODO: Move relevant configurations in this class to dynaconf
        adjusted_combined_features, combined_features_normalized = self.get_embeddings(
            phrases_to_be_clustered, phrase_groups
        )

        best_cluster_size = self.find_optimal_number_of_clusters(adjusted_combined_features)
        # Initialize GMM with the best cluster size
        gmm_final = GaussianMixture(n_components=best_cluster_size, random_state=0)

        # Fit the GMM model with the adjusted combined features
        gmm_final.fit(adjusted_combined_features)
        # Predict the final clusters
        final_clusters = gmm_final.predict(adjusted_combined_features)
        if limit_cluster_size:
            final_clusters_reduced = self.recluster_large_clusters_gmm(
                combined_features_normalized, final_clusters, max_elements_per_cluster=6
            )
        else:
            final_clusters_reduced = final_clusters
        cluster_mappings = {}
        for cluster_num in set(final_clusters_reduced):
            cluster_mappings[cluster_num] = []
            cluster_indices = np.where(final_clusters_reduced == cluster_num)[0]
            for index in cluster_indices:
                if phrases_to_be_clustered[index] not in cluster_mappings[cluster_num]:
                    cluster_mappings[cluster_num].append(phrases_to_be_clustered[index])
        return cluster_mappings

    def get_embeddings(self, phrases_to_be_clustered, phrase_groups=None, n_components=None):
        if phrase_groups is None:
            phrase_groups = []
        sentence_embeddings, avg_word_embeddings, avg_synset_embeddings, processed_themes = (
            self.preprocess_themes_with_relevant_noun_synsets(phrases_to_be_clustered)
        )

        fasttext_embeddings = self.get_fasttext_embeddings(processed_themes)
        combined_features = np.hstack(
            (
                self.weight_sentence * sentence_embeddings,
                self.weight_avg_word * avg_word_embeddings,
                self.weight_fasttext * fasttext_embeddings,
            )
        )
        # Reduce dimensions of combined features to a common size
        # If no PCA model is provided, fit a new PCA model
        # if self.pca_model is None:
        #     if not n_components:
        #         n_components = min(100, len(phrases_to_be_clustered))
        #     self.pca_model = PCA(n_components=n_components)
        #     combined_features_reduced = self.pca_model.fit_transform(combined_features)
        # else:
        #     # Use the existing PCA model for transformation
        #     combined_features_reduced = self.pca_model.transform(combined_features)
        # Normalize the combined features
        combined_features_normalized = normalize(combined_features)
        if phrase_groups:
            core_sentence_embeddings, core_avg_word_embeddings, core_synset_embeddings, _ = (
                self.preprocess_themes_with_relevant_noun_synsets(phrase_groups)
            )
            core_fasttext_embeddings = self.get_fasttext_embeddings(phrase_groups)
            # Combine core theme embeddings
            core_combined_features = np.hstack(
                (
                    self.weight_sentence * core_sentence_embeddings,
                    self.weight_avg_word * core_avg_word_embeddings,
                    self.weight_fasttext * core_fasttext_embeddings,
                )
            )
            core_combined_features_normalized = normalize(core_combined_features)
            # Apply the same PCA transformation to the core theme embeddings
            # core_combined_features_reduced = self.pca_model.transform(core_combined_features)
            # Adjust the combined features based on similarity to core themes
            adjusted_combined_features = self.adjust_features_with_similarity(
                combined_features_normalized, core_combined_features_normalized
            )
        else:
            adjusted_combined_features = combined_features_normalized
        return adjusted_combined_features, combined_features_normalized

    def get_average_word_embedding_with_synsets(self, phrase):
        doc = self.nlp(phrase)
        word_embeddings = []

        for token in doc:
            # Get the word embedding
            word_embedding = self.sbert_model.encode(token.text, convert_to_tensor=True).cpu().numpy()
            word_embeddings.append(word_embedding)

        # Calculate the average word embedding
        avg_word_embedding = (
            np.mean(word_embeddings, axis=0)
            if len(word_embeddings) > 0
            else np.zeros(self.sbert_model.get_sentence_embedding_dimension())
        )

        return avg_word_embedding

    @staticmethod
    def get_root_hypernym(synset):
        """Trace the synset back to its most root form."""
        current_synset = synset
        while current_synset.hypernyms():  # Keep going up the hierarchy if there are hypernyms
            current_synset = current_synset.hypernyms()[0]
        return current_synset

    def get_synset_embedding(self, word):
        # synsets = wn.synsets(word)
        synsets = get_synsets(word)
        if synsets:
            root_synset = self.get_root_hypernym(synsets[0])  # Use the most root synset
            return self.sbert_model.encode(root_synset.name().split(".")[0], convert_to_tensor=True).cpu().numpy()
        else:
            return self.sbert_model.encode(word, convert_to_tensor=True).cpu().numpy()

    def get_average_synset_embedding(self, phrase):
        doc = self.nlp(phrase)
        synset_embeddings = [self.get_synset_embedding(token.text) for token in doc]
        return (
            np.mean(synset_embeddings, axis=0)
            if len(synset_embeddings) > 0
            else np.zeros(self.sbert_model.get_sentence_embedding_dimension())
        )

    def preprocess_themes_with_relevant_noun_synsets(self, themes):
        stop_words = set(nltk.corpus.stopwords.words("english"))
        lemmatizer = nltk.stem.WordNetLemmatizer()
        translator = str.maketrans("", "", string.punctuation)

        themes_normalized = [self.normalizer.tokenize_and_lemmatize(text=theme) for theme in themes]

        # Get embeddings
        sentence_embeddings = self.sbert_model.encode(themes_normalized, convert_to_tensor=True).cpu().numpy()
        average_word_embeddings = np.array(
            [self.get_average_word_embedding_with_synsets(theme) for theme in tqdm(themes_normalized)]
        )
        average_synset_embeddings = np.array(
            [self.get_average_synset_embedding(theme) for theme in tqdm(themes_normalized)]
        )

        return sentence_embeddings, average_word_embeddings, average_synset_embeddings, themes_normalized

    @staticmethod
    def adjust_features_with_similarity(combined_features, core_combined_features):
        adjusted_features = []
        for i, theme_embedding in enumerate(combined_features):
            # Calculate cosine similarity with each core theme
            similarities = cosine_similarity([theme_embedding], core_combined_features)
            # Find the most similar core theme
            most_similar_idx = np.argmax(similarities)
            most_similar_core_theme = core_combined_features[most_similar_idx]
            # Adjust the embedding by adding the similarity-weighted core theme embedding
            adjusted_embedding = theme_embedding + similarities[0, most_similar_idx] * most_similar_core_theme
            adjusted_features.append(adjusted_embedding)
        return np.array(adjusted_features)

    @staticmethod
    def get_fasttext_embeddings(themes: List[str]):
        themes_tokenized = [theme.split() for theme in themes]
        fasttext_model = FastText(sentences=themes_tokenized, window=3, min_count=1, sg=1)

        def get_document_embedding(doc, model):
            embeddings = [model.wv[word] for word in doc if word in model.wv]
            return np.mean(embeddings, axis=0) if embeddings else np.zeros(model.vector_size)

        fasttext_embeddings = np.array([get_document_embedding(theme, fasttext_model) for theme in themes_tokenized])
        return fasttext_embeddings

    @staticmethod
    def find_optimal_number_of_clusters(
        adjusted_combined_features, threshold: float = 0.5, max_clusters: int = 15, min_clusters: int = 3
    ):
        silhouette_scores = []
        for n_clusters in tqdm(range(min_clusters, max_clusters + 1)):
            try:
                # Initialize and fit the GMM model with n_clusters
                gmm = GaussianMixture(n_components=n_clusters, random_state=0)
                gmm.fit(adjusted_combined_features)

                # Predict clusters
                clusters = gmm.predict(adjusted_combined_features)

                # Calculate the silhouette score for the current number of clusters
                silhouette_avg = silhouette_score(adjusted_combined_features, clusters)
                silhouette_scores.append((n_clusters, silhouette_avg))

                # Identify outliers based on a threshold of low probability
                probs = gmm.predict_proba(adjusted_combined_features)
                max_probs = probs.max(axis=1)
                outliers = max_probs < threshold

                # Output the clusters and outliers for this number of clusters
                print(f"\nNumber of Clusters: {n_clusters}")
                print(f"Silhouette Score: {silhouette_avg}")
            except:
                silhouette_avg = -1
                silhouette_scores.append((n_clusters, silhouette_avg))
        # After the loop, identify the cluster size with the best silhouette score
        best_cluster_size, best_silhouette_score = max(silhouette_scores, key=lambda x: x[1])
        print(f"\nBest Cluster Size: {best_cluster_size}")
        print(f"Best Silhouette Score: {best_silhouette_score}")
        # Best cluster size obtained from the previous loop
        return best_cluster_size

    @staticmethod
    def recluster_large_clusters_gmm(
        adjusted_combined_features, initial_clusters, max_elements_per_cluster, min_cluster_size=2
    ):
        final_clusters = np.copy(initial_clusters)
        unique_clusters = np.unique(final_clusters)

        for cluster_label in unique_clusters:
            cluster_indices = np.where(final_clusters == cluster_label)[0]
            cluster_size = len(cluster_indices)

            # If the cluster size is larger than the max allowed, recluster
            if cluster_size > max_elements_per_cluster:
                if cluster_size > min_cluster_size:
                    # Recluster this large cluster
                    sub_features = adjusted_combined_features[cluster_indices]
                    n_sub_clusters = max(2, cluster_size // max_elements_per_cluster)  # Ensure at least 2 clusters
                    sub_gmm = GaussianMixture(n_components=n_sub_clusters, random_state=0)
                    sub_cluster_labels = sub_gmm.fit_predict(sub_features)

                    # Adjust the final clusters
                    max_existing_cluster_label = final_clusters.max()
                    for sub_label in np.unique(sub_cluster_labels):
                        final_clusters[cluster_indices[sub_cluster_labels == sub_label]] = (
                            max_existing_cluster_label + sub_label + 1
                        )

        return final_clusters

    def process_themes_in_batches(self, theme_groups, themes_to_classify, labelled_clusters, batch_size=10):
        final_responses = []
        final_clusters = copy.deepcopy(labelled_clusters)
        # Chunk the themes_to_classify list into batches of batch_size
        for i in tqdm(range(0, len(themes_to_classify), batch_size)):
            batch = themes_to_classify[i : i + batch_size]

            # Create the user prompt with the current batch
            user_prompt = {"theme_groups": theme_groups, "themes_to_classify": batch}

            # llm call
            response = litellm.completion(
                model= LLM_MODEL,
                messages=[
                    {"role": "system", "content": self.grouping_outliers_prompt},
                    {"role": "user", "content": json.dumps(user_prompt)},
                ],
                metadata={
                    "generation_name": "base_clusterer",
                    "trace_name": "process_themes_in_batches",
                },
                response_format={"type": "json_object"}
            )
            final_response = response.choices[0].message.content
            final_response = json.loads(final_response)
            # Store the final response
            final_responses.append(final_response)

            classified_themes = final_response["classified_themes"]
            new_groups = final_response["new_groups"]
            for key, value in new_groups.items():
                final_clusters.append({"label": key, "themes": value})
            for c_tg, c_t in classified_themes.items():
                for data in final_clusters:
                    if data["label"] == c_tg:
                        data["themes"].extend(c_t)
        return final_clusters
