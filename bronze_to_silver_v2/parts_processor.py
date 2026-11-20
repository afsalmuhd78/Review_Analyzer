import re
from collections import Counter

import dspy
from dsp.trackers.langfuse_tracker import LangfuseTracker
import nltk
import pandas as pd
import wordninja
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from os import environ
# from config import settings
lm = dspy.LM("openai/gpt-4o-mini", api_key=environ["OPENAI_API_KEY"])
dspy.configure(lm=lm)

langfuse = LangfuseTracker()

class ExtractPartName(dspy.Signature):
    parts: list[str] = dspy.InputField(desc="List of part names")
    cluster_name: str = dspy.OutputField(
        desc="Suggest a common overall part name, which is a physical part. Use American English for part names"
    )


class PartNameProcessor:
    def __init__(self, distance_threshold=0.5, min_bigram_freq=2):
        """
        Initializes the PartNameProcessor with necessary components and downloads required NLTK data.

        Parameters:
        - distance_threshold (float): The linkage distance threshold for clustering.
        - min_bigram_freq (int): Minimum frequency required for a bigram to be considered as a signature.
        """
        self.distance_threshold = distance_threshold
        self.min_bigram_freq = min_bigram_freq
        self.lemmatizer = WordNetLemmatizer()
        self.module = dspy.Predict(ExtractPartName)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.part_name_mappings = {}
        self.finalized_part_names = {}
        self.df = pd.DataFrame()
        self.grouped_names = pd.DataFrame()
        tqdm.pandas()

    def load_data(self, df_input, part_column="part"):
        """
        Loads the input DataFrame containing part names.

        Parameters:
        - df_input (pd.DataFrame): Input DataFrame.
        - part_column (str): The column name in df_input that contains part names.
        """
        if part_column not in df_input.columns:
            raise ValueError(f"Input DataFrame must contain '{part_column}' column.")
        self.part_names = df_input[part_column].dropna().tolist()

    def convert_number_words(self, text):
        number_mapping = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
            "eleven": "11",
            "twelve": "12",
            "thirteen": "13",
            "fourteen": "14",
            "fifteen": "15",
            "sixteen": "16",
            "seventeen": "17",
            "eighteen": "18",
            "nineteen": "19",
            "twenty": "20",
            "thirty": "30",
            "forty": "40",
            "fifty": "50",
            "sixty": "60",
            "seventy": "70",
            "eighty": "80",
            "ninety": "90",
            "hundred": "100",
            "thousand": "1000",
            "million": "1000000",
            "billion": "1000000000",
            "half": "0.5",
        }

        text_lower = text.lower().replace("-", " ")
        tokens = text_lower.split()
        converted_tokens = []
        i = 0
        while i < len(tokens):
            matched = False
            for j in range(min(4, len(tokens) - i), 0, -1):
                phrase = " ".join(tokens[i : i + j])
                phrase_no_space = phrase.replace(" ", "")
                if phrase in number_mapping:
                    converted_tokens.append(number_mapping[phrase])
                    i += j
                    matched = True
                    break
                elif phrase_no_space in number_mapping:
                    converted_tokens.append(number_mapping[phrase_no_space])
                    i += j
                    matched = True
                    break
            if not matched:
                converted_tokens.append(tokens[i])
                i += 1
        return " ".join(converted_tokens)

    def replace_abbreviations(self, text):
        abbreviations_mapping = {
            "pt": "point",
            "qty": "quantity",
            "no": "number",
            "wd": "width",
            "ht": "height",
            "len": "length",
            "cap": "cap",
            "bd": "breadth",
            "velcro": "velcro",
            # Add more abbreviations as needed
        }

        # Replace standalone abbreviations
        if abbreviations_mapping:
            pattern_standalone = re.compile(
                r"\b(" + "|".join(re.escape(abbrev) for abbrev in abbreviations_mapping.keys()) + r")\b", re.IGNORECASE
            )
            text = pattern_standalone.sub(
                lambda match: abbreviations_mapping.get(match.group(1).lower(), match.group(1)), text
            )

        # Replace abbreviations attached directly to numbers (e.g., '5Pt' -> '5 Point')
        if abbreviations_mapping:
            pattern_attached = re.compile(
                r"(\d+)(%s)\b" % "|".join(re.escape(abbrev) for abbrev in abbreviations_mapping.keys()), re.IGNORECASE
            )
            text = pattern_attached.sub(
                lambda match: f"{match.group(1)} {abbreviations_mapping.get(match.group(2).lower(), match.group(2))}",
                text,
            )

        return text

    def standardize_units(self, text):
        units_mapping = {
            "mm": "mm",
            "millimeter": "mm",
            "millimeters": "mm",
            "cm": "cm",
            "centimeter": "cm",
            "centimeters": "cm",
            "m": "m",
            "meter": "m",
            "meters": "m",
            "km": "km",
            "kilometer": "km",
            "kilometers": "km",
            "in": "in",
            "inch": "in",
            "inches": "in",
            "ft": "ft",
            "foot": "ft",
            "feet": "ft",
            "yd": "yd",
            "yard": "yd",
            "yards": "yd",
            "g": "g",
            "gram": "g",
            "grams": "g",
            "kg": "kg",
            "kilogram": "kg",
            "kilograms": "kg",
            "mg": "mg",
            "milligram": "mg",
            "milligrams": "mg",
            "l": "l",
            "liter": "l",
            "liters": "l",
            "ml": "ml",
            "milliliter": "ml",
            "milliliters": "ml",
            "oz": "oz",
            "ounce": "oz",
            "ounces": "oz",
            "lb": "lb",
            "pound": "lb",
            "pounds": "lb",
            "gb": "gb",
            "gigabyte": "gb",
            "gigabytes": "gb",
            "tb": "tb",
            "terabyte": "tb",
            "terabytes": "tb",
        }

        units_pattern = "|".join(re.escape(unit) for unit in units_mapping.keys())
        pattern = re.compile(r"\b(\d+(?:\.\d+)?)\s*(" + units_pattern + r")\b", re.IGNORECASE)

        def replace_units(match):
            number = match.group(1)
            unit = match.group(2).lower()
            standard_unit = units_mapping.get(unit, unit)
            return f"{number}{standard_unit}"

        return pattern.sub(replace_units, text)

    def split_camel_case(self, text):
        return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)

    def normalize_part_name_original(self, name):
        """
        Normalize part name without applying WordNinja.
        """
        name = self.convert_number_words(name)
        name = self.replace_abbreviations(name)
        name = self.standardize_units(name)
        name = re.sub(r"[^a-zA-Z0-9 ]", "", name)
        name = re.sub(r"\s+", " ", name).strip()
        name = self.split_camel_case(name)
        name = name.lower()
        name = re.sub(r"\s+", " ", name).strip()
        if not name:
            return ""
        name = name.title()
        return name

    def normalize_part_name_wordninja(self, name):
        """
        Normalize part name and apply WordNinja for splitting concatenated words.
        """
        name = self.convert_number_words(name)
        name = self.replace_abbreviations(name)
        name = self.standardize_units(name)
        name = re.sub(r"[^a-zA-Z0-9 ]", "", name)
        name = re.sub(r"\s+", " ", name).strip()
        name = self.split_camel_case(name)
        words = name.split()
        split_words = []
        for word in words:
            split_result = wordninja.split(word)
            if split_result:
                split_words.extend(split_result)
            else:
                split_words.append(word)
        name = " ".join(split_words)
        name = name.lower()
        name = re.sub(r"\s+", " ", name).strip()
        if not name:
            return ""
        name = name.title()
        return name

    def get_signature(self, cluster_names):
        """
        Returns the most common bigram in the list of cluster names.
        If no bigram meets the minimum frequency, returns the most common word.

        Parameters:
        - cluster_names (list): List of normalized original names in the cluster.

        Returns:
        - signature (str): The assigned signature for the cluster.
        """
        # Initialize a Counter for bigrams
        bigram_counter = Counter()

        for name in cluster_names:
            words = nltk.word_tokenize(name)
            bigrams = list(ngrams(words, 2))
            bigram_counter.update(bigrams)

        # Filter bigrams by minimum frequency
        filtered_bigrams = {bigram: count for bigram, count in bigram_counter.items() if count >= self.min_bigram_freq}

        if filtered_bigrams:
            # Find the most common bigram
            most_common_bigram, freq = Counter(filtered_bigrams).most_common(1)[0]
            signature = " ".join(most_common_bigram)
        else:
            # Fallback to most common word
            word_counter = Counter()
            for name in cluster_names:
                words = nltk.word_tokenize(name)
                word_counter.update(words)
            if word_counter:
                # Handle ties by selecting the first alphabetically
                max_count = word_counter.most_common(1)[0][1]
                common_words = [word for word, count in word_counter.items() if count == max_count]
                signature = sorted(common_words)[0]
            else:
                signature = ""

        return signature

    def get_cluster_signature(self, original_names):
        """
        Wrapper function to assign cluster signatures.
        """
        return self.get_signature(original_names)

    # Step 3: Define the processing function with lemmatization
    def get_cluster(self, original_name):
        try:
            # Process the original name through the module to get the response
            response = self.module(parts=original_name)

            # Extract the cluster_name from the response
            cluster_name = response.cluster_name

            # Ensure the cluster_name is a string and convert it to lowercase for consistency
            if isinstance(cluster_name, str):
                cluster_name = cluster_name.lower()
            else:
                cluster_name = str(cluster_name).lower()
            

                # Lemmatize the cluster_name
            # If cluster_name is a single word
            lemmatized_cluster = self.lemmatizer.lemmatize(cluster_name, pos="n")

            # If cluster_name is a phrase (multiple words), lemmatize each word individually
            lemmatized_cluster = " ".join([self.lemmatizer.lemmatize(word, pos="n") for word in cluster_name.split()])
            return lemmatized_cluster
        except Exception as e:
            # Handle exceptions gracefully
            print(f"Error processing '{original_name}': {e}")
            return "unknown_cluster"

    def process(self):
        """
        Executes the entire processing pipeline: normalization, vectorization, clustering, and signature assignment.
        """
        # Initialize lists to store normalized names
        normalized_names_original = []
        normalized_names_wordninja = []
        original_names = []

        # Normalize part names
        for name in tqdm(self.part_names, desc="Normalizing Part Names"):
            normalized_original = self.normalize_part_name_original(name)
            normalized_wordninja = self.normalize_part_name_wordninja(name)
            if normalized_original.strip():
                normalized_names_original.append(normalized_original)
                normalized_names_wordninja.append(normalized_wordninja)
                original_names.append(name)
            else:
                print(f"Excluded empty or non-informative name: '{name}'")

        # Create a DataFrame
        self.df = pd.DataFrame(
            {
                "Original_Name": original_names,
                "Normalized_Name_Original": normalized_names_original,
                "Normalized_Name_WordNinja": normalized_names_wordninja,
            }
        )

        # Fit and transform the WordNinja-normalized names
        tfidf_matrix = self.vectorizer.fit_transform(self.df["Normalized_Name_WordNinja"])

        # Identify zero vectors
        tfidf_sum = tfidf_matrix.sum(axis=1).A1  # Convert to 1D array
        mask = tfidf_sum != 0
        num_zero_vectors = (~mask).sum()
        if num_zero_vectors > 0:
            print(f"Found {num_zero_vectors} zero vectors. Removing them.")
            tfidf_matrix = tfidf_matrix[mask]
            self.df = self.df[mask].reset_index(drop=True)
        else:
            print("No zero vectors found.")

        # Perform Agglomerative Clustering
        if tfidf_matrix.shape[0] > 1:
            try:
                clustering_model = AgglomerativeClustering(
                    n_clusters=None, distance_threshold=self.distance_threshold, metric="cosine", linkage="average"
                )
                clustering_model.fit(tfidf_matrix.toarray())
                cluster_labels = clustering_model.labels_
            except ValueError as ve:
                print(f"Clustering failed: {ve}")
                # Adjust parameters if clustering fails
                self.distance_threshold += 0.1
                print(f"Retrying with increased distance_threshold: {self.distance_threshold}")
                clustering_model = AgglomerativeClustering(
                    n_clusters=None, distance_threshold=self.distance_threshold, metric="cosine", linkage="average"
                )
                clustering_model.fit(tfidf_matrix.toarray())
                cluster_labels = clustering_model.labels_

            # Assign cluster labels to the DataFrame
            self.df["Cluster"] = cluster_labels

            # Group by 'Cluster' and aggregate the original normalized names
            grouped = self.df.groupby("Cluster")["Normalized_Name_Original"].apply(list).reset_index()

            # Assign signatures
            grouped["Signature"] = grouped["Normalized_Name_Original"].apply(self.get_cluster_signature)

            # Select relevant columns
            signature_df = grouped[["Cluster", "Signature"]]

            # Merge signatures into the main DataFrame
            self.df = self.df.merge(signature_df, on="Cluster", how="left")

            # Group original names by cluster
            self.grouped_names = (
                self.df.groupby("Cluster").agg({"Original_Name": list, "Signature": "first"}).reset_index()
            )

            # Assign grouped part names using the get_cluster method with a progress bar
            self.grouped_names["grouped_part_name"] = self.grouped_names["Original_Name"].progress_apply(
                self.get_cluster
            )

            # Build the part_name_mappings dictionary using groupby
            self.part_name_mappings = (
                self.grouped_names.groupby("grouped_part_name")["Original_Name"].apply(list).to_dict()
            )
            self.finalized_part_names = {}
            for part_name, parts in self.part_name_mappings.items():
                part_name_clean = part_name.replace('"', "").title()
                self.finalized_part_names[part_name_clean] = parts[
                    0
                ]  # Assuming you want the first part as the representative
        else:
            self.df["Cluster"] = 0
            print("No data available for clustering after removing zero vectors.")

    def get_mappings(self):
        """
        Returns the final part name mappings.

        Returns:
        - finalized_part_names (dict): Dictionary mapping grouped part names to representative original names.
        """
        return self.finalized_part_names

    def get_dataframe(self):
        """
        Returns the processed DataFrame.

        Returns:
        - df (pd.DataFrame): The DataFrame containing original and normalized names with cluster and signature.
        """
        return self.df.copy()

    def get_grouped_names(self):
        """
        Returns the grouped names DataFrame.

        Returns:
        - grouped_names (pd.DataFrame): DataFrame with clusters, signatures, and grouped part names.
        """
        return self.grouped_names.copy()
