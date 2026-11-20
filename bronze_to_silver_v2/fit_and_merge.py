from normalizer import Normalizer


class FitAndMerge:
    def __init__(self, existing_ontology):
        self.ontology = existing_ontology
        self.theme_mapping = {}
        self.attribute_mapping = {}
        self.normalizer = Normalizer()
        self.create_mappings()

    def create_mappings(self):
        self.create_theme_mapping()
        self.create_attribute_mapping()

    def create_theme_mapping(self):
        for theme_group, themes in self.ontology.items():
            self.theme_mapping[theme_group] = [self.normalizer.tokenize_and_lemmatize(theme) for theme in themes.keys()]

    def create_attribute_mapping(self):
        for theme_group, themes in self.ontology.items():
            for theme, sub_themes in themes.items():
                theme = self.normalizer.tokenize_and_lemmatize(theme)
                self.attribute_mapping[theme] = {}
                for sub_theme, attributes in sub_themes.items():
                    self.attribute_mapping[theme][sub_theme] = [
                        self.normalizer.tokenize_and_lemmatize(attribute) for attribute in attributes
                    ]

    def fit(self, theme_name, attribute_name):
        theme_group = self.get_theme_group(theme_name)
        sub_theme = self.get_sub_theme(theme_name, attribute_name)
        return theme_group, sub_theme

    def get_theme_group(self, theme_name):
        theme_group = None
        theme_name = self.normalizer.tokenize_and_lemmatize(theme_name)
        for key, value in self.theme_mapping.items():
            if theme_name in value:
                theme_group = key
        return theme_group

    def get_sub_theme(self, theme_name, attribute_name):
        sub_theme = None
        theme_name = self.normalizer.tokenize_and_lemmatize(theme_name)
        attribute_name = self.normalizer.tokenize_and_lemmatize(attribute_name)
        sub_theme_map = self.attribute_mapping.get(theme_name, {})
        for key, value in sub_theme_map.items():
            if attribute_name in value:
                sub_theme = key
        return sub_theme
