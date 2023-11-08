# Implementing the UMLSConceptsPreprocessor class with methods
import pandas as pd
from typing import List, Dict


class UMLSConceptsPreprocessor:
    def __init__(self, df: pd.DataFrame, config: Dict):
        self.df = df
        self.config = config

    def filter_language(self, lang_col: str, lang_vals: List[str]) -> pd.DataFrame:
        """
        Filters rows based on the language column.
        """
        return self.df[self.df[lang_col].isin(lang_vals)]

    def remove_null_values(self, cols: List[str]) -> pd.DataFrame:
        """
        Removes rows where certain columns have null values.
        """
        return self.df.dropna(subset=cols)

    def clean_text(self, text: str) -> str:
        """
        Placeholder for the text cleaning logic.
        """
        return text

    def apply_text_cleaning(self, cols: List[str]) -> pd.DataFrame:
        """
        Applies text cleaning on specific columns.
        """
        for col in cols:
            self.df[col] = self.df[col].apply(self.clean_text)
        return self.df

    def merge_additional_info(
        self, additional_df: pd.DataFrame, merge_cols: List[str]
    ) -> pd.DataFrame:
        """
        Merges additional information from other DataFrames.
        """
        return pd.merge(self.df, additional_df, on=merge_cols, how="left")

    def rank_by_precedence(self, rank_df: pd.DataFrame, rank_col: str) -> pd.DataFrame:
        """
        Ranks the rows based on precedence defined in a rank DataFrame.
        """
        # Implement ranking logic here
        return self.df

    def drop_duplicates(self, subset_cols: List[str]) -> pd.DataFrame:
        """
        Drops duplicate rows based on a subset of columns.
        """
        return self.df.drop_duplicates(subset=subset_cols)

    def preprocess(self):
        """
        Main method to apply all the preprocessing steps.
        """
        # Step 1: Filter by language
        self.df = self.filter_language("LAT", self.config["filters"]["LAT"])

        # Step 2: Remove null values
        self.df = self.remove_null_values(["STR"])

        # Step 3: Apply text cleaning
        self.df = self.apply_text_cleaning(["STR"])

        # Additional steps like ranking, merging, and dropping duplicates can be added here

        return self.df


# Sample usage (replace this with actual DataFrame and config)
sample_df = pd.DataFrame(
    {"LAT": ["ENG", "FRE", "ENG"], "STR": ["Text1", None, "Text3"]}
)
sample_config = {"filters": {"LAT": ["ENG"]}}

preprocessor = UMLSConceptsPreprocessor(sample_df, sample_config)
preprocessed_df = preprocessor.preprocess()
preprocessed_df
