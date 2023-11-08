import pandas as pd
from pandas.api.types import CategoricalDtype
from tqdm import tqdm
import re
from bs4 import BeautifulSoup
import html as ihtml


def clean_text(text):
    text = BeautifulSoup(ihtml.unescape(text), "lxml").text
    text = re.sub(r"http[s]?://\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def preprocess_umls_data(umls_data):
    # Retrieve DataFrames from the dictionary
    df_concepts = umls_data["concepts"]
    df_definitions = umls_data["definitions"]
    df_termtype_rank = umls_data["termtype_rank"]
    df_semantic_types = umls_data["semantic_types"]

    # Step 1: Filter ENG LAT
    df_concepts = df_concepts[df_concepts["LAT"].isin(["ENG"])]

    # Step 2: Remove rows with null values
    df_concepts = df_concepts[df_concepts["STR"].notnull()]
    df_definitions = df_definitions[df_definitions["DEF"].notnull()]

    # Step 3: Text cleaning
    # Remove specific patterns and clean text
    remove_patterns = r"(\*\*Definition:\*\*|Definition:|Description:|WHAT:|\*\*\*\*)"
    df_definitions["DEF"] = df_definitions["DEF"].str.replace(
        remove_patterns, "", regex=True
    )
    df_concepts["STR"] = df_concepts["STR"].str.replace(remove_patterns, "", regex=True)

    tqdm.pandas(desc="Cleaning DEF column:")
    df_definitions["DEF"] = df_definitions["DEF"].progress_apply(
        lambda x: clean_text(x) if re.search(r"<[^<]+?>", x) else x
    )
    tqdm.pandas(desc="Cleaning STR column:")
    df_concepts["STR"] = df_concepts["STR"].progress_apply(
        lambda x: clean_text(x) if re.search(r"<[^<]+?>", x) else x
    )

    # Step 4: Rank CUIs by precedence based on rank in TTY file
    termtypes = df_termtype_rank["TTY"].dropna().unique()
    rank_sorter = CategoricalDtype(termtypes, ordered=True)
    df_concepts = df_concepts[df_concepts["TTY"].notnull()]
    df_concepts["TTY"] = df_concepts["TTY"].astype(rank_sorter)
    df_concepts.sort_values("TTY", inplace=True)

    # Step 5: Merge semantic types for each CUI
    df_semantic_types_flat = (
        df_semantic_types.groupby(["CUI"])["STY"].apply(";".join).reset_index()
    )

    # Step 6: Merge the sources for each CUI
    conso_sab_concat = (
        df_concepts.groupby("CUI")["SAB"]
        .apply(lambda x: ";".join(set(x)))
        .reset_index()
    )
    def_sab_concat = (
        df_definitions.groupby("CUI")["SAB"]
        .apply(lambda x: ";".join(set(x)))
        .reset_index()
    )

    # Step 7: Drop duplicates
    df_concepts.drop_duplicates(subset="CUI", inplace=True, keep="first")
    df_definitions.drop_duplicates(subset="CUI", inplace=True, keep="first")

    # Step 8: Merge DataFrames
    df_concepts = pd.merge(df_concepts, conso_sab_concat, on="CUI", how="left")
    df_definitions = pd.merge(df_definitions, def_sab_concat, on="CUI", how="left")
    df_concepts_sty = pd.merge(
        df_concepts, df_semantic_types_flat, on="CUI", how="left"
    )
    df_umls = pd.merge(df_definitions, df_concepts_sty, on="CUI", how="left")

    return df_umls


# Usage
# umls_data = {
#     'concepts': df_CONSO,
#     'definitions': df_DEF,
#     'termtype_rank': df_RANK,
#     'semantic_types': df_STY
# }
# df_umls = preprocess_umls_data(umls_data)
