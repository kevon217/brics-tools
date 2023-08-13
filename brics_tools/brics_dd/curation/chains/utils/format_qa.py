from typing import Dict, Any
from pydantic import BaseModel, Field, validator
from sentence_transformers.cross_encoder import CrossEncoder



class Answer(BaseModel):
    Reasoning: str = Field(description="The reason for selecting this answer")
    Confidence: int = Field(description="Confidence score between 1-10 for this answer")

    @validator("Confidence")
    def check_score(cls, field):
        if field > 10:
            raise ValueError("Confidence score should be between 1-10")
        return field


class ConceptQA(BaseModel):
    Answers: Dict[str, Answer] = Field(
        description="A dictionary of "
        "applicable answers, "
        "each with its own reasoning and confidence score"
    )


def filter_df_for_qa(df, variable_name, overall_rank, top_k_score):
    # Create a new column to store the original index
    df["original_index"] = df.index

    df_filtered = (
        df[df["variable name"] == variable_name]
        .groupby("pipeline_name_alpha")
        .apply(lambda x: x[x["overall_rank"] <= overall_rank])
        .drop_duplicates(subset=["data element concept identifiers"])
        .nlargest(top_k_score, "average_score")
    )

    # Reset the index
    df_filtered.reset_index(drop=True, inplace=True)

    return df_filtered

cfg.curation.preprocessing.cross_encoder_rerank = True
cfg.curation.preprocessing.cross_encoder.model_name = \
    "jamescalam/bert-stsb-cross-encoder"
cfg.curation.preprocessing.cross_encoder.top_k = 5

model_name = "jamescalam/bert-stsb-cross-encoder"
model_name2 = "cross-encoder/stsb-distilroberta-base"
model_name3 = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
top_k = 5

def cross_encoder_rerank(df, model_name, variable_name, top_k):
    model = CrossEncoder(model_name)
    model2 = CrossEncoder(model_name2)
    model3 = CrossEncoder(model_name3)
    variable_title = df["title"].iloc[0]
    variable_definition = df["definition"].iloc[0]
    title_combinations = [[variable_title, title] for title in df["data element concept names"]]
    definition_combinations = [[variable_definition, definition] for
                               definition in df["data element concept definitions"]]
    title_name_similarity = model.predict(title_combinations)
    title_name_similarity2 = model2.predict(title_combinations)
    title_name_similarity3 = model3.predict(title_combinations)

    df["title_name_similarity"] = title_name_similarity
    df["title_name_similarity2"] = title_name_similarity2
    df["title_name_similarity3"] = title_name_similarity3
    definition_name_similarity = model.predict(definition_combinations)

def create_qa_prompt_from_df(df_filtered):
    # Extract the variable's title and definition
    variable_title = df_filtered["title"].iloc[0]
    variable_definition = df_filtered["definition"].iloc[0]

    # Initialize the question
    question = f"""Question: Given the variable with the title "{variable_title}" and the definition "{variable_definition}", which of the following options are the best candidates for mapping to this variable's title and definition?"""

    # Initialize an empty list to store the options
    options = []

    # Loop over each row in the filtered dataframe
    for i, row in df_filtered.iterrows():
        # Extract the result's title and definition
        result_title = row["data element concept names"]
        result_definition = row["data element concept definitions"]

        # Format the option and add it to the list
        options.append(
            f"{i+1}. Title: {result_title}; Definition:" f" {result_definition}"
        )

    # Add "None of the Above" as the last option
    options.append(f"{len(df_filtered)+1}. None of the Above")

    # Combine the question and options into a single string
    prompt = question + "\n\nOptions:\n" + "\n".join(options)

    # Return the list of prompts
    return prompt
