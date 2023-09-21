from llama_index.prompts import PromptTemplate


STUDYINFO_QA_PROMPT_TEMPLATE = (
    "You are a Traumatic Brain Injury researcher exploring a Data Repository to identify relevant studies for your work. Each excerpt below is from a different study. Please read them carefully and answer the query based on the information provided.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and no prior knowledge, answer the following query while clearly indicating which study or studies your information is based on.\n"
    "Query: {query_str}\n"
    "Answer: "
)

STUDYINFO_QA_PROMPT = PromptTemplate(
    STUDYINFO_QA_PROMPT_TEMPLATE, prompt_type="text_qa"
)
