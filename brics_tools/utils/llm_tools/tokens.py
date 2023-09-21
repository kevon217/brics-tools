import tiktoken
from transformers import AutoTokenizer


def tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)


def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def openai_api_calculate_cost(usage, model="gpt-3.5-turbo-16k"):
    pricing = {
        "gpt-3.5-turbo-4k": {
            "prompt": 0.0015,
            "completion": 0.002,
        },
        "gpt-3.5-turbo-16k": {
            "prompt": 0.003,
            "completion": 0.004,
        },
        "gpt-4-8k": {
            "prompt": 0.03,
            "completion": 0.06,
        },
        "gpt-4-32k": {
            "prompt": 0.06,
            "completion": 0.12,
        },
        "text-embedding-ada-002-v2": {
            "prompt": 0.0001,
            "completion": 0.0001,
        },
    }

    try:
        model_pricing = pricing[model]
    except KeyError:
        raise ValueError("Invalid model specified")

    prompt_cost = usage["prompt_tokens"] * model_pricing["prompt"] / 1000
    completion_cost = usage["completion_tokens"] * model_pricing["completion"] / 1000

    total_cost = prompt_cost + completion_cost
    print(
        f"\nTokens used:  {usage['prompt_tokens']:,} prompt + {usage['completion_tokens']:,} completion = {usage['total_tokens']:,} tokens"
    )
    print(f"Total cost for {model}: ${total_cost:.4f}\n")

    return total_cost


all_text = ""
for doc in studyinfo_docs:
    all_text += doc.text
giant_document = Document(text=all_text)
usage = {}
# usage['prompt_tokens'] = num_tokens_from_string(giant_document.text, "gpt-4")
usage["prompt_tokens"] = num_tokens_from_string(giant_document.text, "gpt-3.5-turbo")
usage["completion_tokens"] = 2000
usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
total_cost = openai_api_calculate_cost(usage, model="gpt-3.5-turbo-16k")
total_cost = openai_api_calculate_cost(usage, model="gpt-4-8k")
