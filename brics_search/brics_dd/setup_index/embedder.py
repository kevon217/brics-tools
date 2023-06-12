import os
from pathlib import Path
import pandas as pd
import tqdm
import importlib
import torch


class HybridEmbedder:
    def __init__(self, config):
        self.config = config
        self.df = None
        self.models = {}
        self.tokenizers = {}

        # Load models at initialization
        for model_type in self.config["embed"]["hybrid"]:
            self.load_model(self.config["embed"]["hybrid"][model_type], model_type)

    def load_csv_with_processed_headers(self, filepath_raw=None):
        if filepath_raw is None:
            filepath_raw = self.config["upsert"]["filepath_raw"]
        if os.path.isfile(filepath_raw):
            df = pd.read_csv(filepath_raw)
            # Replace spaces and forward slash in column names with underscores
            df.columns = df.columns.str.replace(" ", "_")
            df.columns = df.columns.str.replace("/", "_")
            df.columns = df.columns.str.replace(".", "_")
            # TODO: Add more cleaning steps here e.g., remove punctuation
            self.df = df
        else:
            raise ValueError(f"{filepath_raw} does not exist or is not a file.")

    def load_tokenizer(self, tokenizer_config, tokenizer_type):
        library_name = tokenizer_config["library"]
        function_name = tokenizer_config["function"]
        try:
            library = importlib.import_module(library_name)
        except ImportError as e:
            print(f"Error importing library '{library_name}': {str(e)}")
            return

        try:
            function_class = getattr(library, function_name)
        except AttributeError as e:
            print(
                f"Error getting function '{function_name}' from library '{library_name}': {str(e)}"
            )
            return

        try:
            self.tokenizers[tokenizer_type] = function_class.from_pretrained(
                tokenizer_config["model_name"]
            )
        except Exception as e:
            print(f"Error loading {tokenizer_type} tokenizer: {str(e)}")

    def load_model(self, model_config, model_type, force_reload=False):
        # Check if the model is already loaded
        if model_type in self.models and not force_reload:
            print(f"{model_type} model is already loaded.")
            return

        library_name = model_config["library"]
        function_name = model_config["function"]

        try:
            library = importlib.import_module(library_name)
        except ImportError as e:
            print(f"Error importing library '{library_name}': {str(e)}")
            return

        try:
            model_function = getattr(library, function_name)
        except AttributeError as e:
            print(
                f"Error getting function '{function_name}' from library '{library_name}': {str(e)}"
            )
            return

        try:
            # Try to load with from_pretrained
            self.models[model_type] = model_function.from_pretrained(
                model_config["model_name"]
            )
        except AttributeError:
            # Fall back to normal loading if from_pretrained does not exist
            try:
                self.models[model_type] = model_function(model_config["model_name"])
            except Exception as e:
                print(
                    f"Error loading {model_type} model using '{model_config['model_name']}': {str(e)}"
                )
                return

        if "tokenizer" in model_config:
            self.load_tokenizer(model_config["tokenizer"], model_type)

    def batch_generator(self, iterable, batch_size):
        """Helper function for splitting the data into batches."""
        length = len(iterable)
        for idx in range(0, length, batch_size):
            yield iterable[idx : min(idx + batch_size, length)]

    def embed_text(self, text, model_type):
        # Handles a single string and a list of strings
        if isinstance(text, str):
            text = [text]

        if model_type == "dense":
            model = self.models[model_type]
            embeddings = model.encode(
                text,
                show_progress_bar=True,
                normalize_embeddings=self.config["embed"]["hybrid"][model_type][
                    "normalize"
                ],
            )
        elif model_type == "sparse":
            model = self.models[model_type]
            tokenizer = self.tokenizers[model_type]
            device = "cuda" if torch.cuda.is_available() else "cpu"

            tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            input_ids = tokens.input_ids.to(device)
            attention_mask = tokens.attention_mask.to(device)

            with torch.no_grad():
                output = model(input_ids=input_ids, attention_mask=attention_mask)

            vec = torch.max(
                torch.log(1 + torch.relu(output.logits)) * attention_mask.unsqueeze(-1),
                dim=1,
            )[0]
            embeddings = []
            for sample_vec in vec:
                sample_vec = sample_vec.squeeze()
                indices = sample_vec.nonzero(as_tuple=True)[0].cpu().tolist()
                values = sample_vec[indices].cpu().tolist()
                sparse_dict = {"indices": indices, "values": values}
                embeddings.append(sparse_dict)

        return embeddings

    def generate_dense_embeddings(self, colname, model_type):
        model = self.models[model_type]
        batch_size = self.config["embed"]["hybrid"][model_type]["batch_size"]
        col_embed = f"{colname}_{model_type}"
        embeddings = []
        for batch in tqdm.tqdm(
            self.batch_generator(self.df[colname].values.tolist(), batch_size)
        ):
            embeddings.extend(self.embed_text(batch, model_type))
        self.df[col_embed] = embeddings

    def generate_sparse_embeddings(self, colname, model_type):
        model = self.models[model_type]
        tokenizer = self.tokenizers[model_type]
        batch_size = self.config["embed"]["hybrid"][model_type]["batch_size"]
        device = self.config["embed"]["hybrid"][model_type]["device"]
        col_embed = f"{colname}_{model_type}"
        embeddings = []

        for batch in tqdm.tqdm(
            self.batch_generator(self.df[colname].values.tolist(), batch_size)
        ):
            embeddings.extend(self.embed_text(batch, model_type))

        self.df[col_embed] = embeddings  # Add this line

    def convert_idx2token(self, colname, model_type):
        tokenizer = self.tokenizers[model_type]
        # extract the ID position to text token mappings
        idx2token = {idx: token for token, idx in tokenizer.get_vocab().items()}

        col_embed = f"{colname}_{model_type}"
        col_sparse_idx2token = f"{colname}_{model_type}_idx2token"

        # process the embeddings: map token IDs to human-readable tokens, round weights, sort by weight
        def process_embeddings(embedding):
            indices = embedding["indices"]
            values = embedding["values"]
            sparse_dict_tokens = {
                idx2token.get(idx, "<UNK>"): round(weight, 2)
                for idx, weight in zip(indices, values)
            }
            # sort so we can see most relevant tokens first
            sparse_dict_tokens = {
                k: v
                for k, v in sorted(
                    sparse_dict_tokens.items(), key=lambda item: item[1], reverse=True
                )
            }
            return sparse_dict_tokens

        self.df[col_sparse_idx2token] = self.df[col_embed].apply(process_embeddings)

    def tokenize_columns(self):
        tokenizer = self.tokenizers["metadata"]
        for colname in self.config["upsert"]["metadata"]["tokenize_columns"]["columns"]:
            if colname in self.df.columns:
                self.df[colname + "_tokens"] = self.df[colname].progress_apply(
                    lambda x: tokenizer.tokenize(x)
                )
            else:
                print(f"Column {colname} not found in DataFrame.")

    def select_metadata(self):
        metadata_config = self.config["upsert"]["metadata"]
        metadata_columns = metadata_config["include_columns"]

        missing_columns = set(metadata_columns) - set(self.df.columns)
        if missing_columns:
            print(f"Columns {missing_columns} not found in DataFrame.")

        # Check if we need to tokenize some columns
        if metadata_config["tokenize_columns"]["tokenize"]:
            tokenizer_columns = metadata_config["tokenize_columns"]["columns"]
            missing_tokenizer_columns = set(tokenizer_columns) - set(self.df.columns)
            if missing_tokenizer_columns:
                print(
                    f"Columns for tokenization {missing_tokenizer_columns} not found in DataFrame."
                )
            else:
                # Load the tokenizer
                self.load_tokenizer(
                    metadata_config["tokenize_columns"]["tokenizer"], "metadata"
                )
                tokenizer = self.tokenizers["metadata"]
                for colname in tokenizer_columns:
                    tokenized_colname = colname + "_tokens"
                    self.df[tokenized_colname] = self.df[colname].apply(
                        lambda x: tokenizer.tokenize(x)
                    )
                    # If tokenized columns need to be included in metadata
                    if metadata_config["tokenize_columns"]["include_tokenized"]:
                        metadata_columns.append(tokenized_colname)

        return self.df[metadata_columns]

    def process_data(self):
        for colname in self.config["embed"]["csv_columns"]:
            self.generate_dense_embeddings(colname, "dense")
            self.generate_sparse_embeddings(colname, "sparse")
            self.convert_idx2token(colname, "sparse")
        metadata_df = self.select_metadata()
        self.df = pd.concat([self.df, metadata_df], axis=1)

    def save_data(self):
        filepath_processed = self.config["upsert"]["filepath_processed"]
        self.df.to_pickle(filepath_processed)

    def run(self):
        self.load_csv_with_processed_headers()
        for model_type in self.config["embed"]["hybrid"]:
            self.load_model(self.config["embed"]["hybrid"][model_type], model_type)
        # self.load_tokenizer(self.config["upsert"]["metadata"]["tokenizer"], "metadata")
        self.process_data()
        self.save_data()
        return self.df


# # Tokenization step
#             tokens = tokenizer(
#                 batch, return_tensors="pt", padding=True, truncation=True
#             )
#             input_ids = tokens.input_ids.to(device)
#             attention_mask = tokens.attention_mask.to(device)

#             # Encoding step
#             with torch.no_grad():
#                 output = model(input_ids=input_ids, attention_mask=attention_mask)

#             # SPLADE vector extraction
#             vec = torch.max(
#                 torch.log(1 + torch.relu(output.logits)) * attention_mask.unsqueeze(-1),
#                 dim=1,
#             )[0]

#             # Handle each sample in the batch separately
#             # Handle each sample in the batch separately
#             batch_embeddings = []
#             for sample_vec in vec:
#                 sample_vec = sample_vec.squeeze()
#                 # # Extract non-zero positions
#                 # cols = (
#                 #     sample_vec.nonzero(as_tuple=True)[0].cpu().tolist()
#                 # )  # modified line

#                 # # Extract non-zero values
#                 # weights = sample_vec[cols].cpu().tolist()
#                 indices = sample_vec.nonzero(as_tuple=True)[0].cpu().tolist()
#                 values = sample_vec[indices].cpu().tolist()
#                 sparse_dict = {"indices": indices, "values": values}

#                 # map token IDs to human-readable tokens

#                 batch_embeddings.append(sparse_dict)
