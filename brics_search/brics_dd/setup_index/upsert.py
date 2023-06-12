import math


class PineconeUpsert:
    def __init__(self, config, index, df):
        self.config = config
        self.index = index
        self.df = df
        self.batch_size = 100  # This could also come from your config file

        # Include tokenized columns if specified
        if (
            self.config["upsert"]["metadata"]["tokenize_columns"]["tokenize"]
            and self.config["upsert"]["metadata"]["tokenize_columns"][
                "include_tokenized"
            ]
        ):
            for col in self.config["upsert"]["metadata"]["tokenize_columns"]["columns"]:
                self.config["upsert"]["metadata"]["include_columns"].append(
                    f"{col}_tokens"
                )

    def _generate_batches(self):
        num_batches = math.ceil(len(self.df) / self.batch_size)
        for i in range(num_batches):
            yield self.df[i * self.batch_size : (i + 1) * self.batch_size]

    def upsert(self):
        for col in self.config["common"]["csv_columns"]:
            for batch_df in self._generate_batches():
                vectors = []
                for item in batch_df.itertuples():
                    metadata = {
                        field: str(getattr(item, field))
                        if str(getattr(item, field)).lower() != "nan"
                        else ""
                        for field in self.config["upsert"]["metadata"][
                            "include_columns"
                        ]
                    }
                    if self.config["upsert"]["metadata"]["tokenize_columns"][
                        "tokenize"
                    ]:
                        metadata[f"{col}_tokens"] = getattr(item, f"{col}_tokens")
                    vectors.append(
                        {
                            "id": getattr(item, "variable_name"),
                            "values": getattr(item, f"{col}_dense"),
                            "sparse_values": getattr(item, f"{col}_sparse"),
                            "metadata": metadata,
                        }
                    )
                try:
                    self.index.upsert(vectors=vectors, namespace=col)
                except Exception as e:
                    print(f"Upsert operation failed with error: {e}")
                    # Depending on your requirements, you might want to handle this differently, e.g., by re-trying, logging the error, or raising it to the caller.


# df_batcher = BatchGenerator(100)
#     for col in self.config['common'][csv_columns]:
#         for batch_df in tqdm(df_batcher(df)):
#             vectors = []
#             for i in range(len(batch_df)):
#                 vectors.append(
#                     {
#                         'id': batch_df.vector_id.iloc[i],
#                         'values': batch_df[f'{col}_dense'].iloc[i],
#                         'sparse_values': batch_df[f'{col}_sparse'].iloc[i],
#                         'metadata': {
#                             f'col_tokens': batch_df[f'{col}_tokens'].iloc[i],
#                             'variable name' : batch_df.variable_name.iloc[i],
#                             'title': batch_df.title.iloc[i],
#                             'definition':   batch_df.definition.iloc[i],
#                             'data element concept identifiers': batch_df.data_element_concept_identifiers.iloc[i],
#                             'data element concept names': batch_df.data_element_concept_names.iloc[i],
#                             'permissible values': batch_df.permissible_values.iloc[i],
#                             'permissible value descriptions':   batch_df.permissible_value_descriptions.iloc[i],
#                             'preferred question text': batch_df.preferred_question_text.iloc[i],
#                             'datatype': batch_df.datatype.iloc[i],
#                             'input restriction':    batch_df.input_restriction.iloc[i],
#                             'guidelines/instructions':      batch_df.guidelines_instructions.iloc[i],
#                             'notes':    batch_df.notes.iloc[i],
#                             'population.all':   batch_df.population_all.iloc[i],
#                             'creation date':    batch_df.creation_date.iloc[i],
#                             'submitting contact name':  batch_df.submitting_contact_name.iloc[i]
#                         },
#                     }
#                 )
#             index.upsert(vectors=vectors, namespace=col)
