import json
import pandas as pd

import panel as pn
from panel.widgets import IntSlider, FloatSlider, TextInput, Button, Select, Tabulator

from brics_search.brics_dd.setup_index.embedder import HybridEmbedder


class HybridSearchApp:
    def __init__(self, config, index):
        # Initialization code goes here
        self.config = config
        # Create Pinecone client
        self.index = index

        # Create HybridEmbedder
        self.embedder = HybridEmbedder(self.config)

        # Create widgets
        self.query_input = TextInput(name="Search Query")
        self.search_button = Button(name="Search", button_type="primary")
        self.search_button.on_click(self.search)
        self.alpha_slider = FloatSlider(
            name="Alpha", start=0, end=1, value=0.5, step=0.1
        )
        self.topk_slider = IntSlider(name="Top K", start=1, end=20, value=10)
        self.namespace_select = Select(
            name="Namespace", options=["title", "definition"]
        )
        # Add text input for each metadata filter field
        self.metadata_fields = [
            "variable_name",
            "title",
            "definition",
            "tokens",
            "data_element_concept_identifiers",
            "data_element_concept_names",
            "permissible_values",
            "permissible_value_descriptions",
            "preferred_question_text",
            "datatype",
            "input_restriction",
            "guidelines_instructions",
            "notes",
            "population_all",
            "creation_date",
            "submitting_contact_name",
        ]
        self.metadata_filters = {
            field: TextInput(name=field, value="") for field in self.metadata_fields
        }
        # Complex filter input
        self.complex_filter_widget = TextInput(name="Complex Filter (JSON)", value="")
        self.results_placeholder = Tabulator(
            pd.DataFrame(),
            theme="semantic-ui",
            layout="fit_columns",
            show_index=False,
            hidden_columns=[
                "tokens",
                "title_tokens",
                "definition_tokens",
                "variable_name",
                "data_element_concept_identifiers",
                "data_element_concept_names",
                "permissible_value_descriptions",
                "preferred_question_text",
                "guidelines_instructions",
                "notes",
                "population_all",
                "creation_date",
                "submitting_contact_name",
            ],
        )

    def make_layout(self):
        # Create an accordion for metadata filters
        metadata_filters_column = pn.Column(*self.metadata_filters.values())
        accordion = pn.Accordion(
            ("Metadata Filters", metadata_filters_column),
            ("Complex Filter", self.complex_filter_widget),
        )
        accordion.active = []

        # Layout
        layout = pn.Row(
            # Left side
            pn.Column(
                pn.pane.Markdown("## Hybrid Search: Data Dictionary"),  # Header
                self.namespace_select,  # Namespace selection
                self.query_input,  # Primary Input
                pn.Row(
                    pn.Column(
                        self.alpha_slider,  # Alpha Slider
                        self.topk_slider,  # Top K Slider
                    ),
                ),
                accordion,  # Advanced Filters
                self.search_button,  # Search Button
            ),
            # Right side
            pn.Column(
                pn.Spacer(height=50),
                self.results_placeholder,  # Results
            ),
        )

        return layout

    def hybrid_search_scale(self, dense, sparse, alpha: float):
        """Scale dense and sparse vectors for hybrid search"""
        # check alpha value is in range
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")
        # scale sparse and dense vectors to create hybrid search vecs
        hsparse = {
            "indices": sparse["indices"],
            "values": [v * (1 - alpha) for v in sparse["values"]],
        }
        hdense = [v * alpha for v in dense]
        return hdense, hsparse

    def format_results_into_df(self, results):
        # Initialize an empty list to hold the rows of data
        rows = []
        # Loop over each match in the results
        for match in results["matches"]:
            # Extract the relevant fields
            id_ = match["id"]
            score = match["score"]
            metadata = match["metadata"]
            # Create a row of data from the fields
            row = {
                "id": id_,
                "score": score,
                **metadata,
            }
            # Append the row to the list of rows
            rows.append(row)

        # Convert the list of rows into a DataFrame
        df = pd.DataFrame(rows)

        # Insert rank column at the beginning
        df.insert(0, "rank", range(1, len(df) + 1))

        # Insert namespace as the second column
        df.insert(1, "namespace", results["namespace"])
        # df = df[col_list] #TODO define in config dataframe column order

        return df

    def search(self, _=None):
        try:
            # Get user's input
            query = self.query_input.value
            alpha = self.alpha_slider.value
            top_k = self.topk_slider.value
            namespace = self.namespace_select.value

            # Validate input
            if not query:
                print("Please enter a search query.")
                return

            # Get metadata filters
            filters = {}
            for field, widget in self.metadata_filters.items():
                filter_value = widget.value.strip()
                if filter_value:
                    try:
                        # Check if filter_value can be parsed as JSON
                        filters[field] = json.loads(filter_value)
                    except json.JSONDecodeError:
                        # If not, treat it as a simple string or int
                        try:
                            # Check if the string can be converted to an integer
                            filters[field] = int(filter_value)
                        except ValueError:
                            # If not, treat it as a simple string
                            filters[field] = filter_value

                        # Get complex filter
            complex_filter_value = self.complex_filter_widget.value.strip()
            if complex_filter_value:
                try:
                    complex_filter = json.loads(complex_filter_value)
                    # Merge complex_filter into filters
                    # This will overwrite any conflicting keys in filters
                    filters.update(complex_filter)
                except json.JSONDecodeError as e:
                    print(
                        f"Invalid complex filter syntax. Please ensure it's a valid JSON. Error: {str(e)}"
                    )
                    return

            # Process the query into dense and sparse embeddings
            dense_embedding = self.embedder.embed_text(query, "dense")
            sparse_embedding = self.embedder.embed_text(query, "sparse")

            # Scale the embeddings
            dense_embedding, sparse_embedding = self.hybrid_search_scale(
                dense_embedding[0], sparse_embedding[0], alpha
            )

            # Perform the hybrid semantic search on the selected namespace
            results = self.index.query(
                namespace=namespace,
                top_k=top_k,
                include_values=True,
                include_metadata=True,
                vector=dense_embedding,
                sparseVector=sparse_embedding,
                filter=filters,  # Add filters to the query
            )

            # Convert the results into a Pandas DataFrame
            results_df = self.format_results_into_df(results)

            # Update the results placeholder with the new DataFrame
            self.results_placeholder.value = results_df
        except Exception as e:
            print(f"An error occurred: {str(e)}")
