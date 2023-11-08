# studysearch_app.py

## Table of Contents
- [Description](#description)
- [Core Features](#core-features)
- [Demonstration](#demonstration)
- [Installation](#installation)
  - [Using virtualenv](#using-virtualenv)
  - [Using Poetry](#using-poetry)
- [Configuration](#configuration)
- [Usage](#usage)
- [Contact](#contact)

## Description

*studysearch_app* is a search interface for the Federal Interagency for Traumatic Brain Injury Research (FITBIR) Data Repository, which is an instance of the National Institutes of Health's (NIH) Biomedical Research Informatics and Computing System (BRICS) platform. It is designed to facilitate natural language querying of metadata associated with traumatic brain injury (TBI) studies stored in the FITBIR Data Repository. Leveraging the capabilities of Retrieval-Augmented Generation (RAG) supported by the LlamaIndex data framework, **studysearch_app** provides efficient semantic search and LLM-based generative capabilities, enabling researchers to quickly find relevant studies in the FITBIR Data Repository. 

**NOTE**: This interface/pipeline is currently only set up to query high-level study metadata, not study data. Please use FITBIR's Query Tool module to retrieve actual study data.

**NOTE**: This interface/pipeline does not currently support chat-like conversations with query results.

## Core Features

- **Semantic Search**: Utilize advanced natural language processing to find the most relevant studies within the FITBIR Data Repository.
- **Retrieval-Augmented Generation**: Enhance search results with information retrieved by the RAG pipeline, providing comprehensive responses to complex queries.
- **User-friendly Interface**: Powered by Streamlit, the app provides a clean and interactive interface for users to input their queries and receive information.
- **Query History**: Track and review past queries and responses directly within the app.
- **Customizable Prompts**: Adjust the LLM (Large Language Model) prompts for tailored search inquiries.

## Demonstration GIF

![Demonstration GIF](./assets/llamaindex_rag_demo.gif)


## Installation
To set up *brics-tools* on your local machine, you can use either `virtualenv` or `Poetry` for managing your Python environment and dependencies. Follow these steps:

```bash
# Clone the repository
git clone https://github.com/kevon217/brics-tools

# Navigate to the project directory
cd brics-tools
```

### Using virtualenv
```bash
# If using virtualenv (recommended for general use)
virtualenv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### Using Poetry
```bash
# If using Poetry (recommended for consistent development environments)
poetry install

# Activate the virtual environment created by Poetry
poetry shell
```

Ensure you have Python 3.8+ installed on your system, and you are using the correct version of `pip` associated with Python 3.8+ when using `virtualenv`, or the correct Python version set in `pyproject.toml` if using Poetry.

## Configuration

Configure the application by setting up the necessary files within the `configs` directory. Here is the expected tree structure along with descriptions for each configuration file:

```plaintext
configs/
├── config_studyinfo.yaml             # Main application configuration file. Set defaults here.
│
├── document_creators/
│   └── studyinfo_document.yaml       # Config for creating documents for indexing.
│
├── evaluators/
│   └── studyinfo_question_dataset.yaml  # Config for datasets used to evaluate the search engine. (NOT CURRENTLY IMPLEMENTED)
│
├── index_managers/
│   └── studyinfo_vectorstore.yaml    # Config for managing the vector store in the index.
│
├── loaders/
│   └── studyinfo_loader.yaml         # Config for loading study information data.
│
├── node_parsers/
│   └── studyinfo_nodes.yaml          # Config for parsing nodes within the study information dataset.
│
├── preprocessors/
│   └── studyinfo_preprocessor.yaml   # Config for preprocessing study information data.
│
└── query_engines/
    └── studyinfo_query_engine.yaml   # Config for the study information query engine.
```

To use the "generative" component of the RAG pipeline, store your OpenAI API key as an environment variable in a top level **.env** file:

```env
OPENAI_API_KEY=sk-yourkeyhere
```

## Usage
After installation and proper yaml configuration, you can run the scripts below.

**NOTE**: *you'll have to specify directory/file paths in yaml config files as modules currently don't support interactive directory/file selection*

```bash
# Build the Data Repository Study Info Index
python brics_modules/data_repository/build_studyinfo_index.py

# Run the Streamlit studysearch_app.py
cd brics-tools
python streamlit run studysearch_app.py
```

## Contact

This pipeline was created by [Kevin Armengol](mailto:kevin.armengol@gmail.com) but any future modifications or enhancements will be performed by [Maria Bagonis](mailto:maria.bagonis@nih.gov) and [Olga Vovk](mailto:olga.vovk@nih.gov). 
