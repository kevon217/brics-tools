import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from brics_search.brics_dd.setup_index.embedder import HybridEmbedder
from brics_search.brics_dd.setup_index.upsert import PineconeUpsert
import pinecone
from brics_search.utils import helper
from dotenv import load_dotenv
import os

load_dotenv()

cfg = helper.compose_config(
    config_path="../configs/semantic_search",
    config_name="configure_brics_index",
    overrides=[],
)


# @hydra.main(config_name="configure_index")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))  # For debug purposes

    # Instantiate and run the HybridEmbedder
    df_processed = HybridEmbedder(cfg).run()

    # Get index object from Pinecone
    pinecone.init(
        api_key=os.getenv("API_KEY_PINECONE"), environment=os.getenv("API_ENV_PINECONE")
    )
    index = pinecone.Index(index_name=cfg.create_pinecone_index.index_name)
    index.describe_index_stats()
    # Instantiate and run the PineconeUpsert
    upserter = PineconeUpsert(cfg, index, df_processed)
    upserter.upsert()


if __name__ == "__main__":
    main(cfg)


# import pandas as pd
# import pinecone
# from sentence_transformers import SentenceTransformer

# from brics_search.utils import helper
# from brics_search.brics_dd.setup_index import setup_logger, log
# from brics_search.brics_dd.utils.embedder import HybridEmbedder
# from .functions import prep_dd, embed_dd, upsert_dd

# cfg = helper.compose_config(
#     config_path="../configs",
#     config_name="config",
#     overrides=[])


# @log(msg="Running BRICS Data Dictionary subset/embed/upsert pipeline")
# def configure_index(cfg):
#     """
#     Main flow pipeline for creating:
#      (1) Load/process BRICS Data Dictionary
#      (2) Embed BRICS Data Elements
#      (3) Upset BRICS Data Elements and metadata into Pinecone
#     """


#     metadata = GetMetadata()

#     upserter = UpsertEmbeddings()

#     df_dd, cfg = prep_dd(cfg.semantic_search)
#     df_embed, cfg = embed_dd(df_dd, cfg.semantic_search)
#     index, cfg = upsert_dd(df_embed, cfg.semantic_search)


#     return df_dd, df_embed, index, cfg


# if __name__ == "__main__":
#     df_dd, df_embed, index, cfg = configure_index(cfg)
