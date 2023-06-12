import os
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from dotenv import load_dotenv
import pinecone
import panel as pn


from brics_search.brics_dd.hybrid_search_app import HybridSearchApp
from brics_search.utils import helper

load_dotenv()

cfg = helper.compose_config(
    config_path="../configs/semantic_search",
    config_name="configure_brics_index",
    overrides=[],
)

# Get index object from Pinecone
pinecone.init(
    api_key=os.getenv("API_KEY_PINECONE"), environment=os.getenv("API_ENV_PINECONE")
)
index = pinecone.Index(index_name=cfg.create_pinecone_index.index_name)
index.describe_index_stats()


if __name__ == "__main__":
    app = HybridSearchApp(cfg, index)
    layout = app.make_layout()
    pn.serve(layout)
