from pathlib import Path
import pandas as pd


class UMLSLoader:
    def __init__(self, config: dict):
        """
        Initialize UMLSLoader with paths and settings from a config dictionary.
        """
        self.cfg = config
        self.data = {}

    def load_rrf(self, rrf_type: str):
        """
        Load an RRF file into a DataFrame.
        """
        filepath = (
            Path(self.cfg["mth_local"]["dirpath_mth"])
            / self.cfg["mth_local"]["RRF_files"][rrf_type]["filename"]
        )
        df = pd.read_csv(
            filepath,
            sep="|",
            header=None,
            names=self.cfg["mth_local"]["RRF_files"][rrf_type]["columns"],
            dtype="object",
        )
        df = df[self.cfg["mth_local"]["RRF_files"][rrf_type]["subset"]]
        self.data[rrf_type] = df

    def load_all(self):
        """
        Load all RRF files specified in the configuration.
        """
        for rrf_type in self.cfg["mth_local"]["RRF_files"].keys():
            self.load_rrf(rrf_type)
        return self.data
