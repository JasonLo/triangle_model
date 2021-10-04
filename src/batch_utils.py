import os, sqlite3
from abc import ABC, abstractmethod
from tqdm import tqdm
import pandas as pd
import meta

class ResultParser(ABC):
    def __init__(self, batch_name: str):
        super().__init__()
        self.batch_name = batch_name
        self.testset_name = None  # type: str (Implement in subclass)
        self.testset_csv_files = None  # type: str or list (Implement in subclass)
        self.batch_folder = f"models/{batch_name}"
        self.batch_config = meta.batch_json_to_df(
            os.path.join(self.batch_folder, "batch_config.json")
        )

    @abstractmethod
    def parse(self, result_csv: str) -> pd.DataFrame:
        """Parse file into condition averaged pandas dataframe"""
        # Testset specific, impelment in subclasses
        pass

    def get_conn(self) -> sqlite3.Connection:
        """Get conenection to sqlite3 database."""
        return sqlite3.connect(f"models/{self.batch_name}/results.db")

    def parse_all(self) -> pd.DataFrame:
        """Parse all self.testest_csv_name files in the batch, and merge with config setting."""
        results = []
        for code_name in tqdm(self.batch_config.code_name):
            if type(self.testset_csv_files) is list:
                for csv in self.testset_csv_files:
                    csv_file = os.path.join(self.batch_folder, code_name, "eval", csv)
                    results.append(self.parse(csv_file))
            else:
                csv_file = os.path.join(
                    self.batch_folder, code_name, "eval", self.testset_csv_files
                )
                results.append(self.parse(csv_file))

        results = pd.concat(results)
        self.df = pd.merge(results, self.batch_config, how="left", on="code_name")

    def df_to_db(self):
        with self.get_conn() as conn:
            self.df.to_sql(
                name=self.testset_name, con=conn, if_exists="replace", index=False
            )

    def df_to_csv(self, file: str = None):
        if file is None:
            file = f"{self.batch_folder}/results/{self.testset_name}.csv"
        self.df.to_csv(file)


class TarabanParser(ResultParser):
    def __init__(self, batch_name: str):
        super().__init__(batch_name)
        self.testset_name = "taraban"
        self.testset_csv_files = "taraban_triangle.csv"

    def parse(self, result_csv) -> pd.DataFrame:
        """Parse file into condition averaged pandas dataframe."""
        gp_vars = [
            "code_name",
            "epoch",
            "testset",
            "task",
            "output_name",
            "timetick",
            "cond",
        ]
        metrics_vars = ["acc", "csse", "sse"]

        df = pd.read_csv(result_csv)
        df["csse"] = df.sse.loc[df.acc == 1]
        df = df.groupby(gp_vars).mean().reset_index()
        df["freq"] = df.cond.apply(
            lambda x: "High"
            if x
            in (
                "High-frequency exception",
                "Regular control for High-frequency exception",
            )
            else "Low"
        )
        df["reg"] = df.cond.apply(
            lambda x: "Regular" if x.startswith("Regular") else "Exception"
        )
        return df[gp_vars + ["freq", "reg"] + metrics_vars]


class LexicalityParser(ResultParser):
    def __init__(self, batch_name: str):
        super().__init__(batch_name)
        self.testset_name = "lexicality"
        self.testset_csv_files = ["glushko_triangle.csv", "taraban_triangle.csv"]

    def parse(self, result_csv) -> pd.DataFrame:
        gp_vars = ["code_name", "epoch", "testset", "task", "output_name", "timetick"]
        metrics_vars = ["acc", "csse", "sse"]

        df = pd.read_csv(result_csv)
        df["csse"] = df.sse.loc[df.acc == 1]
        df = df.groupby(gp_vars).mean().reset_index()

        df["cond"] = df.testset.apply(lambda x: "word" if x == "taraban" else "nonword")
        return df[gp_vars + ["cond"] + metrics_vars]


class ImageabilityParser(ResultParser):
    def __init__(self, batch_name: str):
        super().__init__(batch_name)
        self.testset_name = "imageability"
        self.testset_csv_files = "hs04_img_240_triangle.csv"

    def parse(self, result_csv) -> pd.DataFrame:
        gp_vars = [
            "code_name",
            "epoch",
            "testset",
            "task",
            "output_name",
            "timetick",
            "cond",
        ]
        metrics_vars = ["acc", "csse", "sse"]

        df = pd.read_csv(result_csv)
        df["csse"] = df.sse.loc[df.acc == 1]
        df = df.groupby(gp_vars).mean().reset_index()

        df["fc"] = df.cond.apply(lambda x: x[:5])
        df["img"] = df.cond.apply(lambda x: x[-2:])
        return df[gp_vars + ["fc", "img"] + metrics_vars]
