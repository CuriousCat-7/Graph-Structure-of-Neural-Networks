import os
from .sample import *
from loguru import logger
from glob import glob
import json


def _dump(f, path, overwrite):
    if os.path.exists(path) and not overwrite:
        logger.warning("{} already exists", path)
        return
    logger.info(path)
    f().to_csv(path, compression="gzip")


def create_all(save_root="save/csv/", overwrite=False):
    os.makedirs(save_root, exist_ok=True)
    _dump(sample_harary, os.path.join(save_root, "harary-paper.csv.gz"), overwrite)
    _dump(sample_ring, os.path.join(save_root, "ring-paper.csv.gz"), overwrite)
    _dump(sample_er, os.path.join(save_root, "er-paper.csv.gz"), overwrite)
    _dump(sample_ba, os.path.join(save_root, "ba-paper.csv.gz"), overwrite)
    _dump(sample_ws, os.path.join(save_root, "ws-paper.csv.gz"), overwrite)
    _dump(sample_ws_flex, os.path.join(save_root, "ws_flex-paper.csv.gz"), overwrite)


def calculate_avg_cluster_path(save_root="save/csv/"):
    csv_paths = glob(f"{save_root}/*.csv.gz")
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        ccs = []
        apls = []
        logger.info(csv_path)
        for i in tqdm(range(len(df))):
            g = nx.node_link_graph(eval(df.iloc[i].graph))
            d = get_avg_cluater_path(g)
            ccs.append(d["cluster_coefficient"])
            apls.append(d["avg_path_length"])

        df["cluster_coefficient"] = ccs
        df["avg_path_length"] = apls
        df.to_csv(csv_path, compression="gzip")


if __name__ == "__main__":
    fire.Fire()
