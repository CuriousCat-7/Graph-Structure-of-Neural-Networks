import os
from .sample import *
from loguru import logger


def _dump(f, path, overwrite):
    if os.path.exists(path) and not overwrite:
        logger.warning("{} already exists", path)
        return
    logger.info(path)
    f().to_csv(path)


def create_all(save_root="save/csv/", overwrite=False):
    os.makedirs(save_root, exist_ok=True)
    _dump(sample_harary, os.path.join(save_root, "harary-paper.csv"), overwrite)
    _dump(sample_ring, os.path.join(save_root, "ring-paper.csv"), overwrite)
    _dump(sample_er, os.path.join(save_root, "er-paper.csv"), overwrite)
    _dump(sample_ba, os.path.join(save_root, "ba-paper.csv"), overwrite)
    _dump(sample_ws, os.path.join(save_root, "ws-paper.csv"), overwrite)
    _dump(sample_ws_flex, os.path.join(save_root, "ws_flex-paper.csv"), overwrite)


if __name__ == "__main__":
    fire.Fire()
