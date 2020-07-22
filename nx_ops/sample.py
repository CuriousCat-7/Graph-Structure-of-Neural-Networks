import networkx as nx
from networkx.generators.harary_graph import hnm_harary_graph
from .ws_flex import watts_strogatz_flexible_graph
import numpy as np
import pandas as pd
from tqdm import tqdm
import fire
from loguru import logger


def get_avg_cluater_path(g: nx.Graph) -> dict:
    try:
        cc = nx.average_clustering(g)
    except nx.NetworkXError as e:
        logger.warning(e)
        cc = np.nan
    try:
        apl = nx.average_shortest_path_length(g)
    except nx.NetworkXError as e:
        logger.warning(e)
        apl = np.nan
    return dict(
        cluster_coefficient=cc,
        avg_path_length=apl,
    )


def sample_ws(n=64, k_max=62, k_min=8, p_num=300, seed_num=30) -> pd.DataFrame:
    rows = []
    assert k_min < k_max <= n
    pbar = tqdm(total=int((k_max-k_min)*p_num*seed_num))
    for k in np.arange(k_min, k_max):
        for p in np.linspace(0, 1, p_num)**2:
            for seed in range(seed_num):
                g = nx.generators.watts_strogatz_graph(n, k, p, seed=seed)
                rows.append(dict(
                    method="ws", n=n, k=k, p=p,
                    graph=nx.node_link_data(g),
                ))
                pbar.update()
    return pd.DataFrame(rows)


def sample_ba(n=64, m_max=30, m_min=4, seed_num=300):
    rows = []
    pbar = tqdm(total=int(m_max-m_min)*seed_num)
    for m in np.arange(m_min, m_max):
        for seed in range(seed_num):
            g = nx.generators.barabasi_albert_graph(n, m, seed=seed)
            rows.append(dict(
                method="ba", n=n, m=m,
                graph=nx.node_link_data(g),
            ))
            pbar.update()
    return pd.DataFrame(rows)


def sample_ws_flex(n=64, k_max=62, k_min=8, p_num=300, seed_num=30) -> pd.DataFrame:
    rows = []
    assert k_min < k_max <= n
    pbar = tqdm(total=int((k_max-k_min)*p_num*seed_num))
    for k in np.arange(k_min, k_max):
        for p in np.linspace(0, 1, p_num)**2:
            for seed in range(seed_num):
                g = watts_strogatz_flexible_graph(n, k, p, seed=seed)
                rows.append(dict(
                    method="ws-flex", n=n, k=k, p=p,
                    graph=nx.node_link_data(g),
                ))
                pbar.update()
    return pd.DataFrame(rows)


def sample_er(n=64, m_max=int(64*63/2), m_min=int(64*4), seed_num=30):
    rows = []
    pbar = tqdm(total=int(m_max-m_min)*seed_num)
    e = n*(n-1)
    for m in np.arange(m_min, m_max):
        for seed in range(seed_num):
            g = nx.generators.erdos_renyi_graph(n, m/e)
            rows.append(dict(
                method="er", n=n, m=m,
                graph=nx.node_link_data(g),
            ))
            pbar.update()
    return pd.DataFrame(rows)


def sample_ring(n=64, k_max=62, k_min=8):
    rows = []
    pbar = tqdm(total=k_max-k_min)
    for k in np.arange(k_min, k_max):
        g = nx.generators.watts_strogatz_graph(n, k, 0.0)
        rows.append(dict(
            method="ring", n=n, k=k,
            graph=nx.node_link_data(g),
        ))
        pbar.update()
    return pd.DataFrame(rows)


def sample_harary(n=64, m_max=int(64*63/2), m_min=int(64*4)):
    rows = []
    pbar = tqdm(total=m_max-m_min)
    for m in np.arange(m_min, m_max):
        g = hnm_harary_graph(n, m)
        rows.append(dict(
            method="harary", n=n, m=m,
            graph=nx.node_link_data(g),
        ))
        pbar.update()
    return pd.DataFrame(rows)


if __name__ == "__main__":
    fire.Fire()
