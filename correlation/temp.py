import argparse
import os
import signal
import time
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
# from torch_geometric.utils import to_networkx
from tqdm import tqdm

# try:
#     from topobench.data.loaders.graph import TUDatasetLoader
#     from topobench.data.preprocessor import PreProcessor
#     from topobench.data.utils.utils import get_routes_from_neighborhoods
# except (ImportError, RuntimeError):
#     TUDatasetLoader = None
#     PreProcessor = None
#     get_routes_from_neighborhoods = None
# from torch_geometric.data import Data