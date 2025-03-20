import sys
import random
import logging

import numpy as np
import torch
import pickle
from tqdm import tqdm

def set_seed(seed):
    """Sets random seed everywhere."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # use determinisitic algorithm
    print("Seed set", seed)


def get_logger(level=logging.INFO):
    log = logging.getLogger(__name__)
    if log.handlers:
        return log
    log.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S"
    )
    ch.setFormatter(formatter)
    log.addHandler(ch)
    return log


def save_pkl(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(file):
    with open(file, "rb") as f:
        return pickle.load(f)

def load_mosei(emo="7class"):
    unsplit = load_pkl("data/mosei/data_mosei_7class.pkl")
    print(unsplit.keys())

    data = {
        "train": [], "dev": [], "test": [],
    }
    trainVid = list(unsplit["train"])
    valVid = list(unsplit["dev"])
    testVid = list(unsplit["test"])
    
    # print(trainVid[0])
    
    spliter = {
        "train": trainVid,
        "dev": valVid,
        "test": testVid
    }

    for split in data:
        for copus in tqdm(spliter[split], desc=split):
            # print("---")
            data[split].append(
                {
                    "uid" : copus["vid"],
                    "speakers" : [0] * len(copus["speakers"]),
                    "labels" : copus['labels'],
                    "text": copus["text"],
                    "audio": copus["audio"],
                    "visual": copus["visual"],
                    "sentence" : copus["sentence"],
                }
            )
    
    return data