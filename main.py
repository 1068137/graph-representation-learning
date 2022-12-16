import torch
import os
import argparse
import numpy as np
import pandas as pd

from analysis.visualization import generate_plots
from models.gat import GraphAttentionNetwork
from training import train
from utils.dataloader import get_dataset
from utils.evaluation import test

print(torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def str2bool(v):
    if v.lower() in ("yes", "y", "true", "t"):
        return True
    elif v.lower() in ("no", "n", "false", "f"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", default="Cora")
parser.add_argument("--split", default="random")
parser.add_argument("--hidden_dim", default=64, type=int)
parser.add_argument("--v2", default=False, type=str2bool)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--heads", default=8, type=int)
parser.add_argument("--weight_decay", default=0.0005, type=float)
parser.add_argument("--dropout", default=0.6, type=float)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--runs", default=5, type=int)
parser.add_argument("--max_K", default=16, type=int)
args = parser.parse_args()

root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
dataset = get_dataset(args, root_dir)
num_nodes = dataset.data.num_nodes
num_edges = dataset.data.num_edges // 2
num_features = dataset.num_node_features
num_classes = dataset.num_classes

data = dataset.data.to(device)

K = args.max_K
accuracies = np.empty((args.runs, K))
run = True

if run:
    for i in range(args.runs):
        print("Run {}".format(i+1))
        print("=====")

        xs = []
        for k in range(1, K+1):
            model = GraphAttentionNetwork(
                input_dim=num_features,
                hidden_dim=args.hidden_dim,
                output_dim=num_classes,
                heads=k,
                v2=args.v2,
                dropout=args.dropout
            ).to(device)
            model1 = train(model, data, args)
            test_acc = test(model1, data, data.test_mask)
            xs.append(test_acc)
            print("Model (K = {}) Test Accuracy: {}".format(k, round(test_acc, 3)))
        accuracies[i] = np.array(xs)

    df1 = pd.DataFrame(accuracies, columns=list(range(1, K + 1)))
    df1.to_csv("gat{}_acc_{}".format("v2" if args.v2 else "", args.dataset))

FILES = ["gat{}_acc_{}".format("v2" if args.v2 else "", args.dataset)]

generate_plots(K, FILES)
