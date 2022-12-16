import os
from torch_geometric.datasets import Planetoid

PLANETOID = ["Cora", "CiteSeer", "PubMed"]


def get_dataset(args, root_dir):
    if args.dataset in PLANETOID:
        return Planetoid(root=os.path.join(root_dir, "data", args.dataset), name=args.dataset, split=args.split)
    else:
        raise ValueError("Invalid dataset.")
