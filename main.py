import dgl
from utils import parse_arguments
from torch.utils.data import TensorDataset,ConcatDataset,IterableDataset
from tqdm import tqdm
from dgl import NID
from torch.nn import BCEWithLogitsLoss
from utils import generate_pos_neg_edges, load_ogb_dataset
import torch


def train(model, dataloader, loss_fn, optimizer, device):
    model.train()

    total_loss = 0
    pbar = tqdm(dataloader, ncols=70)
    for g, pair_nodes, labels in pbar:
        g = g.to(device)
        pair_nodes = pair_nodes.to(device)
        labels = labels.to(device)

        logits = model(g, g.ndata['z'], pair_nodes, g.ndata[NID])
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(g)

    return total_loss / len(dataloader)


def main(args):

    # Load dataset
    if args.dataset.startswith('ogbl'):
        graph, split_edge = load_ogb_dataset(args.dataset)
    else:
        raise NotImplementedError

    # gpu setting
    if args.use_gpu !=0 and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # generate positive and negative edges and corresponding labels






if __name__ == '__main__':
    args = parse_arguments()
    main(args)
