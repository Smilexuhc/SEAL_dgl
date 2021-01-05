import dgl
from utils import parse_arguments
from dgl.dataloading.negative_sampler import Uniform
from tqdm import tqdm
from dgl import NID
from torch.nn import BCEWithLogitsLoss


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
    """

    Args:
        args (dict):
    """
    neg_sampler = Uniform(args['neg_samples'])


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
