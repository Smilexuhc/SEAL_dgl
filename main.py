from utils import parse_arguments
from tqdm import tqdm
from dgl import NID
from torch.nn import BCEWithLogitsLoss
from utils import load_ogb_dataset, evaluate_hits
import torch
from sampler import PosNegEdgesGenerator, SEALSampler, SEALDataLoader
from model import GCN
import numpy as np
from logger import LightLogging
import time


def train(model, dataloader, loss_fn, optimizer, device):
    model.train()

    total_loss = 0
    pbar = tqdm(dataloader, ncols=70)
    for g, pair_nodes, labels in pbar:
        g = g.to(device)
        pair_nodes = pair_nodes.to(device)
        labels = labels.to(device)

        logits = model(g, g.ndata['z'], pair_nodes, g.ndata['feat'], g.ndata[NID], g.edata['edge_weight'])
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * g.batch_size

    return total_loss / len(dataloader)


def test_data_loader(dataloader, device, epochs=15, print_fn=print):
    print_fn("Data loader size: {}".format(len(dataloader)))
    start_time = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        for index, (g, pair_nodes, labels) in enumerate(dataloader):
            g = g.to(device)
            pair_nodes = pair_nodes.to(device)
            labels = labels.to(device)

            # if index % 100 == 0:
            #     print_fn("Batch-{}, graph_size: {:.0f}, cost time: {}s".format(index, g.num_nodes() / g.batch_size,
            #                                                                    time.time() - epoch_start))
        print_fn("Epoch-{}: {:.1f}s".format(epoch, time.time() - epoch_start))

    end_time = time.time()
    print_fn("Total {} epochs, mean cost time: {:.1f}s".format(epochs, (start_time - end_time) / epochs))


def evaluate(model, dataloader, device):
    model.eval()

    y_pred, y_true = [], []
    for g, pair_nodes, labels in tqdm(dataloader, ncols=70):
        g = g.to(device)
        pair_nodes = pair_nodes.to(device)
        labels = labels.to(device)

        logits = model(g, g.ndata['z'], pair_nodes, g.ndata[NID], g.edata['edge_weight'])
        y_pred.append(logits.view(-1).cpu())
        y_true.append(labels.y.view(-1).cpu().to(torch.float))

    y_pred, y_true = torch.cat(y_pred), torch.cat(y_true)
    pos_pred = y_pred[y_true == 1]
    neg_pred = y_pred[y_true == 0]

    return pos_pred, neg_pred


def main(args, print_fn=print):
    print_fn("Experiment arguments: {}".format(args))

    if args.random_seed:
        torch.manual_seed(args.random_seed)
    else:
        torch.manual_seed(2021)
    # Load dataset
    if args.dataset.startswith('ogbl'):
        graph, split_edge = load_ogb_dataset(args.dataset)
    else:
        raise NotImplementedError

    attribute_dim = graph.ndata['feat'].size(1)
    num_nodes = graph.num_nodes()

    # set gpu
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu_id)
    else:
        device = 'cpu'

    # Generate positive and negative edges and corresponding labels
    sample_generator = PosNegEdgesGenerator(graph, split_edge, args.neg_samples, args.subsample_ratio)
    train_edges, train_labels = sample_generator('train')
    val_edges, val_labels = sample_generator('valid')
    test_edges, test_labels = sample_generator('test')

    # Set sampler
    sampler = SEALSampler(graph, hop=args.hop, prefix=args.dataset, save_dir='./processed', print_fn=print_fn)
    train_graph_list, train_pair_nodes, train_labels = sampler('train', train_edges, train_labels)
    val_graph_list, val_pair_nodes, val_labels = sampler('val', val_edges, val_labels)
    test_graph_list, test_pair_nodes, test_labels = sampler('test', test_edges, test_labels)

    # Set data loader
    train_loader = SEALDataLoader(train_graph_list, train_pair_nodes, train_labels,
                                  batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = SEALDataLoader(val_graph_list, val_pair_nodes, val_labels,
                                batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = SEALDataLoader(test_graph_list, test_pair_nodes, test_labels,
                                 batch_size=args.batch_size, num_workers=args.num_workers)
    # print_fn('Start testing speed of data loader')
    # test_data_loader(train_loader, print_fn=print_fn)
    # raise ValueError('END')

    # set model
    if args.model == 'gcn':
        model = GCN(num_layers=args.num_layers,
                    hidden_units=args.hidden_units,
                    gcn_type=args.gcn_type,
                    pooling_type=args.pooling,
                    attribute_dim=attribute_dim,
                    node_embedding=None,
                    use_embedding=True,
                    num_nodes=num_nodes,
                    dropout=args.dropout)
    elif args.model == 'dgcnn':
        raise NotImplementedError
    else:
        raise ValueError('Model error')

    model = model.to(device)
    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    loss_fn = BCEWithLogitsLoss()
    # use variable parameters cause bug
    print_fn("Total parameters: {}".format(sum([p.numel() for p in model.parameters()])))

    # train and evaluate loop
    summary_val = []
    summary_test = []
    for epoch in range(args.epochs):
        start_time = time.time()
        loss = train(model=model,
                     dataloader=train_loader,
                     loss_fn=loss_fn,
                     optimizer=optimizer,
                     device=device)
        train_time = time.time()
        if epoch % args.eval_steps == 0:
            val_pos_pred, val_neg_pred = evaluate(model=model,
                                                  dataloader=val_loader,
                                                  device=device)
            test_pos_pred, test_neg_pred = evaluate(model=model,
                                                    dataloader=test_loader,
                                                    device=device)

            val_metric = evaluate_hits(args.dataset, val_pos_pred, val_neg_pred, args.hits_k)
            test_metric = evaluate_hits(args.dataset, test_pos_pred, test_neg_pred, args.hits_k)
            evaluate_time = time.time()
            print_fn("Epoch-{}, train loss: {:.4f}, hits@{}: val-{:.4f}, test-{:.4f}, "
                     "cost time: train-{:.1f}s, total-{:.1f}s".format(epoch, loss, args.hits_k, val_metric, test_metric,
                                                                      train_time - start_time,
                                                                      evaluate_time - start_time))
            summary_val.append(val_metric)
            summary_test.append(test_metric)

    # summary_val = np.array(summary_val)
    summary_test = np.array(summary_test)

    print_fn("Experiment Results:")
    print_fn("Best hits@{}: {:.4f}, epoch: {}".format(args.hits_k, np.max(summary_test), np.argmax(summary_test)))


if __name__ == '__main__':
    args = parse_arguments()
    logger = LightLogging(log_name='SEAL', log_path='./logs')
    main(args, logger.info)
