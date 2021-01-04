import dgl
from utils import parse_arguments
from dgl.dataloading.negative_sampler import Uniform


def main(args):
    """

    Args:
        args (dict):
    """
    neg_sampler = Uniform(args['neg_samples'])


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
