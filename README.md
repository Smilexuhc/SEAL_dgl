# DGL Implementation of the SEAL Paper
This DGL example implements the link prediction model proposed in the paper 
[Link Prediction Based on Graph Neural Networks](https://arxiv.org/pdf/1802.09691.pdf) 
and [REVISITING GRAPH NEURAL NETWORKS FOR LINK PREDICTION](https://arxiv.org/pdf/2010.16103.pdf)  
The author's codes of implementation is in [SEAL](https://github.com/muhanzhang/SEAL) (pytorch)
and [SEAL_ogb](https://github.com/facebookresearch/SEAL_OGB) (torch_geometric)

Example implementor
----------------------
This example was implemented by [Smile](https://github.com/Smilexuhc) during his intern work at the AWS Shanghai AI Lab.

The graph dataset used in this example 
---------------------------------------

ogbl-collab

Yeast

TODO:  
1. Dataset prepare
2. negative sample  (finished)
3. node2vec with negative injection  
4. subgraph sample with node labeling  (finished)
5. gnn model
6. train and test loop 

