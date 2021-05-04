## MRFasGCN

This is a TensorFlow implementation of MRFasGCN for the task of semi-supervised community detection
The basic framework is GCN (Thomas N. Kipf, Max Welling, [Semi-Supervised Classification with Graph Convolutional Networks](http://arxiv.org/abs/1609.02907) (ICLR 2017))

## Installation

```bash
python setup.py install
```

## Requirements
* tensorflow (>0.12)
* networkx

## Run the demo

```bash
cd gcn
python train.py
```

## Data

We use the same datasets in [Semi-Supervised Classification with Graph Convolutional Networks].
You can download the datasets from https://github.com/kimiyoung/planetoid.

In order to use your own data, you have to provide 
* an N by N adjacency matrix (N is the number of nodes), 
* an N by D feature matrix (D is the number of features per node), and
* an N by E binary label matrix (E is the number of classes).