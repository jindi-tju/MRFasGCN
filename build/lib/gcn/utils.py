import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from sklearn import preprocessing
import sys
import os
import gc
import math

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def load_data2(dataset_str):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset_str))
    path = "data/"
    idx_features = np.genfromtxt("{}{}.content".format(path, dataset_str), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features, dtype=np.float32)
    #features = sp.vstack((allx, tx)).tolil()
    #features[test_idx_reorder, :] = features[test_idx_range, :]

    idx_labels = np.genfromtxt("{}{}.labels".format(path, dataset_str), dtype=np.dtype(str))
    labels = encode_onehot(idx_labels)

    # build graph
    idx2=list(range(1,1019))
    idx = np.array(idx2, dtype=np.int32)
    print(idx.shape)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset_str), dtype=np.int32)
    a=list(map(idx_map.get, edges_unordered.flatten()))
    file=open('data.txt','w')  
    file.write(str(a));  
    file.close() 
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

    return features.todense(), adj, labels

def get_splits(y):
    idx_train = range(70)	#0,1,2,...,139
    idx_val = range(100, 250)	#200,201,...,499
    idx_test = range(250, 750)	#500,501,...,1499
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])
    val_mask = sample_mask(idx_val, y.shape[0])
    test_mask = sample_mask(idx_test, y.shape[0])
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask, val_mask, test_mask

def load_data(dataset_str):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    print("graph is....")
    print(type(graph))
    print("allx is....")
    print(type(allx))#
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    if dataset_str == 'nell.0.001':
        # Find relation nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(allx.shape[0], len(graph))
        isolated_node_idx = np.setdiff1d(test_idx_range_full, test_idx_reorder)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-allx.shape[0], :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-allx.shape[0], :] = ty
        ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]

        idx_all = np.setdiff1d(range(len(graph)), isolated_node_idx)

        if not os.path.isfile("data/planetoid/{}.features.npz".format(dataset_str)):
            print("Creating feature vectors for relations - this might take a while...")
            features_extended = sp.hstack((features, sp.lil_matrix((features.shape[0], len(isolated_node_idx)))),
                                          dtype=np.int32).todense()
            features_extended[isolated_node_idx, features.shape[1]:] = np.eye(len(isolated_node_idx))
            features = sp.csr_matrix(features_extended)
            print("Done!")
            save_sparse_csr("data/planetoid/{}.features".format(dataset_str), features)
        else:
            features = load_sparse_csr("data/planetoid/{}.features.npz".format(dataset_str))

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    np.savetxt('labels',labels)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    print(type(features))
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)

def  xuxiaowei(adj, content):
    adj=adj.todense()
    sample_num = adj.shape[0]
    sim_t = np.zeros((sample_num,sample_num))

    print ("adj---start")
    # compute similar by cost similar
    if dataset_str == 'cora':
		m=5429
    if dataset_str == 'citeseer':
		m=4732
    if dataset_str == 'pubmed':
		m=44338
    if dataset_str == 'nell.0.001':
		m=266144
    if dataset_str == 'fk107':
		m=26717
    sum = np.sum(adj,axis=1)	#n*1
    sum = sum.astype("float64")
    sum_trans = sum.T
    sum = sum*sum_trans
    sum = sum/(2*m)
    sim_t = sum-adj
    sim_t = np.asarray(sim_t)
    scale = np.max(sim_t)-np.min(sim_t)
    sim_t = sim_t/scale
    return sim_t

	
def  shijianbo(adj, content):
    content = content.todense()
    sample_num = content.shape[0]
    sim_c = np.zeros((sample_num,sample_num))

    # compute similar by cost similar
    content_t = content.T
    sim_c = np.asarray(content*content_t)
    content_2 = np.multiply(content,content)
    sum = np.sum(content_2,axis=1)
    sum_trans = sum.T
    sum = sum*sum_trans
    sum = np.sqrt(sum)
    min = np.ones((sample_num,sample_num))*1e-10
    sum = sum+min
    sim_c = sim_c/sum
    sim_c = np.asarray(sim_c)
    sim_c_row = np.sum(sim_c,axis=1)
    one_vec = np.ones(sample_num)
    one_vec = one_vec.T		#1*n
    sim_c_row = sim_c_row*one_vec
    sim_c = sim_c/(sim_c_row+min)
    scale = np.max(sim_c)-np.min(sim_c)
    sim_c = sim_c/scale

    #KNN
    print ("content------KNN")
    k_values=100
    sample_2_final=np.zeros((sample_num,sample_num))
    for i in xrange(sample_num):
        sample_2_sort=sorted(sim_c[i],reverse=True)
        k_order=0
        for j in xrange(k_values):
            find_index=np.where(sim_c[i]==sample_2_sort[j])
            for k in xrange(len(find_index[0])):
                sample_2_final[i][find_index[0][k]]=sample_2_sort[j]
                k_order+=1
                if (k_order==k_values):
                    break
            if (k_order==k_values):
                break
    print ("content------KNN---end")
    print np.max(sample_2_final)
    #sample_2_final=sim_c
    return sample_2_final
	
def NMI(A,B):
    B=np.argmax(B,axis=1)
    # len(A) should be equal to len(B)
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    #Mutual information
    MI = 0
    eps = 1.4e-45
    a=0
    for idA in A_ids:
        for idB in B_ids:
	    if(idA>0):
	        a=1
	    if(a==1):
            	idAOccur = np.where(A==idA)
            	idBOccur = np.where(B==idB)
            	idABOccur = np.intersect1d(idAOccur,idBOccur)
            	px = 1.0*len(idAOccur[0])/total
            	py = 1.0*len(idBOccur[0])/total
            	pxy = 1.0*len(idABOccur)/total
            	MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
    # Normalized Mutual information
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A==idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B==idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
    MIhat = 2.0*MI/(Hx+Hy)
    return MIhat