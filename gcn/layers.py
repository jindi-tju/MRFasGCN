from gcn.inits import *
import tensorflow as tf
from gcn.utils import *
import snap

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # gcn_convolution
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

class MRF_Convolution(Layer):
    """MRF_Convolutional layers"""
    def __init__(self, input_dim, output_dim,
                 theta_alpha, theta_beta, theta_gamma,
				placeholders, dropout=0.8, act=tf.nn.softmax, bias=False, **kwargs):
        super(MRF_Convolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.support = placeholders['support']
        self.act = act
        self.bias = bias
        if FLAGS.dataset == 'cora':
            num_node=2708
            output_dim=7
        if FLAGS.dataset == 'citeseer':
            num_node=3327
            output_dim=6
        if FLAGS.dataset == 'pubmed':
            num_node=19717
            output_dim=3
        if FLAGS.dataset == 'nell.0.001':
            num_node=65755
            output_dim=210
        if FLAGS.dataset == 'fk107':
            num_node=1018
            output_dim=10

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']
        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights1_' + str(i)] = uniform([1, 1],
                                                        name='weights1_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights2_' + str(i)] = uniform([1, 1],
                                                        name='weights2_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights3_' + str(i)] = uniform([output_dim, output_dim],
                                                        name='weights3_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        if FLAGS.dataset == 'cora':
            num_node=2708
            output_dim=7
        if FLAGS.dataset == 'citeseer':
            num_node=3327
            output_dim=6
        if FLAGS.dataset == 'pubmed':
            num_node=19717
            output_dim=3
        if FLAGS.dataset == 'nell.0.001':
            num_node=65755
            output_dim=210
        if FLAGS.dataset == 'fk107':
            num_node=1018
            output_dim=10

        #Calculate similarity
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
        if not os.path.isfile("data/xuxiaowei/{}.sim_top.npy".format(FLAGS.dataset)):
            print("Creating sin_top for relations - this might take a while...")
            sim_top = xuxiaowei(adj,features,FLAGS.dataset)	#sim_top is n*n matrix
            np.savetxt("res2",sim_top[0])
            print("Done!")
            np.save("data/xuxiaowei/{}.sim_top.npy".format(FLAGS.dataset), sim_top)
        else:
            sim_top = np.load("data/xuxiaowei/{}.sim_top.npy".format(FLAGS.dataset))
            print ("xuxiaowei------OK")
        if not os.path.isfile("data/shijianbo/{}.sim_con.npy".format(FLAGS.dataset)):
                print("Creating sin_con for relations - this might take a while...")
                sim_con = shijianbo(adj,features)	#sim_content is n*n matrix
                print("Done!")
                np.save("data/shijianbo/{}.sim_con.npy".format(FLAGS.dataset), sim_con)
        else:
                sim_con = np.load("data/shijianbo/{}.sim_con.npy".format(FLAGS.dataset))
        print ("shijianbo------OK")

        # Initialization
        q_values_1 = inputs
        q_values=(-1)*q_values_1
        q_values = tf.nn.softmax(q_values_1)

        # Constructing similarity matrix
        sim_top = np.where(abs(sim_top)>5e-2, sim_top, 0)
        alpha = 0.1
        beta = 0.1
        kernel_11 = alpha*sim_top - beta*sim_con
        kernel_11 = kernel_11.astype(np.float32)
        kernel_11 = sp.coo_matrix(kernel_11)
        kernel_1 = tf.SparseTensor(indices=np.array([kernel_11.row, kernel_11.col]).T, values=kernel_11.data,dense_shape=kernel_11.shape)

        # Message passing
        supports = list()
        for i in range(len(self.support)):
            message_passing = tf.sparse_tensor_dense_matmul(kernel_1, q_values)
            # Compatibility transform
            compatibility1 = (-2)*np.eye(output_dim) + np.ones((output_dim,output_dim))
            compatibility1 = (-1)*compatibility1
            compatibility = tf.convert_to_tensor(compatibility1,dtype=tf.float32)	#���úõ���Ӧ����
            compatibility = self.vars['weights3_' + str(i)]*compatibility
            alpha2 = 1
            pairwise = alpha2*tf.matmul(message_passing, compatibility)
            # Adding unary potentials
            beta2 = 1
            q_values = beta2*(q_values_1 - pairwise)
            support = q_values
            supports.append(support)
            output = tf.add_n(supports)
            print(output)
            # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)
