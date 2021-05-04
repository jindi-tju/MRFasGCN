from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn.utils import *
from gcn.models import GCN, MLP


y_test = np.loadtxt("y_test_val")
outputs = np.loadtxt("outputs_val")
test_nmi=NMI(outputs,y_test)
print("NMI=","{:.5f}".format(test_nmi))