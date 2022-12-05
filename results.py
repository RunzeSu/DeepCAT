import numpy as np
import h5py
import os
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import torch
from utils import *
from model_convtrans import  *
torch.cuda.empty_cache()
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from torch import sigmoid, log, sub, neg, mul, add

devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda = torch.cuda.is_available()
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

model = torch.load('output/m_conv_ls_3_ff_400_dm_320_N_2_a_1e-06_b_100_init')
#model.load_state_dict(torch.load('output/m_conv_ls_3_ff_400_dm_320_N_2_a_1e-06_b_100_init'))

train_loader, valid_loader, test_loader, class_weights = load_data(100, 
                                             is_onehot=True,
                                             is_shuffle=True,
                                             get_class_weights=False,
                                             data_dir='dataset/')
                           
all_score = []
all_y = []
for batch_x, batch_y in test_loader:
    batch_x = batch_x.cuda()
    batch_y = batch_y.cuda()
    score = model.predict(batch_x)
    all_score.append(score.data.cpu())
    all_y.append(batch_y.cpu())

yhat = np.concatenate(all_score)
y_test = np.concatenate(all_y)

auc_s = []
for i in range(57):
    if not np.count_nonzero(y_test[:,i]) == 0:
        tmp = roc_auc_score(y_test[:, i], yhat[:, i])
        #auc_s.append(round(tmp, 4))
        auc_s.append(tmp)
        
prauc = []
avgpr = []
for i in range(57):
    if not np.count_nonzero(y_test[:,i]) == 0:
        precision, recall, thresholds = precision_recall_curve(y_test[:, i], yhat[:, i])
        tmp1 = auc(recall, precision)
        #avgpr.append(average_precision_score(y_test[:, i], yhat[:, i]))
        prauc.append(tmp1)
prauc_m = np.mean(prauc)
auc_m = np.mean(auc_s)


np.save('results/roc.npy', auc_s)
np.save('results/pr.npy', prauc)

print(auc_m)
print(prauc_m)
