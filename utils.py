import torch
import numpy as np
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_curve
from torch.utils import data
from sklearn.utils import class_weight

class Dataset(data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

def load_data(batch_size=64, is_onehot=False, is_shuffle=True, get_class_weights=False, data_dir="../dataset/"):
    """
	# X_test = torch.FloatTensor(np.load(data_dir + 'X_test.npy'))	
	# Y_test = torch.FloatTensor(np.load(data_dir + 'Y_test.npy'))
	X_valid = torch.LongTensor(np.load(data_dir + 'X_valid.npy'))
	X_valid = X_valid.view(X_valid.shape[0], -1)
	Special_valid = torch.LongTensor(np.ones((X_valid.shape[0],1))) * 4

	X_valid = torch.cat((Special_valid, X_valid), dim=1)
	#X_valid = X_valid[:200]
	Y_valid = torch.FloatTensor(np.load(data_dir + 'Y_valid.npy'))
	#Y_valid = Y_valid[:200]
	return X_valid, Y_valid, X_valid, Y_valid
    """
    para = {'batch_size':batch_size,
            'shuffle': is_shuffle}

    if is_onehot:
        print ('load train')
        train_x = torch.FloatTensor(np.load(data_dir + 'train_x.npy'))
        train_y = torch.FloatTensor(np.load(data_dir + 'train_y.npy'))
        
        print ('load test')
        test_x = torch.FloatTensor(np.load(data_dir + 'test_x.npy'))
        test_y = torch.FloatTensor(np.load(data_dir + 'test_y.npy'))
        
        print ('load valid')
        valid_x = torch.FloatTensor(np.load(data_dir + 'valid_x.npy'))
        valid_y = torch.FloatTensor(np.load(data_dir + 'valid_y.npy'))
        
        
        train_x_rev = train_x.detach().clone()
        for s in range(train_x.shape[0]):
            train_x_rev[s][:,[0, 3]] = train_x_rev[s][:,[3, 0]] 
            train_x_rev[s][:,[1, 2]] = train_x_rev[s][:,[2, 1]] 

        valid_x_rev = valid_x.detach().clone()
        for s in range(valid_x.shape[0]):
            valid_x_rev[s][:,[0, 3]] = valid_x_rev[s][:,[3, 0]] 
            valid_x_rev[s][:,[1, 2]] = valid_x_rev[s][:,[2, 1]] 
            
        test_x_rev = test_x.detach().clone()
        for s in range(valid_x.shape[0]):
            test_x_rev[s][:,[0, 3]] = test_x_rev[s][:,[3, 0]] 
            test_x_rev[s][:,[1, 2]] = test_x_rev[s][:,[2, 1]] 

        train_x=torch.cat((train_x, train_x_rev), 0)
        train_y=torch.cat((train_y,train_y), 0)

        valid_x=torch.cat((valid_x, valid_x_rev), 0)
        valid_y=torch.cat((valid_y, valid_y), 0)
        
        test_x=torch.cat((test_x, test_x_rev), 0)
        test_y=torch.cat((test_y, test_y), 0)


        para['batch_size'] = batch_size
        train_set = Dataset(train_x, train_y)
        train_loader = data.DataLoader(train_set, **para)
            
        test_set = Dataset(test_x, test_y)
        test_loader = data.DataLoader(test_set, **para)

        valid_set = Dataset(valid_x, valid_y)
        valid_loader = data.DataLoader(valid_set, **para)
        
        # get balanced class weights for loss function - weights from training data
        class_weights_dict = {}
        if get_class_weights:
            class_weights=[class_weight.compute_class_weight('balanced',np.unique(train_y[:,k]),train_y[:,k].numpy()) for k in range(train_y.shape[1])]
            #convert to torch
            class_weights=torch.tensor(class_weights,dtype=torch.float)

            class_weights_dict['negative_weights'] = class_weights[:,1]
            class_weights_dict['positive_weights'] = class_weights[:,0]
        else:
            class_weights_dict['negative_weights'] = None
            class_weights_dict['positive_weights'] = None
        
    else:
        train_x = torch.LongTensor(np.load(data_dir + 'train_x.npy'))
        train_x = train_x.view(train_x.shape[0], -1)
        #train_x = train_x[:, :500]
        #train_x = train_x[:500]
        train_special = torch.LongTensor(np.ones((train_x.shape[0],1))) * 4
        train_x = torch.cat((train_special, train_x), dim=1)
        train_y = torch.FloatTensor(np.load(data_dir + 'train_y.npy'))

        para['batch_size'] = batch_size
        train_set = Dataset(train_x, train_y)
        train_loader = data.DataLoader(train_set, **para)

        para['batch_size'] = batch_size//3
        test_x = torch.LongTensor(np.load(data_dir + 'test_x.npy'))
        test_x = test_x.view(test_x.shape[0], -1)
        test_special = torch.LongTensor(np.ones((test_x.shape[0],1))) * 4
        test_x = torch.cat((test_special, test_x), dim=1)
        test_y = torch.FloatTensor(np.load(data_dir + 'test_y.npy'))

        test_set = Dataset(test_x, test_y)
        test_loader = data.DataLoader(test_set, **para)

        para['batch_size'] = batch_size//4
        valid_x = torch.LongTensor(np.load(data_dir + 'valid_x.npy'))
        valid_x = valid_x.view(valid_x.shape[0], -1)
        valid_special = torch.LongTensor(np.ones((valid_x.shape[0],1))) * 4
        valid_x = torch.cat((valid_special, valid_x), dim=1)
        valid_y = torch.FloatTensor(np.load(data_dir + 'valid_y.npy'))

        valid_set = Dataset(valid_x, valid_y)
        valid_loader = data.DataLoader(valid_set, **para)

    print (train_x.shape, valid_x.shape, test_x.shape)
    return  train_loader, valid_loader, test_loader, class_weights_dict

def pr_auc_score(labels, scores):
    precision, recall, th = precision_recall_curve(labels, scores)
    auc1=auc( recall, precision)
    return auc1

def evaluate(labels, scores, form):
    auc_list = []
    m_list = []
    n_class = labels.shape[1]
    if form == 'mean':
        for i in range(n_class):
            if not np.count_nonzero(labels[:,i]) == 0:
                auc_list.append(pr_auc_score(labels[:, i], scores[:, i]))
                #auc = pr_auc_score(labels[:,i], scores[:,i])
                #auc_list.append('{:.4f}'.format(auc1))
                #m_list.append(auc1)
        auc_mean=np.mean(auc_list)
        return auc_mean
    else:
        for i in range(n_class):
            if not np.count_nonzero(labels[:,i]) == 0:
                auc = pr_auc_score(labels[:,i], scores[:,i])
                auc_list.append('{:.4f}'.format(auc))
                #m_list.append(auc)
        #return ",".join(auc_list) + "\n{}\nMean: {:.4f}".format(len(m_list),np.mean(m_list))
        return(auc_list)

def evaluateroc(labels, scores, form):
    auc_list = []
    m_list = []
    n_class = labels.shape[1]
    
    if form == 'mean':
        for i in range(n_class):
            if not np.count_nonzero(labels[:,i]) == 0:
                auc_list.append(roc_auc_score(labels[:, i], scores[:, i]))
                #auc = roc_auc_score(labels[:,i], scores[:,i])
                #auc_list.append('{:.4f}'.format(auc1))
                #m_list.append(auc1)
        auc_mean=np.mean(auc_list)
        return auc_mean
    else:
        for i in range(n_class):
            if not np.count_nonzero(labels[:,i]) == 0:
                auc = roc_auc_score(labels[:,i], scores[:,i])
                auc_list.append('{:.4f}'.format(auc))
                #m_list.append(auc)
        #return ",".join(auc_list) + "\n{}\nMean: {:.4f}".format(len(m_list),np.mean(m_list))
        return(auc_list)
