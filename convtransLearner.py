import torch
import sys
import numpy as np
import time
from model_convtrans import  *
from utils import *
# teee
class convtransLearner(object):
    def __init__(self, **kwargs):
        src_vocab = kwargs['src_vocab']
        learning_rate = kwargs['lr']
        weight_decay = kwargs['wd']
        dropout = kwargs['dropout']
        batch_size = kwargs['bs']
        n_class = kwargs['n_class']
        K_d = kwargs['kernel_dim']
        kernel_w = kwargs['kernel_w']
        kernel_b = kwargs['kernel_b']
        act = kwargs['act']
        N, d_model, d_ff, h= kwargs['N'], kwargs['d_model'], kwargs['d_ff'], kwargs['h']
        print ('learning rate is: ', learning_rate)
        self.patience = 100
        self.filename = kwargs['record_file']
        self.model_name = kwargs['model_name']
        
        if kwargs['test']:
            self.model = torch.load(self.model_name)
        else:
            self.model = make_model(src_vocab, n_class, kernel_w, kernel_b, act=act, N=N, K_d=K_d, d_model=d_model, d_ff=d_ff, h=h, dropout=dropout)
            print ('weight_decay={}'.format(weight_decay))
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            #uses default settings for betas and eps
            #self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
            #self.scheduler = ReduceLROnPlateau(self.optimizer, 'max')

            self.batch_size = batch_size

    def train(self, num_epoch, train_loader, valid_loader, class_weights):
        
        pos_weight = class_weights['positive_weights']
        neg_weight = class_weights['negative_weights']

        verbose=False
        self.file_obj = open(self.filename, 'w')
        self.file_obj.write('Training:\n')
        self.file_obj.close()
        best_score = -10000000000
        k=0
        for epoch in range(num_epoch):
            self.model.train()
            start = time.time()
            total_loss = 0
            #auc = self.eval(train_loader)
            #output = ('(pre) train auc {:.4f}'.format(auc))
            #print(output)
            #with open(self.filename, 'a') as outfile:
            #    outfile.write(output+'\n')
            for batch_train_x, batch_train_y in train_loader:
                batch_train_x = batch_train_x.cuda()
                batch_train_y = batch_train_y.cuda()
                batch_start_time = time.time()
                self.optimizer.zero_grad()
                loss_train = self.model.loss(x=batch_train_x, y=batch_train_y, pos_weight=pos_weight, neg_weight=neg_weight, reduction='mean')
                loss_train.backward()
                total_loss += float(loss_train.data)
                self.optimizer.step()
                batch_cost_time = time.time() - batch_start_time
                if verbose:
                    sys.stdout.write(str(batch_cost_time))
                    sys.stdout.flush()

            auc = self.eval(valid_loader)
            seconds = time.time() - start
            output = ('Epoch={}, time {:.4f}, train loss {:.4f} '.format(epoch, seconds, total_loss)
                    +  'valid auc {:.4f}'.format(auc))
            print (output)
            with open(self.filename, 'a') as outfile:
                outfile.write(output+'\n')
            #self.scheduler.step(auc)
            if auc >= best_score+.0001:
                #reset early stop counter
                k=0
                best_score = auc
                torch.save(self.model, self.model_name)
            #add 1 to early top counter
            else: k+=1
            #stop training if beyond patience - # of epochs without validation improvement
            if k >= self.patience:
              break

   
    def eval(self, data_loader, form='mean'):
        self.model.eval()
        all_score = []
        all_y = []
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            
            #get reverse complements
            batch_x_rev = batch_x.detach().clone()
            for s in range(batch_x.shape[0]):
                batch_x_rev[s][:,[0, 3]] = batch_x_rev[s][:,[3, 0]] 
                batch_x_rev[s][:,[1, 2]] = batch_x_rev[s][:,[2, 1]] 
            
            #get forward scores
            score_fw = self.model.predict(batch_x)
            #get reverse scores
            score_rev = self.model.predict(batch_x_rev)
            
            #set score to avg of forward and reverse
            score = torch.mul(score_fw + score_rev, .5)
            
            all_score.append(score.data.cpu())
            all_y.append(batch_y.cpu())

        all_score = np.concatenate(all_score)
        all_y= np.concatenate(all_y)
            
        return evaluate(all_y, all_score, form)
