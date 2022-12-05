import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from torch import sigmoid, log, sub, neg, mul, add
from torch.nn import MultiheadAttention, LayerNorm

devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x):
        "Pass the input through each layer in turn." 
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class EncoderModel(nn.Module):
    """
    The overal model
    """
    def __init__(self, conv, pool, encoder, position_emb, n_class, d_model):
        super(EncoderModel, self).__init__()
        self.conv = conv
        self.d_model = d_model
        #self.normalize = norm
        #dim_model = d_model
        #self.avgPool = nn.AvgPool1d(K_hlf,K_hlf)
        self.pool = pool
        self.encoder = encoder
        self.pos_emb = position_emb
        self.out_ffn = nn.Linear(d_model, n_class)
        self.out_pred = nn.Sigmoid()
        #self.lay_predict = nn.Sequential(nn.Linear(d_model, n_class), nn.Sigmoid())
    
    
    def forward(self, src):
        "Take in and process masked src and target sequences."
        #print("Pre Conv: ",src.shape)
        x = self.conv(torch.transpose(src,1,2))
        x = F.relu(x)
        #print("Post Conv: ",x.shape)
        x = self.pool(x)
        #print("Post Pool: ",x.shape)
        
        tmp1 = torch.from_numpy(np.zeros((len(x), self.d_model-1, 1), dtype=np.float32)).float().to(devices)
        tmp2 = torch.from_numpy(np.zeros((len(x), 1, x.shape[2]+1), dtype=np.float32)).float().to(devices)
        tmp2[:, 0, 0] = 1
        x = torch.cat((tmp1, x), 2)
        x = torch.cat((tmp2, x), 1)
        x = self.pos_emb(torch.transpose(x,1,2))
        x = self.encode(x)
        x = x[:,0, :] #Get the special token representation
        
        #final output FF layer
        x = self.out_ffn(x)
        
        return x

    def encode(self, x):
        return self.encoder(x)

    def predict(self, x):
        # (batch_size, seq_length) = y_seq_question.shape
        # preds = torch.gather(all_preds[:,-1,:].view(batch_size, -1), 1, y_seq_question[:,-1].view(batch_size, 1))[:,0]
        x = self.forward(x) #x: nbatch x seq_len x vocab
        
        #take sigmoid 
        pred = self.out_pred(x)
        #pred = self.lay_predict(x)
        return pred
    
    def binary_cross_entropy_with_logits(self, y_logit, y_true, pos_weight=None, neg_weight=None, reduction='sum'):

        p = torch.tensor([1])
        p = p.to(devices)
        sig_x = sigmoid(y_logit)
        log_sig_x = log(sig_x)
        sub_1_x = sub(p, sig_x)
        sub_1_y = sub(p, y_true)
        log_1_x = log(sub_1_x)

        if pos_weight is not None:
            output = neg( add( mul(mul(y_true, log_sig_x), pos_weight), mul(mul(sub_1_y, log_1_x), neg_weight) ) )
        else:
            output = neg( add( mul(y_true, log_sig_x), mul(sub_1_y, log_1_x) ) )
        
        if reduction == 'mean':
            return output.mean()
        elif reduction == 'sum':
            return output.sum()
        elif reduction == 'none':
            return output
        
    def loss(self, x, y, pos_weight=None, neg_weight=None, reduction='mean'):
        if neg_weight is not None:
            pos_weight = pos_weight.to(devices)
            neg_weight = neg_weight.to(devices)
            out_logits = self.forward(x)
            l = self.binary_cross_entropy_with_logits(out_logits, y, pos_weight, neg_weight, reduction)
            return l
        else:
            pred = self.predict(x)
            l = F.binary_cross_entropy(pred, y)
            return l

def make_model(src_vocab=5, n_class=57, kernel_w=None, kernel_b=None, act='init', N=2, K_d=10,
               d_model=320, d_ff=1024, h=4, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiheadAttention(embed_dim = d_model, num_heads = h)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    conv = nn.Conv1d(in_channels=4, out_channels=d_model - 1, kernel_size=K_d)
    K_hlf = int(K_d / 2)
    pool = nn.MaxPool1d(K_hlf,K_hlf)
    model = EncoderModel(
        c(conv),
        c(pool),
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        c(position),
        n_class, 
        d_model
        )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model.to(devices)