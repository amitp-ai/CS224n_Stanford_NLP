#BiDAF.py
"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
    Amit Patel (amitpatel.gt@gmail.com)
"""

'''
NOTE: ALL THE PATHS ARE REFERENCED FROM THE DIRECTORY WHERE THE COLAB NOTEBOOK IS SAVED BCSE OS.CHDIR() \
IS MADE TO POINT THERE AND ALL .py FILES INSIDE SRC DIRECTORY ARE ONLY CALLED FROM THE COLAB NOTEBOOK AND ALSO BCSE \
THE PATH TO ALL THE SRC/*.PY FILES IS ADDED TO SYS.PATH.
'''

import torch
import torch.nn as nn
from functools import reduce

import layers

from ujson import load as json_load
#Declare Important Global Variables
#load char2idx_dict
char2idx_file = './data/char2idx.json'
with open(char2idx_file, 'r') as fh:
    char2idx_dict = json_load(fh)


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD BiDAF:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)
        self.reset()


    def reset(self):
        '''
        Initialize with Glorot initialization
        '''
        # z = list(self.children()) #not recursive
        # print(list(z[0].modules()), print(type(self.lyr_norm1)))
        with torch.no_grad():
            for m in self.modules(): #self.modules() will recursively return all the modules in the network
                if type(m) in [torch.nn.modules.linear.Linear, torch.nn.modules.normalization.LayerNorm]:
                    dims = reduce(lambda x,y: x+y, m.weight.shape)
                    rng = torch.sqrt(torch.tensor(6.0/(dims)))
                    m.weight.uniform_(-rng, rng)
                    # print(dims, type(m), m.weight.shape)
                elif type(m) == torch.nn.modules.conv.Conv1d:
                    dims = reduce(lambda x,y: x*y, m.weight.shape[1:]) + m.weight.shape[0]
                    rng = torch.sqrt(torch.tensor(6.0/(dims)))
                    m.weight.uniform_(-rng, rng)
                    # print(dims, type(m), m.weight.shape)
                # print(m, type(m))


    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs #(batch_size, c_len). 0 == padded word
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs #(batch_size, q_len). 0 == padded word
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1) #gives the true length (i.e. unpadded len) of each example in the batch

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


class BiDAF_with_char_embed(nn.Module):
    """Baseline BiDAF model for SQuAD with CNN based char level embedding

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD BiDAF:
        - Embedding layer: Embed word and char indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
        char_vocab_size (int): size of the char2idx vocabulary
        char_cnn_kernel (int): cnn kernel. use kernel of 5 to get better context within a word.        
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF_with_char_embed, self).__init__()

        #char2idx_dict is a global variable
        char_vocab_size = len(char2idx_dict)+2 #add pad and --OOV--- tokens (i.e 0 and 1)
        char_cnn_kernel=5

        self.emb = layers.Embedding_with_charCNN(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob, char_vocab_size=char_vocab_size,
                                    char_cnn_kernel=char_cnn_kernel)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):

        c_mask = torch.zeros_like(cw_idxs) != cw_idxs #(batch_size, c_len)
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs #(batch_size, q_len)
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1) #(batch_size,)

        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out