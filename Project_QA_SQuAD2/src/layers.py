#layers.py
"""Assortment of layers for use in BiDAF.py (and any other models incl. QANet.py).

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
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from util import masked_softmax


def my_leaky_relu(x):
    y = F.leaky_relu(x, 0.01)
    return y


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        
        self.embed = nn.Embedding.from_pretrained(word_vectors) #if don't want to further train the embeddings
        #self.embed = nn.Embedding(word_vectors.shape[0], word_vectors.shape[1], 0) #train embedding from scratch
        #if want to further train the embeddings. Use hooks so that it has a lower learning rate for finetuning.
        # with torch.no_grad():
        #     self.embed.weight = torch.nn.Parameter(word_vectors) #for finetuning
    
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb


class Embedding_with_charCNN(nn.Module):
    #Added by Amit
    """Embedding layer used by BiDAF, with the character-level component.
    char level embed: See assignment 5 handout (https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/assignments/a5.pdf)

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
        char_vocab_size (int): size of the char2idx vocabulary
        char_cnn_kernel (int): cnn kernel. use kernel of 5 to get better context within a word.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob, char_vocab_size, char_cnn_kernel=5):
        super(Embedding_with_charCNN, self).__init__()

        #word embedding
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors) #if don't want to further train the embeddings
        #self.embed = nn.Embedding(word_vectors.shape[0], word_vectors.shape[1], 0) #train embedding from scratch
        #if want to further train the embeddings. Use hooks so that it has a lower learning rate for finetuning.
        # with torch.no_grad():
        #     self.embed.weight = torch.nn.Parameter(word_vectors) #for finetuning
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        
        #char embedding
        self.char_embed = torch.nn.Embedding(num_embeddings=char_vocab_size, embedding_dim=hidden_size, padding_idx=0) #padding_idx is included in char_vocab_size
        self.char_conv = torch.nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=char_cnn_kernel, padding=1)

        #combined
        self.proj_combine = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)


    def forward(self, xw, xc):
        '''
        xw: word inputs (shape: (batch_size, seq_len)) #already padded at seq level
        xc: char inputs (shape: (batch_size, seq_len, max_word_len)) #already padded at seq level and word level
        '''
        #word embedding
        emb = self.embed(xw)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)

        #char embedding (a less efficient version is to use a for loop to go along seq_len instead of reshaping xc)
        batch_size, seq_len, max_word_len = xc.shape
        emb_c = xc.view(batch_size*seq_len, -1) #(batch_size*seq_len, max_word_len)
        emb_c = self.char_embed(emb_c) #(batch_size*seq_len, max_word_len, embed_dim)
        emb_c = F.dropout(emb_c, self.drop_prob, self.training)
        emb_c = emb_c.transpose(1,2) #(batch_size*seq_len, embed_dim, max_word_len) b'cse conv1d does convolution across the last dim
        emb_c = self.char_conv(emb_c) #(batch_size*seq_len, hidden_size, fout) where fout is (max_word_len+2p-k)/s + 1
        emb_c, _ = torch.max(emb_c, dim=2) #(batch_size*seq_len, hidden_size)
        emb_c = emb_c.view(batch_size, seq_len, -1) #(batch_size, seq_len, hidden_size)

        emb = torch.cat([emb, emb_c], dim=2) #(batch_size, seq_len, 2*hidden_size)
        emb = self.proj_combine(emb) #(batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb

class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = my_leaky_relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.
    

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        '''
        x is of shape `(batch, seq_len, input_size)`
        lengths is of shape (batch,). Use pad_pack_sequence() to keep the lengths sorted for ONNX
        '''
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # Apply RNN
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size). h0,c0 defaults to 0 matrix if not specified.

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight): #or could do for weight in self.parameters()
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size). Interesting to see dropout used here...
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size). Interesting to see dropout used here...

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias #(batch_size, c_len, q_len) #gives the similarity between each word in the context and the query

        # more details on shape
        # c.shape = (batch_size, c_len, hidden_dim)
        # q.shape = (batch_size, q_len, hidden_dim)
        # self.c_weight.shape = (hidden_dim, 1)
        # self.q_weight.shape = (hidden_dim, 1)
        # self.cq_weight.shape = (1, 1, hidden_dim)

        # s0
        # a = torch.matmul(c, self.c_weight)
        # a.shape = (batch_size, c_len, 1)
        # a2 = a.expand([-1,-1, q_len]) #basically broadcasts the array across the last dimension
        # a2.shape = (batch_size, c_len, q_len) 

        # s1
        # b = torch.matmul(q, self.q_weight)
        # b.shape = (batch_size, q_len, 1)
        # b2 = b.transpose(1,2)
        # b2.shape = (batch_size, 1, q_len)
        # b3 = b2.expand([-1,c_len,-1]) #basically broadcasts the array across the 2nd dimension
        # b3.shape = (batch_size, c_len, q_len)

        # s2
        # d = c * self.cq_weight
        # d.shape = (batch_size, c_len, hidden_dim) #will do broadcasting
        # d2 = q.transpose(1,2)
        # d2.shape = (batch_size, hidden_dim, q_len)
        # d3 = torch.matmul(d, d2)
        # d3.shape = (batch_size, c_len, q_eln)

        return s

    # def get_similarity_naive_veryslow(self, c, q):
    #     #c: batch x c_len x hidden_dim
    #     #q: batch x q_len x hidden_dim
    #     b, c_len = c.shape[0:2]
    #     q_len = q.shape[1]
    #     S = torch.zeros(b, c_len, q_len)
    #     for i in range(b):
    #         for j in range(c_len):
    #             for k in range(q_len):
    #                 x = torch.cat((q[i,j], c[i,k], q[i,j] * c[i,k]), dim=0).view(-1,1) #3*hidden_dim x 1
    #                 S[i,j,k] = torch.matmul(self.weight, x) #scalar
    

class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        '''
        att: (batch_size, c_len, 8 * hidden_size)
        mod: (batch_size, c_len, 2 * hidden_size)
        mask: (batch_size, c_len)
        '''
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod) # (batch_size, c_len, 1)
        mod_2 = self.rnn(mod, mask.sum(-1)) #(batch_size, c_len, 2 * hidden_size)
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2) # (batch_size, c_len, 1)

        # Shapes: (batch_size, c_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2