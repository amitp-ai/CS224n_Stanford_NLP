import layers
import torch
import util
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import numpy as np
import sys
from functools import reduce

"""
Author:
    Amit Patel (amitpatel.gt@gmail.com)
"""

'''
NOTE: ALL THE PATHS ARE REFERENCED FROM THE DIRECTORY WHERE THE COLAB NOTEBOOK IS SAVED BCSE OS.CHDIR() \
IS MADE TO POINT THERE AND ALL .py FILES INSIDE SRC DIRECTORY ARE ONLY CALLED FROM THE COLAB NOTEBOOK AND ALSO BCSE \
THE PATH TO ALL THE SRC/*.PY FILES IS ADDED TO SYS.PATH.
'''

'''
References:
        1. https://arxiv.org/pdf/1804.09541.pdf
        2. https://github.com/NLPLearn/QANet (Note: everywhere this uses a conv layer to do projection instead of a linear layer)
        3. https://github.com/hackiey/QAnet-pytorch
        4. https://github.com/BangLiu/QANet-PyTorch
'''


from ujson import load as json_load
#Declare Important Global Variables
#load char2idx_dict
char2idx_file = './data/char2idx.json'
with open(char2idx_file, 'r') as fh:
    char2idx_dict = json_load(fh)


class QANet(nn.Module):
    """QANet model for SQuAD2 with CNN based char level embedding

    Based on the paper:
    QANet (https://arxiv.org/pdf/1804.09541.pdf)

    Follows a high-level structure commonly found in SQuAD BiDAF:
        - Embedding layer: Embed word and char indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor of size vocab_size x 300): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size=128, drop_prob=0.0, num_mod_enc_blocks=7, bias=False):
        super(QANet, self).__init__()

        #args
        #char2idx_dict is a global variable
        char_vocab_size = len(char2idx_dict)+2 #add pad and --OOV--- tokens (i.e 0 and 1)
        char_dim=200 #word_embed dim is 300
        if drop_prob == 0.0:
            self.wrd_drp_prb = self.chr_drp_prb = self.drop_prob = self.stchstc_dpth_drp_prb = 0.0
        else:
            self.wrd_drp_prb = drop_prob #QANet paper
            self.chr_drp_prb = drop_prob/2 #QANet paper
            self.drop_prob = drop_prob #QANet paper
            self.stchstc_dpth_drp_prb = 0.1 #QANet paper

        #model definition
        self.emb = QANet_Embed_with_charCNN(word_vectors=word_vectors, char_dim=char_dim,
                                    hidden_size=hidden_size, w_drop_prob=self.wrd_drp_prb,
                                    c_drop_prob = self.chr_drp_prb, char_vocab_size=char_vocab_size, char_cnn_kernel=5)

        self.emb_enc = QANet_Transformer(hidden_size=hidden_size, kernel=7, num_conv_layers=4, 
                            num_enc_blocks=1, drop_prob=self.drop_prob, 
                            stchstc_dpth_drp_prb = self.stchstc_dpth_drp_prb, num_atten_heads=8, bias=bias)

        self.att = layers.BiDAFAttention(hidden_size=hidden_size, drop_prob=drop_prob)

        self.mod_enc_input_prj = torch.nn.Linear(4*hidden_size, hidden_size)
        self.mod_enc = QANet_Transformer(hidden_size=hidden_size, kernel=5, num_conv_layers=2, 
                            num_enc_blocks=num_mod_enc_blocks, drop_prob=self.drop_prob, 
                            stchstc_dpth_drp_prb = self.stchstc_dpth_drp_prb, num_atten_heads=8, bias=bias)

        self.out = QANetOutput(hidden_size=hidden_size, bias=bias)
        self.reset()


    def reset(self):
        '''
        Initialize with Kaiming initialization
        '''
        # z = list(self.children()) #not recursive
        # print(list(z[0].modules()), print(type(self.lyr_norm1)))
        for m in self.modules(): #self.modules() will recursively return all the modules in the network
            # print(m.weight.shape, type(m), m.bias.shape if m.bias != None else m.bias)
            if type(m) in [torch.nn.modules.linear.Linear, torch.nn.modules.conv.Conv1d]:
                torch.nn.init.kaiming_uniform_(m.weight, a=0.01) #2.3
                if m.bias != None: torch.nn.init.zeros_(m.bias)
            elif type(m) == torch.nn.modules.normalization.LayerNorm:
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)



    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        # cw_idxs = cw_idxs.to(torch.long)
        # cc_idxs = cc_idxs.to(torch.long)
        # qw_idxs = qw_idxs.to(torch.long)
        # qc_idxs = qc_idxs.to(torch.long)

        # print('AT THE BEGINNING')
        # GPU_Memory_Usage()

        c_mask = torch.zeros_like(cw_idxs) != cw_idxs #(batch_size, c_len)
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs #(batch_size, q_len)
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1) #(batch_size,) #don't need this for QANet
        # print('AFTER C/Q_MASK')
        # GPU_Memory_Usage()

        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, hidden_size)
        #c_emb = F.dropout(c_emb, self.drop_prob, self.training)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, hidden_size)
        #q_emb = F.dropout(q_emb, self.drop_prob, self.training)
        # print('AFTER C/Q_EMB')
        # GPU_Memory_Usage()

        c_enc = self.emb_enc(c_emb, c_mask)    # (batch_size, c_len, hidden_size)
        #c_enc = F.dropout(c_enc, self.drop_prob, self.training)
        q_enc = self.emb_enc(q_emb, q_mask)    # (batch_size, q_len, hidden_size)
        #q_enc = F.dropout(q_enc, self.drop_prob, self.training)
        # print('AFTER C/Q_ENC')
        # GPU_Memory_Usage()

        atten_out = self.att(c_enc, q_enc, c_mask, q_mask)    # (batch_size, c_len, 4*hidden_size)
        # print('AFTER ATTEN_OUT')
        # GPU_Memory_Usage()

        atten_out_prj = self.mod_enc_input_prj(atten_out) # # (batch_size, c_len, hidden_size)
        atten_out_prj = F.dropout(atten_out_prj, self.drop_prob, self.training) #https://github.com/NLPLearn/QANet uses a conv layer to do this
        
        M0 = self.mod_enc(atten_out_prj, c_mask)        # (batch_size, c_len, hidden_size)
        # print('AFTER MOD_ENC1')
        # GPU_Memory_Usage()

        M1 = self.mod_enc(M0, c_mask)        # (batch_size, c_len, hidden_size)
        # M1 = F.dropout(M1, self.drop_prob, self.training) #https://github.com/NLPLearn/QANet
        # print('AFTER MOD_ENC2')
        # GPU_Memory_Usage()

        M2 = self.mod_enc(M1, c_mask)        # (batch_size, c_len, hidden_size)
        # print('AFTER MOD_ENC3')
        # GPU_Memory_Usage()

        p_strt, p_stp = self.out(M0, M1, M2, c_mask)  # 2 tensors, each (batch_size, c_len)
        # print('AFTER OUTPUT LAYER')
        # GPU_Memory_Usage()

        return (p_strt, p_stp)


class QANet_Embed_with_charCNN(nn.Module):
    """Embedding layer used by QANet, with the character-level component.
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
    def __init__(self, word_vectors, char_dim, hidden_size, w_drop_prob, c_drop_prob, char_vocab_size, char_cnn_kernel):
        super(QANet_Embed_with_charCNN, self).__init__()
        self.w_drop_prob = w_drop_prob #per QANet paper
        self.c_drop_prob = c_drop_prob #per QANet paper

        #word embedding
        self.embed = nn.Embedding.from_pretrained(word_vectors) #if don't want to further train the embeddings
        
        #char embedding
        padding = (char_cnn_kernel-1)//2 #to maintain same dimension between input and output assuming stride=1 and kernel size is odd number
        self.char_embed = torch.nn.Embedding(num_embeddings=char_vocab_size, embedding_dim=char_dim, padding_idx=0) #padding_idx is included in char_vocab_size
        self.char_conv = torch.nn.Conv1d(in_channels=char_dim, out_channels=char_dim, kernel_size=char_cnn_kernel, padding=padding)

        #combine
        self.proj_combine = torch.nn.Linear(char_dim+word_vectors.shape[1], hidden_size, bias=False)
        self.hwy = layers.HighwayEncoder(2, hidden_size)


    def forward(self, xw, xc):
        '''
        xw: word inputs (shape: (batch_size, seq_len)) #already padded at seq level
        xc: char inputs (shape: (batch_size, seq_len, max_word_len)) #already padded at seq level and word level
        '''
        #word embedding
        emb_w = self.embed(xw)   # (batch_size, seq_len, w_dim)
        emb_w = F.dropout(emb_w, self.w_drop_prob, self.training)

        #char embedding (a less efficient version is to use a for loop to go along seq_len instead of reshaping xc)
        batch_size, seq_len, max_word_len = xc.shape
        emb_c = xc.view(batch_size*seq_len, -1) #(batch_size*seq_len, max_word_len)
        emb_c = self.char_embed(emb_c) #(batch_size*seq_len, max_word_len, char_dim)
        emb_c = F.dropout(emb_c, self.c_drop_prob, self.training)
        emb_c = emb_c.transpose(1,2) #(batch_size*seq_len, char_dim, max_word_len) b'cse conv1d does convolution across the last dim
        emb_c = self.char_conv(emb_c) #(batch_size*seq_len, char_dim, fout) where fout is (max_word_len+2p-k)/s + 1
        emb_c, _ = torch.max(emb_c, dim=2) #(batch_size*seq_len, char_dim)
        emb_c = emb_c.view(batch_size, seq_len, -1) #(batch_size, seq_len, char_dim)

        emb = torch.cat([emb_w, emb_c], dim=2) #(batch_size, seq_len, w_dim+char_dim)
        emb = self.proj_combine(emb) #(batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb


class QANet_Transformer(nn.Module):
    '''
    This is the main innovation of QANet. Instead of RNN, it uses a transformer to get contextual encoding of the question/context embedding
    '''
    def __init__(self, hidden_size, kernel, num_conv_layers, num_enc_blocks, drop_prob, stchstc_dpth_drp_prb, num_atten_heads, bias=False):
        super(QANet_Transformer, self).__init__()

        encoder_blocks_list = []
        for _ in range(num_enc_blocks):
            encoder_block = QANet_Encoder(hidden_size, kernel, num_conv_layers, drop_prob, stchstc_dpth_drp_prb, num_atten_heads, bias=bias)
            encoder_blocks_list.append(encoder_block)
        self.encoder_blocks_list = nn.ModuleList(encoder_blocks_list)

    def forward(self, x, x_mask):
        #x: batch_size x max_seq_len x hidden_size
        #x_mask: batch_size x max_seq_len

        tot_numof_blocks = len(self.encoder_blocks_list)
        x_mask = x_mask.unsqueeze(-1)
        for block_num, encoder in enumerate(self.encoder_blocks_list):
            x = encoder(x, x_mask, block_num, tot_numof_blocks)
        return x #batch_size x max_seq_len x hidden_size



def residual_layer_dropout(x, x_post_layer, drop_prob, is_training):
    """Stochastic Depth Dropout: It is used to dropout the entire "main" layer when doing residual connection.
    (based upon https://github.com/NLPLearn/QANet)
    Ref: https://arxiv.org/pdf/1603.09382.pdf
         https://users.cecs.anu.edu.au/~sgould/papers/dicta16-depthdropout.pdf
    
    Args:
        x (tensor): input to the "main" layer
        x_post_layer (tensor of same shape as x): output of the "main" layer
        drop_prob (float): probability to dropout "main" layer
        is_training (bool): flag used to modify the behavior during inference vs training
    """
    # return x_post_layer + x #don't do layer dropout

    #Note: relu() is done in the calling function after calling this layer   
    if is_training:
        if torch.rand(1) < drop_prob:
            out = x
        else:
            out = x_post_layer + x
    else: #compute the expected value of the sampling process used during training (this will make it deterministic during inference)
        out = drop_prob * x + (1-drop_prob) * (x_post_layer + x)
    return out


class QANet_Encoder(nn.Module):
    """
    This is the key building block
    """
    def __init__(self, hidden_size, kernel, num_conv_layers, drop_prob, stchstc_dpth_drp_prb, num_atten_heads, bias):
        super(QANet_Encoder, self).__init__()

        self.num_conv_layers = num_conv_layers
        self.drop_prob = drop_prob
        self.stchstc_dpth_drp_prb = stchstc_dpth_drp_prb
        self.pos_enc = PositionalEncoding(hidden_size, max_len=int(1e4)) #not sure why in the paper they have PositionalEncoding inside the encoder
        conv_layers = []
        conv_norm_layers = []
        for _ in range(num_conv_layers):
            layer_norm = nn.LayerNorm(hidden_size) #do layer norm only along the last dimension which is hidden_size
            conv_norm_layers.append(layer_norm)
            conv = Depthwise_Sep_Conv_Layer(kernel, hidden_size, hidden_size)
            conv_layers.append(conv)

        self.conv_norm_layers = nn.ModuleList(conv_norm_layers)
        self.conv_layers = nn.ModuleList(conv_layers)

        self.self_atten_layer_norm = nn.LayerNorm(hidden_size) #do layer norm only along the last dimension which is hidden_size
        self.self_attention = MultiHeadAtten_Pytorch(hidden_size, num_atten_heads, drop_prob, bias=bias)

        self.feed_fwd_layer_norm = nn.LayerNorm(hidden_size) #do layer norm only along the last dimension which is hidden_size
        self.feed_fwd = torch.nn.Linear(hidden_size, hidden_size, bias=bias)


    def forward(self, x, x_mask, block_num, tot_numof_blocks):
        #x: batch_size x max_seq_len x input_size
        #x_mask: batch_size x max_seq_len (0 where it is padded)
        #block_num: block number (used with residual_layer_dropout())
        #tot_numof_blocks: total number of blocks in this module (used with residual_layer_dropout())

        tot_sub_layers = (self.num_conv_layers+2)*tot_numof_blocks
        pos_lyr_drpout = (self.num_conv_layers+2)*block_num
        #positional encoding
        x = x + self.pos_enc(x)
        x = x * x_mask

        # print('ENCODER: AFTER POSITIONAL ENCODING')
        # GPU_Memory_Usage()
        
        #conv
        for i in range(self.num_conv_layers):
            x_temp = self.conv_norm_layers[i](x)
            x_temp = x_temp * x_mask
            x_temp = self.conv_layers[i](x_temp, x_mask)
            if i%2 == 0: #have dropout on only half of the conv layers
                x_temp = F.dropout(x_temp, self.drop_prob, self.training)
            x = residual_layer_dropout(x, x_temp, self.stchstc_dpth_drp_prb * float(pos_lyr_drpout)/tot_sub_layers, 
                self.training)
            x_temp = layers.my_leaky_relu(x_temp)
            pos_lyr_drpout += 1
        # print('ENCODER: AFTER CONVOLUTIONAL LAYERS')
        # GPU_Memory_Usage()

        #self-attention
        x_temp = self.self_atten_layer_norm(x)
        x_temp = F.dropout(x_temp, self.drop_prob, self.training)
        x_temp = x_temp * x_mask
        x_temp = self.self_attention(x_temp, x_mask)
        x_temp = x_temp * x_mask #this is redundant and not necessary (as using masked softmax in self-atten)
        x_temp = F.dropout(x_temp, self.drop_prob, self.training)
        x = residual_layer_dropout(x, x_temp, self.stchstc_dpth_drp_prb * float(pos_lyr_drpout)/tot_sub_layers, 
            self.training)
        x_temp = layers.my_leaky_relu(x_temp)
        pos_lyr_drpout += 1
        # print('ENCODER: AFTER SELF ATTENTION')
        # GPU_Memory_Usage()

        #feed-forward
        x_temp = self.feed_fwd_layer_norm(x)
        x_temp = F.dropout(x_temp, self.drop_prob, self.training)
        x_temp = x_temp * x_mask
        x_temp = self.feed_fwd(x_temp)
        x_temp = layers.my_leaky_relu(x_temp)
        x_temp = x_temp * x_mask
        x_temp = self.feed_fwd(x_temp) #based upon https://github.com/NLPLearn/QANet
        x_temp = x_temp * x_mask
        x_temp = F.dropout(x_temp, self.drop_prob, self.training)
        x = residual_layer_dropout(x, x_temp, self.stchstc_dpth_drp_prb * float(pos_lyr_drpout)/tot_sub_layers, 
            self.training)
        # x_temp = layers.my_leaky_relu(x_temp) #don't do relu as it's the output of the layer
        pos_lyr_drpout += 1
        # print('ENCODER: AFTER FEEDFORWARD')
        # GPU_Memory_Usage()

        return x


class Depthwise_Sep_Conv_Layer(nn.Module):
    '''Implementation of depthwise separable convolution'''
    def __init__(self, kernel_size, num_in_channels, num_out_channels):
        super(Depthwise_Sep_Conv_Layer, self).__init__()
        #stride is always 1
        padding = (kernel_size-1)//2 #to maintain same dimension between input and output assuming stride=1 and kernel size is odd number
        self.separable_layer = torch.nn.Conv1d(num_in_channels, num_in_channels, kernel_size=kernel_size, groups=num_in_channels, padding=padding)
        self.depthwise_layer = torch.nn.Conv1d(num_in_channels, num_out_channels, kernel_size=1)

    def forward(self, x, x_mask):
        #x: batch x max_seq_len x hidden_size
        #x_mask: batch x max_seq_len x 1 (0 where it is padded)

        x = x.permute(0,2,1) #b'cse conv layer input is Batch x channel x input_size
        x_mask = x_mask.permute(0,2,1)

        x = self.separable_layer(x) * x_mask
        x = self.depthwise_layer(x) * x_mask
        x = x.permute(0,2,1)

        return x



class PositionalEncoding(nn.Module):
    #this based upon https://nlp.seas.harvard.edu/2018/04/03/attention.html
    "Implement the PE function."
    def __init__(self, hidden_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, hidden_size) #max_len x hidden_size
        position = torch.arange(0, max_len).unsqueeze(1) #max_len x 1
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(torch.log(torch.tensor(10000.0)) / hidden_size)) #1 x hidden_size/2
        pe[:, 0::2] = torch.sin(position * div_term) #RHS is max_len x hidden_size/2
        pe[:, 1::2] = torch.cos(position * div_term) #RHS is max_len x hidden_size/2
        pe = pe.unsqueeze(0) #1 x max_len x hidden_size
        self.register_buffer('pe', pe)  #put in register_buffer so it can be saved as well as moved to gpu/cpu easily. Not trainable. Will be saved as self.var_name

    def forward(self, x):
        '''
        x (float32 tensor of shape batch_size x max_seq_len_in_batch x hidden_size): padded input (with pad token being 0)
        return output (float32 tensor of shape batch_size x max_seq_len_in_batch x embed_dim): 
        padded token will have a zero vector representing it. 
        Note that pos=0 has the following vectr representing it [0,1,0,1,0,1...]
        '''
        temp = torch.zeros(x.shape[:2]).unsqueeze(-1).to(device=self.pe.device, dtype=self.pe.dtype)
        #not sure if really need to clone and detach
        y = temp + self.pe[:, :x.size(1), :].clone().detach() #batch x max_seq_len_in_batch x hidden_size
        return y


class MultiHeadAtten_Pytorch(nn.Module):
    def __init__(self, hidden_size, num_atten_heads, drop_prob, bias):
        '''
            hidden_size (int): hidden (input/output) dimension
            num_atten_heads (int): number of layers for multi-head attention
            return None
        '''
        super(MultiHeadAtten_Pytorch, self).__init__()
        self.mha = torch.nn.MultiheadAttention(hidden_size, num_atten_heads, drop_prob, bias=bias)
    
    def forward(self, x, mask):
        '''
        x (float32 tensor of shape batch_size x seq_len x hidden_size): input
        mask (Bool Tensor of shape batch_size x key_seq_len x 1): mask showing padded tokens in the input (0 where it is padded)
        returns (float32 tensor of shape batch_size x seq_len x hidden_size)
        '''
        mask_qry = mask #key and query masks are same for self-attention
        mask_temp = mask.logical_not().squeeze(2)
        x = x.permute(1,0,2)
        out, _ = self.mha(x, x, x, key_padding_mask=mask_temp)
        out = out.permute(1,0,2)
        out = out * mask_qry #to zero out for all the padded tokens in the query
        return out


class MultiHeadAtten(nn.Module):
    def __init__(self, hidden_size, num_atten_heads, drop_prob, bias):
        '''
        hidden_size (int): hidden (input/output) dimension
        num_atten_heads (int): number of layers for multi-head attention
        return None
        '''

        assert False, 'Need to properly implement dropout'

        super(MultiHeadAtten, self).__init__()
        assert hidden_size%num_atten_heads == 0, 'The number of layers desired for Multi-head self-attention is not correct!'

        self.hidden_size = hidden_size
        self.num_atten_heads = num_atten_heads
        layers_list = []
        for i in range(num_atten_heads):
            layer = self.Attention(hidden_size, hidden_size//num_atten_heads, bias=bias)
            layers_list.append(layer)
        self.layers = torch.nn.ModuleList(layers_list)
        self.out_prj = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
    
    def forward(self, x, mask):
        '''
        x (float32 tensor of shape batch_size x seq_len x hidden_size): input
        mask (Bool Tensor of shape batch_size x seq_len): mask showing padded tokens in the input (0 where it is padded)
        returns (float32 tensor of shape batch_size x seq_len x hidden_size)
        '''
        output = []
        for l in self.layers:
            output.append(l(x, mask))
        output = torch.cat(output, dim=-1)
        output = self.out_prj(output)
        return output


    class Attention(nn.Module):
        def __init__(self, inpt_dim, atten_head_hidden_dim, bias):
            '''
            Self-Attention Layer used inside multi-head attention
            inpt_dim (int): input dimension (the overall hidden dimension)
            atten_head_hidden_dim (int): hidden dimension of each attention head in multi-head attention
            return None
            '''
            super(MultiHeadAtten.Attention, self).__init__()
            self.atten_head_hidden_dim = atten_head_hidden_dim
            self.key_prj = torch.nn.Linear(inpt_dim, atten_head_hidden_dim, bias=bias)
            self.val_prj = torch.nn.Linear(inpt_dim, atten_head_hidden_dim, bias=bias) #should this be same as key_prj, it's not in the Transformers paper
            self.qry_prj = torch.nn.Linear(inpt_dim, atten_head_hidden_dim, bias=bias)       
        
        def forward(self, x, mask):
            '''
            x (int tensor of shape batch_size x seq_len x inpt_dim): input
            mask (Bool Tensor of shape batch_size x seq_len x 1): mask showing padded tokens in the input (0 where it is padded)
            returns (float32 tensor of shape batch_size x seq_len x atten_head_hidden_dim)

            uses scaled dot product for similarity matrix attention.
            '''
            def apply_mask_and_softmax(sim_mtx, mask):
                '''
                sim_mtx (tensor of shape batchsize x qry_seq_len x seq_len)
                mask (Bool Tensor of shape batch_size x seq_len): mask showing padded tokens in the input
                '''

                key_mask = mask.permute(0,2,1) #(Tensor of shape batch_size x 1 x seq_len)
                qry_mask = mask #(Tensor of shape batch_size x qry_seq_len x 1)
    
                sim_mtx += key_mask.logical_not()*-1e10 #inplace operation on sim_mtx
                sim_mtx = F.softmax(sim_mtx, dim=-1) #(batch_size x qry_seq_len x seq_len). To covert into probabilities using softmax

                sim_mtx = sim_mtx * qry_mask #not inplace operation on sim_mtx (need to do this b'cse of how softmax function is implemented)
                return sim_mtx


            #transform key,val,qry to a different representations where they can be compared (as well as to a lower dimension for multi-head attention)
            key_p = self.key_prj(x) #batch_size x seq_len x atten_head_hidden_dim
            val_p = self.val_prj(x) #batch_size x seq_len x atten_head_hidden_dim
            qry_p = self.qry_prj(x) #batch_size x q_seq_len x atten_head_hidden_dim

            #dot product based similarity matrix (batch_size x qry_seq_len x seq_len)
            # sim_mtx = torch.zeros(key_p.shape[0], key_p.shape[1], key_p.shape[1]).cuda()
            #matmul between (batch_size x seq_len x atten_head_hidden_dim) by (batch_size x qry_seq_len x atten_head_hidden_dim) => (batch_size x qry_seq_len x seq_len)
            sim_mtx = torch.rsqrt(torch.tensor(self.atten_head_hidden_dim, dtype=torch.float32)) * torch.matmul(qry_p, key_p.permute(0,2,1)) #(batch_size x qry_seq_len x seq_len)
            sim_mtx = apply_mask_and_softmax(sim_mtx, mask)

            #matmul between (batch_size x qry_seq_len x seq_len) by (batch_size x seq_len x atten_head_hidden_dim) 
            output = torch.matmul(sim_mtx, val_p) #(batch_size x qry_seq_len x atten_head_hidden_dim)
            return output




class QANetOutput(nn.Module):
    """Output layer used by QANet for question answering.

    Computes a linear transformation of the attention and modeling
    outputs (M0,M1), then takes the softmax of the result to get the start pointer.
    
    A second linear+softmax of the attention output (M0,M2) is used to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the QANet model.
    """
    def __init__(self, hidden_size, bias):
        super(QANetOutput, self).__init__()
        self.att_linear_1 = nn.Linear(2 * hidden_size, 1, bias=bias)
        self.att_linear_2 = nn.Linear(2 * hidden_size, 1, bias=bias)


    def forward(self, M0, M1, M2, mask):
        '''
        M0 (batch_size, c_len, hidden_size): output of attention layer
        M1 (batch_size, c_len, hidden_size): output of attention layer
        M2 (batch_size, c_len, hidden_size): output of attention layer
        mask: (batch_size, c_len) (0 where it is padded)
        '''
        logits_1 = self.att_linear_1(torch.cat([M0,M1], dim=-1)) # (batch_size, c_len, 1)
        logits_2 = self.att_linear_2(torch.cat([M0,M2], dim=-1)) # (batch_size, c_len, 1)

        log_p1 = util.masked_softmax(logits_1.squeeze(), mask, log_softmax=True) #(batch_size, c_len)
        log_p2 = util.masked_softmax(logits_2.squeeze(), mask, log_softmax=True) #(batch_size, c_len)

        return log_p1, log_p2



import BiDAF
import gc
def test_QANet():
    vocab_size = int(1e4)
    char_vocab_size = 100
    device = util.get_available_devices()[0]
    word_vocab = torch.rand(vocab_size, 300) #model.to() will send it to right device

    #generate data
    batch, sen_len, word_len = 32, 400, 30, #batch and seq_len are the ciritcal one
    cw = torch.randint(0, vocab_size, (batch, sen_len)).to(device)
    cc = torch.randint(0, char_vocab_size, (batch, sen_len, word_len)).to(device)
    sen_len = 50
    qw = torch.randint(0, vocab_size, (batch, sen_len)).to(device)
    qc = torch.randint(0, char_vocab_size, (batch, sen_len, word_len)).to(device)

    # model = BiDAF.BiDAF_with_char_embed(word_vocab, hidden_size=200)
    model = QANet(word_vocab, hidden_size=128, num_mod_enc_blocks=7)
    model = model.to(device)
    # util.get_model_info(model)
    # model = torch.jit.script(model) #gives error
    # model = torch.jit.trace(model, (cw, cc, qw, qc)) #gives error


    # hidden_size = 128
    # x = torch.rand(batch, sen_len, 4*hidden_size).to(device)
    # x_mask = torch.randint(0,2, (batch, sen_len)).to(device)
    # model_enc = QANet_Transformer(hidden_size=4*hidden_size, kernel=5, num_conv_layers=2, 
    #                         num_enc_blocks=7, drop_prob=0.0, num_atten_heads=8).to(device)

    def profile1():
        #memory usage profiler (very primitive)
        # with torch.no_grad():
        # torch.autograd.set_detect_anomaly(True)
        for e in range(2):            
            print(f'EPOCH: {e}')
            pred = model(cw, cc, qw, qc)
            loss = pred[0].mean() + pred[1].mean()
            # print('AFTER LOSS COMPUTATION')
            # GPU_Memory_Usage()

            loss.backward() #clears out the buffers after each epoch

            print('AFTER LOSS.BACKWARD')
            GPU_Memory_Usage()

            #print(e, pred[0].shape, pred[1].shape)
            # model_enc(x, x_mask)
            print('***************')


    def profile2():
        #Profiling using cProfile
        print("Profiling QANet's forward call")
        import cProfile
        pr = cProfile.Profile()
        pr.enable()
        model(cw, cc, qw, qc)
        pr.disable()
        pr.print_stats(sort='time')
    

    def profile3():
        #cpu/gpu usage profiler
        with torch.autograd.profiler.profile() as prof:
            for e in range(1):            
                pred = model(cw, cc, qw, qc)
                loss = pred[0].mean() + pred[1].mean()
                # loss.backward() #clears out the buffers after each epoch
        print(prof.key_averages().table(sort_by="self_cpu_time_total"))


    def profile4():
        #cpu/gpu memory usage profiler (available after Pytorch version 1.6)
        with torch.autograd.profiler.profile(profile_memory=True, record_shapes=True) as prof:
            for e in range(1):            
                pred = model(cw, cc, qw, qc)
                loss = pred[0].mean() + pred[1].mean()
                # loss.backward() #clears out the buffers after each epoch
        print(prof.key_averages().table(sort_by="self_cpu_memory_usage"))


    def profile5():
        # prints currently alive Tensors and Variables (for memory usage debug)
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    print(type(obj), obj.size())
            except:
                pass

    profile4()



#!pip install nvidia-ml-py3
import nvidia_smi
def GPU_Memory_Usage():
    # print('nvidia-smi:')
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate
    mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print(f'memory used: {mem_res.used / (1024**3)} (GiB) i.e. {100 * (mem_res.used / mem_res.total):.3f}%') # usage in GiB/percentage

    # print('Pytorch memory info:')
    # print(f'memory_allocated: {torch.cuda.memory_allocated(device=None)/1e6}MB')
    # print(f'memory_cached: {torch.cuda.memory_cached(device=None)/1e6}MB')


if __name__ == '__main__':
    test_QANet()
    # GPU_Memory_Usage()


