import layers
import torch
import util
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import numpy as np
import sys
from functools import reduce
import QANet

"""
Author:
    Amit Patel (amitpatel.gt@gmail.com)
"""

'''
THIS IS BASED UPON QANET, BUT IT USES the ORIGINAL TRANSFORMER ARCHITECTURE.

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


class QANet_xfmr(nn.Module):
    """QANet model for SQuAD2 with CNN based char level embedding.
    Using transformer architecture similar to the original paper.

    Based on the paper:
    QANet (https://arxiv.org/pdf/1804.09541.pdf) and
    Attention is all you need


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
        super(QANet_xfmr, self).__init__()

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
        self.emb = QANet.QANet_Embed_with_charCNN(word_vectors=word_vectors, char_dim=char_dim,
                                    hidden_size=hidden_size, w_drop_prob=self.wrd_drp_prb,
                                    c_drop_prob = self.chr_drp_prb, char_vocab_size=char_vocab_size, char_cnn_kernel=5)

        self.emb_enc = QANet_Orig_Transformer(hidden_size=hidden_size, num_enc_blocks=4, drop_prob=self.drop_prob, 
                            stchstc_dpth_drp_prb = self.stchstc_dpth_drp_prb, num_atten_heads=6, bias=bias)

        self.att = layers.BiDAFAttention(hidden_size=hidden_size, drop_prob=drop_prob)

        self.mod_enc_input_prj = torch.nn.Linear(4*hidden_size, hidden_size)
        self.mod_enc = QANet_Orig_Transformer(hidden_size=hidden_size, num_enc_blocks=num_mod_enc_blocks, drop_prob=self.drop_prob, 
                            stchstc_dpth_drp_prb = self.stchstc_dpth_drp_prb, num_atten_heads=6, bias=bias)

        self.out = QANet.QANetOutput(hidden_size=hidden_size, bias=bias)
        self.reset()


    def reset(self):
        '''
        Initialize with Kaiming initialization
        '''
        # z = list(self.children()) #not recursive
        # print(list(z[0].modules()), print(type(self.lyr_norm1)))
        for m in self.modules(): #self.modules() will recursively return all the modules in the network
            #print(m.weight.shape, type(m), m.bias.shape if m.bias != None else m.bias)
            if type(m) in [torch.nn.modules.linear.Linear, torch.nn.modules.conv.Conv1d]:
                torch.nn.init.kaiming_uniform_(m.weight, a=0.01) #a=2.3
                if m.bias != None: torch.nn.init.zeros_(m.bias)
            elif type(m) == torch.nn.modules.normalization.LayerNorm:
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
            # elif type(m) == torch.nn.modules.activation.MultiheadAttention:
            #     for p_n, p in m.named_parameters():
            #         if 'weight' in p_n:
            #             torch.nn.init.kaiming_uniform_(p, a=1.0) #this hurts performance vs using default initialization


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
        # M1 = F.dropout(M1, self.drop_prob, self.training) #https://github.com/NLPLearn/QANet uses a conv layer to do this   
        # print('AFTER MOD_ENC2')
        # GPU_Memory_Usage()

        M2 = self.mod_enc(M1, c_mask)        # (batch_size, c_len, hidden_size)
        # print('AFTER MOD_ENC3')
        # GPU_Memory_Usage()

        p_strt, p_stp = self.out(M0, M1, M2, c_mask)  # 2 tensors, each (batch_size, c_len)
        # print('AFTER OUTPUT LAYER')
        # GPU_Memory_Usage()

        return (p_strt, p_stp)



class QANet_Orig_Transformer(nn.Module):
    '''
    This is the main innovation of QANet. Instead of RNN, it uses a transformer to get contextual encoding of the question/context embedding
    In this we use the original transformer network instead of whats used in QANet.
    '''
    def __init__(self, hidden_size, num_enc_blocks, drop_prob, stchstc_dpth_drp_prb, num_atten_heads, bias=False):
        super(QANet_Orig_Transformer, self).__init__()

        encoder_blocks_list = []
        for _ in range(num_enc_blocks):
            encoder_block = QANet_xfmr_Encoder(hidden_size, drop_prob, stchstc_dpth_drp_prb, num_atten_heads, bias=bias)
            encoder_blocks_list.append(encoder_block)
        self.encoder_blocks_list = nn.ModuleList(encoder_blocks_list)

    def forward(self, x, x_mask):
        #x: batch_size x max_seq_len x hidden_size
        #x_mask: batch_size x max_seq_len

        tot_numof_blocks = len(self.encoder_blocks_list)
        x_mask = x_mask.unsqueeze(-1)

        #Encoding blocks
        for block_num, encoder in enumerate(self.encoder_blocks_list):
            #encoder
            x = encoder(x, x_mask, block_num, tot_numof_blocks)
        return x #batch_size x max_seq_len x hidden_size


class QANet_xfmr_Encoder(nn.Module):
    """
    This is the key building block
    """
    def __init__(self, hidden_size, drop_prob, stchstc_dpth_drp_prb, num_atten_heads, bias):
        super(QANet_xfmr_Encoder, self).__init__()

        self.drop_prob = drop_prob
        self.stchstc_dpth_drp_prb = stchstc_dpth_drp_prb

        self.pos_enc = QANet.PositionalEncoding(hidden_size, max_len=int(1e4))

        self.self_atten_layer_norm = nn.LayerNorm(hidden_size) #do layer norm only along the last dimension which is hidden_size
        self.self_attention = QANet.MultiHeadAtten_Pytorch(hidden_size, num_atten_heads, drop_prob, bias=bias)

        self.feed_fwd_layer_norm = nn.LayerNorm(hidden_size) #do layer norm only along the last dimension which is hidden_size
        self.feed_fwd1 = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.feed_fwd2 = torch.nn.Linear(hidden_size, hidden_size, bias=bias)


    def forward(self, x, x_mask, block_num, tot_numof_blocks):
        #x: batch_size x max_seq_len x input_size
        #x_mask: batch_size x max_seq_len (0 where it is padded)
        #block_num: block number (used with residual_layer_dropout())
        #tot_numof_blocks: total number of blocks in this module (used with residual_layer_dropout())

        tot_sub_layers = 2*tot_numof_blocks
        pos_lyr_drpout = 2*block_num

        #positional encoding
        x = x + self.pos_enc(x)
        x = x * x_mask
        
        #self-attention
        x_temp = self.self_attention(x, x_mask)
        x_temp = x_temp * x_mask #this is redundant and not necessary (as using masked softmax in self-atten)
        x_temp = F.dropout(x_temp, self.drop_prob, self.training)
        x_temp = QANet.residual_layer_dropout(x, x_temp, self.stchstc_dpth_drp_prb * float(pos_lyr_drpout)/tot_sub_layers, self.training)
        x_temp = self.self_atten_layer_norm(x_temp)
        x = x_temp * x_mask #this is redundant and not necessary (as using masked softmax in self-atten)
        pos_lyr_drpout += 1
        # print('ENCODER: AFTER SELF ATTENTION')
        # GPU_Memory_Usage()

        #feed-forward
        x_temp = self.feed_fwd1(x)
        x_temp = x_temp * x_mask
        x_temp = layers.my_leaky_relu(x_temp)
        x_temp = self.feed_fwd2(x_temp)
        x_temp = x_temp * x_mask
        x_temp = F.dropout(x_temp, self.drop_prob, self.training)
        x_temp = QANet.residual_layer_dropout(x, x_temp, self.stchstc_dpth_drp_prb * float(pos_lyr_drpout)/tot_sub_layers, self.training)
        x_temp = self.feed_fwd_layer_norm(x_temp)
        x = x_temp * x_mask
        pos_lyr_drpout += 1
        # print('ENCODER: AFTER FEEDFORWARD')
        # GPU_Memory_Usage()

        return x


import gc
import BiDAF

#this decorator is for use with line_profiler
# @profile
def test_QANet_xfmr():
    vocab_size = int(1e4)
    char_vocab_size = 100
    device = util.get_available_devices()[0]
    print(device)
    word_vocab = torch.rand(vocab_size, 300) #model.to() will send it to right device

    # model = BiDAF.BiDAF_with_char_embed(word_vocab, hidden_size=200)
    model = QANet_xfmr(word_vocab, hidden_size=126)
    model = model.to(device)
    # util.get_model_info(model)

    #generate data
    batch, sen_len, word_len = 32, 400, 30, #batch and seq_len are the ciritcal one
    cw = torch.randint(0, vocab_size, (batch, sen_len)).to(device)
    cc = torch.randint(0, char_vocab_size, (batch, sen_len, word_len)).to(device)
    sen_len = 50
    qw = torch.randint(0, vocab_size, (batch, sen_len)).to(device)
    qc = torch.randint(0, char_vocab_size, (batch, sen_len, word_len)).to(device)

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
if __name__ == '__main__':
    test_QANet_xfmr()
    # GPU_Memory_Usage()


