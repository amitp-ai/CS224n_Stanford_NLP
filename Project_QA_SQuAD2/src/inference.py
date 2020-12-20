#added by Amit for inference
import time
import torch.utils.data as data
import numpy as np
import util
import BiDAF
import torch
from ujson import load as json_load
import torch.nn.functional as F
from collections import OrderedDict
import tqdm

"""
Author:
    Amit Patel (amitpatel.gt@gmail.com)
"""

'''
NOTE: ALL THE PATHS ARE REFERENCED FROM THE DIRECTORY WHERE THE COLAB NOTEBOOK IS SAVED BCSE OS.CHDIR() \
IS MADE TO POINT THERE AND ALL .py FILES INSIDE SRC DIRECTORY ARE ONLY CALLED FROM THE COLAB NOTEBOOK AND ALSO BCSE \
THE PATH TO ALL THE SRC/*.PY FILES IS ADDED TO SYS.PATH.
'''


#Declare Important Global Variables
#generate idx2word_dict (used for inference)
word2idx_file = './data/word2idx.json'
with open(word2idx_file, 'r') as fh:
    word2idx_dict = json_load(fh)
idx2word_dict = {}
for k,v in word2idx_dict.items():
    idx2word_dict[v] = k

#load char2idx_dict
char2idx_file = './data/char2idx.json'
with open(char2idx_file, 'r') as fh:
    char2idx_dict = json_load(fh)


def print_eval_results(pred_dict, eval_dict, num_visuals=5, window_size = 18):
    '''
    To print a few examples.
    '''
    print('Printing Eval')
    if num_visuals <= 0:
        return
    if num_visuals > len(pred_dict):
        num_visuals = len(pred_dict)

    visual_ids = np.random.choice(list(pred_dict), size=num_visuals, replace=False)
    # visual_ids = list(pred_dict)[0:num_visuals]

    for i, id_ in enumerate(visual_ids):
        pred = pred_dict[id_] or 'N/A' #if pred_dict[id_] is None then select 'N/A'

        example = eval_dict[str(id_)]
        question = example['question']
        context = example['context']
        answers = example['answers']
        gold = answers[0] if answers else 'N/A'

        #to print context without having to scroll horizontally
        context = context.split(' ')
        pretty_context = []
        for i in range(len(context)//window_size+1):
            pretty_context += (context[i*window_size:(i+1)*window_size]+['\n'])
        pretty_context[-1] = '' #remove last '\n'
        context = ' '.join(pretty_context)

        tbl_fmt = (f'- **Question:** {question}\n'
                   + f'- **Context:** {context}\n'
                   + f'- **Answer:** {gold}\n'
                   + f'- **Prediction:** {pred}')
        print(tbl_fmt)
        print()
        time.sleep(0.05) #to not exceed the output display rate for large num_visuals


def print_inference(inputs, preds, gt_ans_, window_size = 18):
    print('\nPrinting Inference')
    ids, qw_idxs, cw_idxs = inputs
    starts, ends = preds

    for ii in range(ids.size(0)):
        #pred
        strt_idx = starts[ii]
        end_idx = ends[ii]
        pred = []
        for iii in range(strt_idx, end_idx+1):
            word_id = cw_idxs[ii,iii].item()
            if word_id == 1: #if __oov__ token
                pred.append('N/A')
            else:
                pred.append(idx2word_dict[word_id])

        #gt_answer
        if gt_ans_ is not None:
            y1, y2 = gt_ans_
            strt_idx = y1[ii]
            end_idx = y2[ii]
            gt_ans = []
            for iii in range(strt_idx, end_idx+1):
                word_id = cw_idxs[ii,iii].item()
                if word_id == 1: #if _oov__ token
                    gt_ans.append('N/A')
                else:
                    gt_ans.append(idx2word_dict[word_id])
        else:
            gt_ans = ['Ground truth answer not available']

        #question
        ques = []
        for iii in range(1, qw_idxs.size(1)): #ignore first word as its __oov__ (each sequence is prepended with __oov__ for 'no answer' capability)
            word_id = qw_idxs[ii,iii].item()
            if word_id is not 0: #ignore pad token
                ques.append(idx2word_dict[word_id])

        #context
        context = []
        for iii in range(1, cw_idxs.size(1)): #ignore first word as its __oov__ (each sequence is prepended with __oov__ for 'no answer' capability)
            word_id = cw_idxs[ii,iii].item()
            if word_id is not 0: #ignore pad token
                context.append(idx2word_dict[word_id])
        #to print context without having to scroll horizontally
        pretty_context = []
        for i in range(len(context)//window_size+1):
            pretty_context += (context[i*window_size:(i+1)*window_size]+['\n'])
        pretty_context[-1] = '' #remove last '\n'
        context = ' '.join(pretty_context)

        #print
        print('**Context:** ' + context)
        print('**Question:** ' + ' '.join(ques))
        print('**GT Answer:** ' + ' '.join(gt_ans))
        print('**Prediction:** ' + ' '.join(pred))
        print()
        time.sleep(0.05) #to not exceed the output display rate for long list of ids


class SQuAD_inference(util.SQuAD):
    '''Dataset for inference'''

    def __init__(self, dataset, use_v2=True):
        data.Dataset.__init__(self) #instead of super(SQuAD_inference, self).__init__()

        self.context_idxs = dataset[0].long()
        self.context_char_idxs = dataset[1].long()
        self.question_idxs = dataset[2].long()
        self.question_char_idxs = dataset[3].long()
        self.y1s = dataset[4].long()
        self.y2s = dataset[5].long()

        if use_v2:
            # SQuAD 2.0: Use index 0 for no-answer token (token 1 = OOV)
            batch_size, c_len, w_len = self.context_char_idxs.size()
            ones = torch.ones((batch_size, 1), dtype=torch.int64)
            self.context_idxs = torch.cat((ones, self.context_idxs), dim=1)
            self.question_idxs = torch.cat((ones, self.question_idxs), dim=1)

            ones = torch.ones((batch_size, 1, w_len), dtype=torch.int64)
            self.context_char_idxs = torch.cat((ones, self.context_char_idxs), dim=1)
            self.question_char_idxs = torch.cat((ones, self.question_char_idxs), dim=1)

            self.y1s += 1
            self.y2s += 1

        # SQuAD 1.1: Ignore no-answer examples
        self.ids = dataset[6].long()
        self.valid_idxs = [idx for idx in range(len(self.ids))
                           if use_v2 or self.y1s[idx].item() >= 0]



def get_word_and_char_idxs(inputs):
    def get_word_idxs(inputs):
        '''
        convert string inputs to padded torch ints using word2idx_dict
        inputs <list[str]>: list of strings. where the len of the list is the batch size
        The first word is made to be --OOV-- with token = 1 to answer N/A in the SQUAD dataset class so don't do it here.
        pad token: 0 and --OOV-- token: 1
        '''
        outputs = [i.split(' ') for i in inputs]
        w_idxs = [torch.tensor([word2idx_dict[w] if w in word2idx_dict else word2idx_dict['--OOV--'] for w in c]) for c in outputs]
        w_idxs = torch.nn.utils.rnn.pad_sequence(w_idxs, batch_first=True) #size: batch x seq_len
        return w_idxs


    def get_char_idxs(inputs, max_word_len=16):
        '''
        padding token is 0
        The first word is made to be --OOV-- with token = 1 to answer N/A in the SQUAD dataset class so don't do it here.
        pad token: 0 and --OOV-- token: 1
        '''
        outputs = [i.split(' ') for i in inputs]
        c_idxs = []
        for e in outputs:
            exmp = []
            for w in e:
                word = []
                for c in w:
                    if c in char2idx_dict:
                        word.append(char2idx_dict[c])
                    else:
                        word.append(1) #--OOV--
                exmp.append(torch.tensor(word))
            exmp = torch.nn.utils.rnn.pad_sequence(exmp, batch_first=True)
            temp = torch.zeros(exmp.shape[0], max_word_len) #padding the rest with 0
            temp[0:exmp.shape[0], 0:exmp.shape[1]] = exmp
            c_idxs.append(temp)
        c_idxs = torch.nn.utils.rnn.pad_sequence(c_idxs, batch_first=True)
        return c_idxs
    
    return get_word_idxs(inputs), get_char_idxs(inputs)


def inference(model, dataset, data_loader, device, max_ans_len, use_squad_v2=True):    
    with torch.no_grad(), tqdm.tqdm(total=len(dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            # cc_idxs and qc_idxs are character level representations (with max char len of 16)
            # print(cw_idxs.shape, cc_idxs.shape)
            # print(qw_idxs.shape, qc_idxs.shape)
            # print(qw_idxs[0])
            # print(qc_idxs[0])

            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            batch_size = cw_idxs.size(0)

            # Forward
            #log_p1, log_p2 = model(cw_idxs, qw_idxs) #without char_embeddings
            log_p1, log_p2 = model(cw_idxs, cc_idxs, qw_idxs, qc_idxs) #with char_embeddings
            y1, y2 = y1.to(device), y2.to(device)
            loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
            nll_meter.update(loss.item(), batch_size)

            # Get F1 and EM scores
            p1, p2 = log_p1.exp(), log_p2.exp()
            starts, ends = util.discretize(p1, p2, max_ans_len, use_squad_v2) #from util.py

            #for debug only
            #print(p1[10].max(), p1[10].argmax(), p2[10].max(), p2[10].argmax()) #for debug only
            #print(nll_meter.avg) #for debug only
            #print(qw_idxs.shape)
            
            #idx2pred, _ = util.convert_tokens(gold_dict, ids.tolist(), starts.tolist(), ends.tolist(), use_squad_v2) #from util.py
            #pred_dict.update(idx2pred)
            
            inputs = (ids, qw_idxs, cw_idxs)
            preds = (starts, ends)
            gt_ans = (y1, y2)
            print_inference(inputs, preds, None)
            #print_eval_results(pred_dict=idx2pred, eval_dict=gold_dict, num_visuals=5)


            progress_bar.update(batch_size)
            continue

            # print results (except for test set, since it does not come with labels)
            if have_labels:
                results = util.eval_dicts(gold_dict, pred_dict, use_squad_v2) #from util.py
                results_list = [('NLL', nll_meter.avg),
                                ('F1', results['F1']),
                                ('EM', results['EM'])]
                if use_squad_v2:
                    results_list.append(('AvNA', results['AvNA']))
                results = OrderedDict(results_list)

                # print to console
                results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                progress_bar.set_postfix(Results=results_str)

    #print few examples
    #print(pred_dict)
    #print_eval_results(pred_dict=pred_dict, eval_dict=gold_dict, num_visuals=5)


# Evaluate
nll_meter = util.AverageMeter() #from util.py
