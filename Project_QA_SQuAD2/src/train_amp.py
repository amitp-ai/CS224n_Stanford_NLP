#train.py
"""Train a model on SQuAD.

Author:
    Chris Chute (chute@stanford.edu)
    Amit Patel (amitpatel.gt@gmail.com)
"""

'''
NOTE: ALL THE PATHS ARE REFERENCED FROM THE DIRECTORY WHERE THE COLAB NOTEBOOK IS SAVED BCSE OS.CHDIR() \
IS MADE TO POINT THERE AND ALL .py FILES INSIDE SRC DIRECTORY ARE ONLY CALLED FROM THE COLAB NOTEBOOK AND ALSO BCSE \
THE PATH TO ALL THE SRC/*.PY FILES IS ADDED TO SYS.PATH.
'''

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data

from collections import OrderedDict
from json import dumps
from torch.utils.tensorboard import SummaryWriter
import tqdm
from ujson import load as json_load

import util


def main(args, model_type):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True) #from util.py
    log = util.get_logger(args.save_dir, args.name) #from util.py
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = util.get_available_devices() #from util.py
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = util.torch_from_json(args.word_emb_file) #from util.py

    # Get model
    log.info('Building model...')
    model = model_type(word_vectors=word_vectors, hidden_size=args.hidden_size, drop_prob=args.drop_prob)
    model = nn.DataParallel(model, args.gpu_ids)
    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = util.load_model(model, args.load_path, args.gpu_ids) #from util.py
    else:
        step = 0
    model = model.to(device)
    model.train()

    ema = util.EMA(model, args.ema_decay) #from util.py

    # Get saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log) #from util.py

    # Get optimizer and scheduler
    # optimizer = optim.Adadelta(params=model.parameters(), lr=args.lr, weight_decay=args.l2_wd)
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.l2_wd)

    scheduler = sched.LambdaLR(optimizer, lambda e: 1.0/min(max(2, e/15*2)-1, 2)) #start reducing lr after 15 epochs then reduce by a maximum of 2x

    # Get data loader
    log.info('Building dataset...')
    train_dataset = util.SQuAD(args.train_record_file, args.use_squad_v2) #from util.py
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   collate_fn=util.collate_fn)  #collate_fn from util.py
    dev_dataset = util.SQuAD(args.dev_record_file, args.use_squad_v2) #from util.py
    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn=util.collate_fn)  #collate_fn from util.py
    
    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    epoch = step // len(train_dataset)

    # Creates a GradScaler once at the beginning of training
    scaler = torch.cuda.amp.GradScaler()
    # util.cast_model_to(model, torch.float32)
    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch} out of {args.num_epochs} epochs...')
        with torch.enable_grad(), tqdm.tqdm(total=len(train_loader.dataset)) as progress_bar:
            for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in train_loader:
                # cc_idxs and qc_idxs are character level representations (with max char len of 16)
                '''
                Input dimensions:
                Xw_**** is word level representation
                Xc_**** is char level representation of each word
                cw_idxs: (batch, max_contx_seqlen_in_batch).
                cc_idxs: (batch, max_contx_seqlen_in_batch, 16). 16 is the max numbers of characers allowed in a word.
                qw_idxs: (batch, max_questn_seqlen_in_batch)
                qc_idxs: (batch, max_questn_seqlen_in_batch, 16). 16 is the max numbers of characers allowed in a word.
                y1: (batch,)
                y2: (batch,)
                ids: (batch,)
                '''
                
                # Setup for forward
                cw_idxs = cw_idxs.to(device)
                qw_idxs = qw_idxs.to(device)
                y1, y2 = y1.to(device), y2.to(device)
                batch_size = cw_idxs.size(0)
                optimizer.zero_grad()

                '''
                # Runs the forward pass with autocasting.
                with torch.cuda.amp.autocast():
                    # Forward
                    # log_p1, log_p2 = model(cw_idxs, qw_idxs) #without char_embeddings
                    log_p1, log_p2 = model(cw_idxs, cc_idxs, qw_idxs, qc_idxs) #with char_embeddings
                    loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2) #hard labelling loss
                    # loss = soft_labelling_loss(log_p1, y1, cw_idxs, device) + soft_labelling_loss(log_p2, y2, cw_idxs, device) #soft labelling loss


                # Backward
                loss_val = loss.item()

                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                # Backward passes under autocast are not recommended.
                # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                scaler.scale(loss).backward(allow_unreachable=True)

                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                # nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)

                # Updates the scale for next iteration.
                scaler.update()
                
                
                '''
                torch.autograd.set_detect_anomaly(True)
                with torch.cuda.amp.autocast():
                    log_p1, log_p2 = model(cw_idxs, cc_idxs, qw_idxs, qc_idxs) #with char_embeddings
                    loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2) #hard labelling loss
                
                loss_val = loss.item()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                

                # print('DEBUG')
                # QANet.GPU_Memory_Usage()


                ema(model, step // batch_size)
                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         NLL=loss_val)
                tbx.add_scalar('train/NLL', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info(f'Evaluating at step {step}...')
                    ema.assign(model)
                    results, pred_dict = evaluate(model, dev_loader, device,
                                                  args.dev_eval_file,
                                                  args.max_ans_len,
                                                  args.use_squad_v2)
                    saver.save(step, model, results[args.metric_name], device)
                    ema.resume(model)

                    # Log to console
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Dev {results_str}')

                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar(f'dev/{k}', v, step)
                    util.visualize(tbx,
                                pred_dict=pred_dict,
                                eval_path=args.dev_eval_file,
                                step=step,
                                split='dev',
                                num_visuals=args.num_visuals)
            # print(f"\nlr is {optimizer.state_dict()['param_groups'][0]['lr']}") #print out the lr
            # scheduler.step() #call lr scheduler every epoch


def evaluate(model, data_loader, device, eval_file, max_len, use_squad_v2):
    nll_meter = util.AverageMeter() #util.py

    model.eval()
    pred_dict = {}
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm.tqdm_notebook(total=len(data_loader.dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            # cc_idxs and qc_idxs are not used for anything
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            batch_size = cw_idxs.size(0)

            # Forward
            # log_p1, log_p2 = model(cw_idxs, qw_idxs) #without char_embeddings
            log_p1, log_p2 = model(cw_idxs, cc_idxs, qw_idxs, qc_idxs) #with char_embeddings
            y1, y2 = y1.to(device), y2.to(device)
            loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
            nll_meter.update(loss.item(), batch_size)

            # Get F1 and EM scores
            p1, p2 = log_p1.exp(), log_p2.exp()
            starts, ends = util.discretize(p1, p2, max_len, use_squad_v2) #util.py

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(NLL=nll_meter.avg)

            preds, _ = util.convert_tokens(gold_dict,
                                           ids.tolist(),
                                           starts.tolist(),
                                           ends.tolist(),
                                           use_squad_v2)
            pred_dict.update(preds)

    model.train()

    results = util.eval_dicts(gold_dict, pred_dict, use_squad_v2) #util.py
    results_list = [('NLL', nll_meter.avg),
                    ('F1', results['F1']),
                    ('EM', results['EM'])]
    if use_squad_v2:
        results_list.append(('AvNA', results['AvNA']))
    results = OrderedDict(results_list)

    return results, pred_dict



def soft_labelling_loss(log_p, y, cw_idxs, device):
    '''
    log_p shape (B,C)
    y shape (B,)
    
    for soft labelling, y_soft = (1-eps)*y + eps*1/C
    Use KLDiv loss as it is similar to the cross entropy loss within a constant
    References: 
    1. https://towardsdatascience.com/label-smoothing-making-model-robust-to-incorrect-labels-2fae037ffbd0
    2. https://arxiv.org/pdf/1906.02629.pdf
    '''

    eps = 0.2
    c_mask = torch.zeros_like(cw_idxs) != cw_idxs #(batch_size, max_c_len)
    c_len = c_mask.sum(dim=-1, keepdim=True) #(batch_size, 1)


    B,C = log_p.shape
    y_soft = torch.ones(B,C).to(device) #(B,C)
    y_soft = y_soft * eps / c_len #y=0 case
    y_soft[torch.arange(B), y] = ((1-eps) + torch.div(eps,c_len)).squeeze() #y=1 case
    y_soft *= c_mask #handle masking
    #dumy = y_soft.element_size() * y_soft.nelement() / 1e6

    loss = (-y_soft * log_p).sum() / B #soft cross entropy loss
    #loss = F.kl_div(log_p, y_soft, reduction='batchmean') #KL div loss. won't speed up as generating y_soft is the bottle neck
    #loss = F.nll_loss(log_p, y) #hard cross entropy loss (same as soft with eps=0)

    return loss


#################################################
import args
import BiDAF
import QANet
import QANet_xfmr_based

if __name__ == '__main__':
    train_args = args.get_train_args()  #from args.py
    # print(train_args)
    # print(vars(train_args))

    # model_type = eval(args.model_type)
    model_dict = {'QANet.QANet': QANet.QANet, 'QANet_xfmr_based.QANet_xfmr': QANet_xfmr_based.QANet_xfmr, \
    'BiDAF.BiDAF': BiDAF.BiDAF, 'BiDAF.BiDAF_with_char_embed': BiDAF.BiDAF_with_char_embed}
    model_type = model_dict[train_args.model_type]

    main(train_args, model_type)
#################################################
