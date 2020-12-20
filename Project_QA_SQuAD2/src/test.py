#test.py
"""Test a model and generate submission CSV.

Usage:
    > python test.py --split SPLIT --load_path PATH --name NAME
    where
    > SPLIT is either "dev" or "test"
    > PATH is a path to a checkpoint (e.g., save/train/model-01/best.pth.tar)
    > NAME is a name to identify the test run

Author:
    Chris Chute (chute@stanford.edu)
    Amit Patel (amitpatel.gt@gmail.com)
"""

'''
NOTE: ALL THE PATHS ARE REFERENCED FROM THE DIRECTORY WHERE THE COLAB NOTEBOOK IS SAVED BCSE OS.CHDIR() \
IS MADE TO POINT THERE AND ALL .py FILES INSIDE SRC DIRECTORY ARE ONLY CALLED FROM THE COLAB NOTEBOOK AND ALSO BCSE \
THE PATH TO ALL THE SRC/*.PY FILES IS ADDED TO SYS.PATH.
'''

import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from collections import OrderedDict
from json import dumps
from os.path import join
import tqdm
from ujson import load as json_load
from torch.utils.tensorboard import SummaryWriter

import util


def main(args, model_type):
    # Set up logging
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=False) #from util.py
    log = util.get_logger(args.save_dir, args.name) #from util.py
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    device, gpu_ids = util.get_available_devices() #from util.py
    args.batch_size *= max(1, len(gpu_ids))

    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = util.torch_from_json(args.word_emb_file) #from util.py

    # Get model
    log.info('Building model...')
    model = model_type(word_vectors=word_vectors, hidden_size=args.hidden_size)
    model = nn.DataParallel(model, gpu_ids)
    log.info(f'Loading checkpoint from {args.load_path}...')
    model = util.load_model(model, args.load_path, gpu_ids, return_step=False) #from util.py
    model = model.to(device)
    model.eval()

    # Get data loader
    log.info('Building dataset...')
    record_file = vars(args)[f'{args.split}_record_file']
    dataset = util.SQuAD(record_file, args.use_squad_v2) #from util.py
    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  collate_fn=util.collate_fn)  #collate_fn from util.py

    # Evaluate
    log.info(f'Evaluating on {args.split} split...')
    nll_meter = util.AverageMeter() #from util.py
    pred_dict = {}  # Predictions for TensorBoard
    sub_dict = {}   # Predictions for submission
    eval_file = vars(args)[f'{args.split}_eval_file']
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)

    with torch.no_grad(), tqdm.tqdm(total=len(dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            # cc_idxs and qc_idxs are character level representations (with max char len of 16)
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
            starts, ends = util.discretize(p1, p2, args.max_ans_len, args.use_squad_v2) #from util.py

            #print(p1[10].max(), p1[10].argmax(), p2[10].max(), p2[10].argmax()) #for debug only

            # Log info
            progress_bar.update(batch_size)
            if args.split != 'test':
                # No labels for the test set, so NLL would be invalid
                progress_bar.set_postfix(NLL=nll_meter.avg)

            idx2pred, uuid2pred = util.convert_tokens(gold_dict,
                                                      ids.tolist(),
                                                      starts.tolist(),
                                                      ends.tolist(),
                                                      args.use_squad_v2) #from util.py
            pred_dict.update(idx2pred)
            sub_dict.update(uuid2pred)

    # Log results (except for test set, since it does not come with labels)
    if args.split != 'test':
        results = util.eval_dicts(gold_dict, pred_dict, args.use_squad_v2) #from util.py
        results_list = [('NLL', nll_meter.avg),
                        ('F1', results['F1']),
                        ('EM', results['EM'])]
        if args.use_squad_v2:
            results_list.append(('AvNA', results['AvNA']))
        results = OrderedDict(results_list)

        # Log to console
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
        log.info(f'{args.split.title()} {results_str}')

        # Log to TensorBoard
        tbx = SummaryWriter(args.save_dir) #'./save'
        # tbx.add_graph(model, (cw_idxs, qw_idxs), verbose=False) #without char_embeddings
        tbx.add_graph(model, (cw_idxs, cc_idxs, qw_idxs, qc_idxs), verbose=False) #with char_embeddings
        util.visualize(tbx,
                  pred_dict=pred_dict,
                  eval_path=eval_file,
                  step=0,
                  split=args.split,
                  num_visuals=args.num_visuals) #from util.py
        tbx.close()
    # Write submission file
    sub_path = join(args.save_dir, args.split + '_' + args.sub_file)
    log.info(f'Writing submission file to {sub_path}...')
    #with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh: #for some reason complains about 'newline='
    with open(sub_path, 'w', encoding='utf-8') as csv_fh:
        csv_writer = csv.writer(csv_fh, delimiter=',')

        # #original version
        # csv_writer.writerow(['Id', 'Predicted'])
        # for uuid in sorted(sub_dict):
        #     csv_writer.writerow([uuid, sub_dict[uuid]])

        csv_writer.writerow(['Id', 'Predicted', 'Id', 'Predicted'])
        for a,b in zip(sub_dict.items(), pred_dict.items()):
            csv_writer.writerow([a[0], a[1], b[0], b[1]])

####################################
from args import get_test_args
import BiDAF
import QANet

if __name__ == "__main__":
    test_args = get_test_args()  #from args.py
    # print(test_args)
    # print(vars(test_args))

    # model_type = eval(test_args.model_type)
    model_dict = {'QANet.QANet': QANet.QANet, 'BiDAF.BiDAF': BiDAF.BiDAF, \
    'BiDAF.BiDAF_with_char_embed': BiDAF.BiDAF_with_char_embed}
    model_type = model_dict[test_args.model_type]

    main(test_args, model_type)
