#train_utils.py
"""Things to help witht the training process

Author:
    Amit Patel (amitpatel.gt@gmail.com)
"""

'''
NOTE: ALL THE PATHS ARE REFERENCED FROM THE DIRECTORY WHERE THE COLAB NOTEBOOK IS SAVED BCSE OS.CHDIR() \
IS MADE TO POINT THERE AND ALL .py FILES INSIDE SRC DIRECTORY ARE ONLY CALLED FROM THE COLAB NOTEBOOK AND ALSO BCSE \
THE PATH TO ALL THE SRC/*.PY FILES IS ADDED TO SYS.PATH.
'''

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import numpy as np
import time
import util


class LR_Finder(object):
    def __init__(self, model_type, dataset, seed, n_epochs, l2_wd, batch_size, **model_kwargs):        
        # Set random seed
        print(f'Using random seed {seed}...')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed) #no need to do this
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        #some constants
        self.seed = seed
        self.minLR=1.e-5
        self.maxLR=1.e-1
        self.lrMult=1.5
        self.data_size=len(dataset)
        self.batch_size=batch_size
        self.n_epochs=n_epochs
        self.l2_wd=l2_wd
        self.last_window=10
        self.model_type=model_type
        self.model_kwargs = model_kwargs
        self.device, self.dtype = 'cuda:0', torch.float32

        n = len(dataset)
        self.dataset = dataset
        # self.dataset.move_to(self.device) #no improvement in speed
        self.dataset.shuffle(self.data_size)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        print(f'Total data size: {n} and training data size: {len(self.dataset)}')


    def delete_batchify_dataset(self):
        #this is not used... using dataloader instead
        strt = 0
        end = self.batch_size
        while end <= self.dataset[0].shape[0]:
            res = []
            for d in self.dataset:
                res.append(d[strt:end])
            yield res
            strt += self.batch_size
            end += self.batch_size


    def train(self, model, optimizer, hooks=None, n_epochs=None, l_window=None, use_lr_scheduler=False, verbose=False):
        model = model.to(device=self.device, dtype=self.dtype)
        if use_lr_scheduler:
            lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, \
            lambda n_iter: min(n_iter//2000, max(n_iter/4000 - 0.5,0), 1.0))
        num_epochs = n_epochs if n_epochs else self.n_epochs
        tot_loss = 0.0
        last_window = l_window if l_window else self.last_window
        losses = []
        iteration = 0
        model.train()
        with torch.enable_grad():
            for e in range(num_epochs):
                for b in self.dataloader:
                    cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids = b
                    cw_idxs = cw_idxs.to(self.device)
                    qw_idxs = qw_idxs.to(self.device)
                    cc_idxs = cc_idxs.to(self.device)
                    qc_idxs = qc_idxs.to(self.device)
                    y1, y2 = y1.to(self.device), y2.to(self.device)

                    optimizer.zero_grad()
                    log_p1, log_p2 = model(cw_idxs, cc_idxs, qw_idxs, qc_idxs) #with char_embeddings
                    loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2) #hard labelling loss
                
                    loss.backward()
                    if hooks: hooks.mybackward_hook()
                    optimizer.step()

                    loss_val = loss.item()
                    if e >= (num_epochs-last_window):
                        losses.append(loss_val)

                    lr_rate = optimizer.state_dict()['param_groups'][0]['lr']
                    if use_lr_scheduler: lr_sched.step()
                    curr_time = str(time.localtime()[3])+':'+str(time.localtime()[4])

                    if (verbose and (iteration%500 == 0)): print(f'{curr_time}Hrs - Loss val for iteration {iteration} and epoch {e} is: {loss_val} and learning rate is: {lr_rate}')
                    iteration += 1
            if verbose: print(f'{curr_time} -Loss val for iteration {iteration} of last epoch i.e. #{e} is: {loss_val} and learning rate is: {lr_rate}')
        return (torch.tensor(losses).mean().item(), losses)



    def find(self):
        print('Running LR Finder...')

        best_loss = float('inf')
        best_lr = None
        lr = self.minLR
        lr_lst, loss_lst = [], []

        while True:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            # torch.cuda.manual_seed_all(self.seed) #no need to do this
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            model = self.model_type(**self.model_kwargs)
            model = model.to(device=self.device, dtype=self.dtype)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=self.l2_wd)
            tot_loss, _ = self.train(model, optimizer)
            print(lr, tot_loss)
            lr_lst.append(lr)
            loss_lst.append(tot_loss)
            if tot_loss < best_loss:
                best_loss = tot_loss
                best_lr = lr
            if lr > self.maxLR or best_loss == float('inf') or tot_loss > 20*best_loss:
                print(f'Best lr: {best_lr} and the corresponding loss is: {best_loss}')
                self.best_lr = best_lr
                plt.plot(lr_lst, loss_lst)
                plt.xscale('log')
                plt.yscale('log')
                plt.xlabel('Learning Rate')
                plt.ylabel('Loss')
                plt.grid()
                plt.savefig('./save/LR_finder.png', dpi=200)
                plt.clf()
                break
            lr *= self.lrMult


    def train_debug(self, lr, use_hooks=False):
        print('Running Train Debug...')
        n_epochs = self.n_epochs
        l_window=n_epochs
        model = self.model_type(**self.model_kwargs)
        model = model.to(device=self.device, dtype=self.dtype)
        hooks = AddHooks(model) if use_hooks else None
        # util.get_model_info(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=self.l2_wd)
        _, losses = self.train(model, optimizer, hooks=hooks, n_epochs=n_epochs, l_window=l_window, use_lr_scheduler=True, verbose=True)

        #plot loss
        plt.plot(losses)
        plt.xlabel('Training Iterations')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('./save/hooks_loss.png', dpi=200)
        plt.clf()

        #plot hooks
        if hooks: hooks.plot()


class AddHooks(object):
    def __init__(self, model):
        self.model = model
        self.hooks = {}
        self.data_mean_dist = {}
        self.data_percent_near_zero = {}
        self.data_mean = {}
        self.data_std = {}
        self.grads = {}
        for m, c in model.named_children():
            self.grads[m] = []
            self.data_percent_near_zero[m] = []
            self.data_mean_dist[m] = []
            self.data_mean[m] = []
            self.data_std[m] = []
            func1 = self.func(m)
            # self.hooks[m] = c.register_forward_hook(lambda m_,i_,o_: self.func(m,m_,i_,o_))
            self.hooks[m] = c.register_forward_hook(func1)


    def mybackward_hook(self):
            with torch.no_grad():
                for m,c in self.model.named_children():
                    params = []
                    params_grad = []
                    for _, p in c.named_parameters():
                        if p.grad is None: continue
                        params.append(p.view(-1))
                        params_grad.append(p.grad.view(-1))
                    if params == []: continue
                    norm_grad = torch.cat(params_grad).norm() / torch.cat(params).norm() * 100
                    self.grads[m].append(norm_grad)


    def func(self, m):
        def func1(m_,i_,o_):
            with torch.no_grad():
                # print('\n1', m,'\n2', m_, '\n3', type(m_))
                if isinstance(o_, (tuple,list)):
                    tmp1 = 0
                    for o in o_:
                        tmp1 = tmp1 + o #add log probs (for model output layer)
                    o_ = tmp1
                tmp1 = o_.reshape(-1) #o_.view(-1)
                self.data_mean_dist[m].append(tmp1)
                tmp2 = torch.logical_and(tmp1 < 0.1, tmp1 > -0.1).float().sum()/tmp1.numel()
                self.data_percent_near_zero[m].append(tmp2)
                self.data_mean[m].append(o_.mean().item())
                self.data_std[m].append(o_.std().item())
        return func1


    def plot(self):
        for m, c in self.model.named_children():
            if self.data_mean_dist[m] == []:
                continue
            y_data = []
            for i in range(len(self.data_mean_dist[m])):
                x_data = self.data_mean_dist[m][i]
                size = 1
                for s in x_data.shape:
                    size *= s
                y_data.append(i*torch.ones(size))
                # kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40)
                # plt.hist(x_data, **kwargs)
                # plt.hist2d(x_data.detach().numpy(), y_data[-1].detach().numpy(), bins=40, density=True)
                # plt.show()

            fig, axs = plt.subplots(2,2)
            # fig.suptitle('f{m}')
            x_data = torch.cat(self.data_mean_dist[m]).detach().cpu().numpy() #tolist()
            y_data = torch.cat(y_data).detach().cpu().numpy() #tolist()
            axs[0,0].set_title(f'Activation Distribution: {m}')
            h = axs[0,0].hist2d(x_data, y_data, bins=[30,len(self.data_mean[m])], density=False)
            axs[0,0].set_xlabel('Activation Value')
            axs[0,0].set_ylabel('Training Iteration')
            fig.colorbar(h[3], ax=axs[0,0])
            del x_data, y_data

            # axs[0,1].set_title('Activation -- Near Zero (+-0.1)')
            axs[0,1].plot(self.data_percent_near_zero[m])
            axs[0,1].set_ylabel('Percent Activations \n near 0 (+/-0.1)')
            axs[0,1].set_xlabel('Training Iteration')
            axs[0,1].grid(True)


            # axs[1,0].set_title('Activation -- Mean')
            axs[1,0].plot(self.data_mean[m])
            axs[1,0].set_ylabel('Mean')
            axs[1,0].set_xlabel('Training Iteration')
            axs[1,0].grid(True)

            # axs[1,1].set_title('Activation -- Standard Deviation')
            axs[1,1].plot(self.data_std[m])
            axs[1,1].set_ylabel('STD Dev')
            axs[1,1].set_xlabel('Training Iteration')
            axs[1,1].grid(True)

            plt.tight_layout()
            plt.savefig('./save/hooks_' + m + '.png', dpi=200)
            plt.clf()

            #plot the grads
            plt.title(f'Percent Gradient Norm for: {m}')
            plt.plot(self.grads[m])
            plt.grid(True)
            plt.savefig('./save/hooks_grads_' + m + '.png', dpi=200)
            plt.clf()

        self.remove()


    def remove(self):
        for k,v in self.hooks.items():
            v.remove()
        del self.hooks
        del self.data_mean_dist
        del self.data_mean
        del self.data_std
