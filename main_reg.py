"""
Author: ***
Code to produce the results in PHNN
"""

from data_builder import get_dataset
from utils import *
from model_builder import get_models
import pickle
import os
import torch
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('-ni', '--num_iters', type=int, default=1000)
parser.add_argument("-n_test_traj", '--ntesttraj', type=int, default=25)
parser.add_argument("-n_train_traj", '--ntraintraj', type=int, default=25)
parser.add_argument('-dt', '--dt', type=float, default=0.1)
parser.add_argument('-tmax', '--tmax', type=float, default=3.1)
parser.add_argument('-dname', '--dname', type=str, default='mass_spring')
parser.add_argument('-noise_std', '--noise', type=float, default=0)
parser.add_argument('-type','--type',type=int,default=1)
parser.add_argument('-batch_size','--batch_size',type=int,default=2000)
parser.add_argument('-learning_rate','--learning_rate',type=float,default=1e-3)
parser.add_argument('-embed_integ','--embed_integ',action="store_false")



args = parser.parse_args()
iters = args.num_iters
n_test_traj = args.ntesttraj
n_train_traj = args.ntraintraj
T_max = args.tmax
T_max_t = T_max
dt = args.dt
device = 'cpu'
print(device)
type_vec = args.type
num_samples_per_traj = int(np.ceil((T_max / dt))) - 1
lr_step = iters//2
dataset_name = args.dname



def train_model(model_name,model, optimizer, lr_sched, num_epochs=1, integrator_embedded=False,args_reg=None):
    loss_collater = {'train': [], 'valid': []}

    #load all training data
    for batch_i, (q, q_next, _, qdx, tevals) in enumerate(data_dict['train']):
        q, q_next, qdx = q.float(), q_next.float(), qdx.float()
        q=q
        q_next=q_next
        qdx=qdx
        tevals = tevals.float()
        tevals =tevals
        q.requires_grad = True
        tevals.requires_grad = True
        if args_reg:
            alpha = args_reg[0]
            beta = args_reg[1]

    # load all training data
    for batch_i, (qtest, q_nexttest, _, qdxtest, tevalstest) in enumerate(data_dict['valid']):
        qtest, q_nexttest, qdxtest = qtest.float(), q_nexttest.float(), qdxtest.float()
        qtest = qtest
        q_nexttest = q_nexttest
        qdxtest = qdxtest
        tevalstest = tevalstest.float()
        tevalstest = tevalstest
        qtest.requires_grad = True
        tevalstest.requires_grad = True

    #iterate over batches - epochs here is proxy for bs
    for epoch in range(num_epochs):
        ixs = torch.randperm(q.shape[0])[:args.batch_size]
        model.train()
        print('epoch:{}'.format(epoch))
        optimizer.zero_grad()

        if integrator_embedded:
            next_step_pred = model.next_step(q[ixs], tevals[ixs])
            state_loss = torch.mean((next_step_pred - q_next[ixs]) ** 2)
        else:
            next_step_pred = model.time_deriv(q[ixs], tevals[ixs])
            state_loss = (next_step_pred - qdx[ixs]).pow(2).mean()

        train_loss = state_loss

        if model_name =='TDHNN4':
            # print(model.get_weight())
            train_loss += alpha*torch.mean(torch.abs(model.get_F(tevals[ixs].reshape(-1,1))))
            train_loss += beta*torch.mean(torch.abs(model.get_D()))
        train_loss.backward()
        optimizer.step()

        loss_collater['train'].append(train_loss.detach().item())

        model.eval()
        ixstest = torch.randperm(qtest.shape[0])[:args.batch_size]

        if integrator_embedded:
            next_step_pred = model.next_step(qtest[ixstest], tevalstest[ixstest])
            test_state_loss = torch.mean((next_step_pred - q_nexttest[ixstest]) ** 2)
        else:
            next_step_pred = model.time_deriv(qtest[ixstest], tevalstest[ixstest])
            test_state_loss = (next_step_pred - qdxtest[ixstest]).pow(2).mean()

        loss_collater['valid'].append(test_state_loss.detach().item())

        print('{} Epoch Loss: {:.10f}'.format('train', loss_collater['train'][-1]))
        print('{} Epoch Loss: {:.10f}'.format('valid', loss_collater['valid'][-1]))
        lr_sched.step()

    plt.figure()
    plt.plot(loss_collater['train'], label='train')
    plt.plot(loss_collater['valid'], label='valid')
    plt.yscale('log')
    plt.title(f'{dataset_name},{model_name}, ntrain_inits:{n_train_traj},ntest_inits:{n_test_traj},tmax:{T_max},dt:{dt}')
    plt.legend()
    plt.savefig(f'{dataset_name}_{type_vec}_training.jpg')

    return model,loss_collater
    # return model


def get_model_results(arga,argb):
    model_type = get_models(dt, type='TDHNN4', hidden_dim=200)
    args_reg = [arga,argb]

    model_type = model_type
    model_name = 'TDHNN4'
    params_a = list(model_type.parameters())[:]
    optimizer_ft = torch.optim.Adam([{"params": params_a},
                                    ],
                                    args.learning_rate)
    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer_ft, lr_step//2, gamma=0.1)
    _,losses = train_model(model_name, model_type, optimizer_ft, lr_sched, num_epochs=iters,
                             integrator_embedded=args.embed_integ,args_reg=args_reg)
    return losses['valid'][-1]

model_dct = get_models(dt, type=None, hidden_dim=200)

noise_var = [0]

if __name__ == '__main__':

    for noise_val in noise_var:
        # dataset preprocessing
        train_data = get_dataset(dataset_name, n_train_traj, T_max, dt, noise_std=noise_val, seed=0,typev=type_vec)
        valid_data = get_dataset(dataset_name, n_test_traj, T_max_t, dt, noise_std=0, seed=14,typev=type_vec)
        BS = num_samples_per_traj

        tnow, tnext, tenergy, tdx, tevals = nownext(train_data, n_train_traj, T_max, dt, dt)
        vnow, vnext, venergy, vdx, vevals = nownext(valid_data, n_test_traj, T_max_t, dt, dt)

        print_every = 1000

        traindat = pendpixdata(tnow,tnext,tenergy,tdx,tevals)
        train_dataloader = DataLoader(traindat, batch_size=len(tnow), shuffle=True)
        valdat = pendpixdata(vnow, vnext, venergy, vdx, vevals)
        val_dataloader = DataLoader(valdat, batch_size=len(vnow), shuffle=True)

        data_dict = {'train': train_dataloader, 'valid': val_dataloader}
        running_losses = 0.
        best_scores ={}

        for model_name, model_type in model_dct.items():
            if model_name != 'TDHNN4':
                model_type = model_type
                params_a = list(model_type.parameters())[:]
                optimizer_ft = torch.optim.Adam([{"params": params_a}], args.learning_rate)
                lr_sched = torch.optim.lr_scheduler.StepLR(optimizer_ft, lr_step//2, gamma=0.1)
                trained_model,scores = train_model(model_name,model_type, optimizer_ft, lr_sched, num_epochs=iters, integrator_embedded=args.embed_integ)
                best_scores[model_name] = scores
            else:
                a_params = [1e-4,1e-6,1e-8]
                b_params = [1e-4,1e-6,1e-8]
                results = np.zeros((len(a_params),len(b_params)))
                for idex,a_param in enumerate(a_params):
                    for jdex,b_param in enumerate(b_params):
                        results[idex,jdex]= get_model_results(a_param,b_param)
                ids = np.unravel_index(results.argmin(), results.shape)
                results = np.array(results)
                np.save(f'results_{dataset_name}_{type_vec}_noise_{noise_val}',results)
                model_type = model_type
                params_a = list(model_type.parameters())[:]
                optimizer_ft = torch.optim.Adam([{"params": params_a}], args.learning_rate)
                lr_sched = torch.optim.lr_scheduler.StepLR(optimizer_ft, lr_step//2, gamma=0.1)
                trained_model,scores = train_model(model_name,model_type, optimizer_ft, lr_sched, num_epochs=iters, integrator_embedded=args.embed_integ,args_reg=[a_params[ids[0]],b_params[ids[1]]])
                best_scores[model_name] = scores

            parent_dir = os.getcwd()
            path = f"{dataset_name}_{type_vec}/{model_name}/noise_{noise_val}"
            if not os.path.exists(path):
                os.makedirs(parent_dir+'/'+path)
            torch.save(trained_model, path+'/'+'model')
            # test_model(model_name,trained_model)
            del trained_model

        with open(f'best_score_results_{dataset_name}_{type_vec}_noise_{noise_val}.pkl', 'wb') as f:
            pickle.dump(best_scores, f, pickle.HIGHEST_PROTOCOL)
