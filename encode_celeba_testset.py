import argparse
import os
import pdb

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from vae_quant import SensVAE, setup_data_loaders
import lib.dist as dist
import lib.flows as flows
import lib.datasets as dset


SENS_IDX = [13, 15, 20]  # celeb-a sens attr; TODO make these an argument

def load_model_and_dataset(checkpt_filename):
    checkpt = torch.load(checkpt_filename)
    args = checkpt['args']
    state_dict = checkpt['state_dict']

    # backwards compatibility
    if not hasattr(args, 'conv'):
        args.conv = False

    x_dist = dist.Normal() if args.dataset == 'celeba' else dist.Bernoulli()
    a_dist = dist.Bernoulli()

    # model
    if args.dist == 'normal':
        prior_dist = dist.Normal()
        q_dist = dist.Normal()
    elif args.dist == 'laplace':
        prior_dist = dist.Laplace()
        q_dist = dist.Laplace()
    elif args.dist == 'flow':
        prior_dist = flows.FactorialNormalizingFlow(dim=args.latent_dim, nsteps=32)
        q_dist = dist.Normal()
    #vae = SensVAE(z_dim=args.latent_dim, use_cuda=True, prior_dist=prior_dist, q_dist=q_dist, conv=args.conv)
    vae = SensVAE(z_dim=args.latent_dim, use_cuda=True, prior_dist=prior_dist, 
            q_dist=q_dist, include_mutinfo=not args.exclude_mutinfo, 
            tcvae=args.tcvae, conv=args.conv, mss=args.mss, 
            n_chan=3 if args.dataset == 'celeba' else 1, sens_idx=SENS_IDX,
            x_dist=x_dist, a_dist=a_dist)

    vae.load_state_dict(state_dict, strict=False)
    vae.beta = args.beta
    vae.beta_sens = args.beta_sens
    vae.eval()

    # dataset loader
    loader = setup_data_loaders(args, use_cuda=True)

    # test loader
    test_set = dset.CelebA(mode='test')
    kwargs = {'num_workers': 4, 'pin_memory': True}
    test_loader = DataLoader(dataset=test_set,
        batch_size=args.batch_size, shuffle=False, **kwargs)

    return vae, loader, test_loader, args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            'encode celeb-a test set starting from a trained model checkpoint')
    parser.add_argument('--ckpt', type=str, 
            default='sweep/beta100betasens500/checkpt-0078.pth', 
            help='path to model checkpoint')
    args = parser.parse_args()
    ckpt = args.ckpt

    vae, loader, test_loader, args = load_model_and_dataset(ckpt)

    output_dir = os.path.splitext(ckpt)[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_npz_filename = os.path.join(output_dir, 'encoded_test_set.npz')

    x_test = []
    a_test = []
    z_test = []

    bar = tqdm(range(len(test_loader)))
    for i, (x, a) in enumerate(test_loader):
        bar.update()
        x_test.append(x)
        a_test.append(a)

        x = x.cuda(async=True)
        a = a.float()
        a = a.cuda(async=True)
        # wrap the mini-batch in a PyTorch Variable
        x = Variable(x)
        a = Variable(a)

        _, z_params = vae.encode(x)
        z_mu = z_params.select(-1, 0)
        z_test.append(z_mu.detach().cpu())

    x_test = torch.cat(x_test, 0).numpy()
    a_test = torch.cat(a_test, 0).numpy()
    z_test = torch.cat(z_test, 0).numpy()


    np.savez(output_npz_filename, x=x_test, a=a_test, z=z_test, args=args)
    print('done encoding test set:\n', output_npz_filename)

