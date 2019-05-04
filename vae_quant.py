import argparse
import os
from pprint import pprint
import time
import math
from numbers import Number
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import visdom
from torch.autograd import Variable
from torch.utils.data import DataLoader

from attr_functions import CELEBA_SENS_IDX

import lib.dist as dist
import lib.utils as utils
import lib.datasets as dset
from lib.flows import FactorialNormalizingFlow
from lib.models import MLPClassifier

from elbo_decomposition import elbo_decomposition
from plot_latent_vs_true import plot_vs_gt_shapes, plot_vs_gt_faces, plot_vs_gt_celeba  # noqa: F401

from audit import get_label_fn, get_attr_fn, get_repr_fn

SENS_IDX = [13, 15, 20]  # celeb-a sens attr; TODO make these an argument


class MLPEncoder(nn.Module):
    def __init__(self, output_dim):
        super(MLPEncoder, self).__init__()
        self.output_dim = output_dim

        self.fc1 = nn.Linear(4096, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, output_dim)

        self.conv_z = nn.Conv2d(64, output_dim, 4, 1, 0)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 64 * 64)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        h = self.fc3(h)
        z = h.view(x.size(0), self.output_dim)
        return z


class MLPDecoder(nn.Module):
    def __init__(self, input_dim):
        super(MLPDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 4096)
        )

    def forward(self, z):
        h = z.view(z.size(0), -1)
        h = self.net(h)
        mu_img = h.view(z.size(0), 1, 64, 64)
        return mu_img


class ConvEncoder(nn.Module):
    def __init__(self, output_dim, n_chan):
        super(ConvEncoder, self).__init__()
        self.output_dim = output_dim
        self.n_chan = n_chan

        self.conv1 = nn.Conv2d(n_chan, 64, 4, 2, 1)  # 32 x 32
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)  # 16 x 16
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1)  # 4 x 4
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 4)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_z = nn.Conv2d(512, output_dim, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, self.n_chan, 64, 64)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        z = self.conv_z(h).view(x.size(0), self.output_dim)
        return z


class ConvDecoder(nn.Module):
    def __init__(self, input_dim, n_chan):
        super(ConvDecoder, self).__init__()
        self.n_chan = n_chan
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, 1, 1, 0)  # 1 x 1
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 64, 4, 1, 0)  # 4 x 4
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # 16 x 16
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 32, 4, 2, 1)  # 32 x 32
        self.bn5 = nn.BatchNorm2d(32)
        self.conv_final = nn.ConvTranspose2d(32, self.n_chan, 4, 2, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        h = z.view(z.size(0), z.size(1), 1, 1)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        mu_img = self.conv_final(h)
        return mu_img


class SensVAE(nn.Module):
    def __init__(self, z_dim, use_cuda=False, prior_dist=dist.Normal(), q_dist=dist.Normal(),
                 include_mutinfo=True, tcvae=False, conv=False, mss=False, n_chan=1, 
                 sens_idx=[], x_dist=dist.Bernoulli(), a_dist=dist.Bernoulli(),
                 clf_samps=False):
        super(SensVAE, self).__init__()

        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.include_mutinfo = include_mutinfo
        self.tcvae = tcvae
        self.lamb = 0
        self.beta = 1
        self.beta_sens = 1
        # ^ the values of these hyperparams are correctly set later on
        self.mss = mss
        self.x_dist = x_dist
        self.a_dist = a_dist
        self.clf_samps = clf_samps
        self.n_chan = n_chan
        self.sens_idx = sens_idx

        # Model-specific
        # distribution family of p(z)
        self.prior_dist = prior_dist
        self.q_dist = q_dist
        # hyperparameters for prior p(z)
        self.register_buffer('prior_params', torch.zeros(self.z_dim, 2))

        # create the encoder and decoder networks
        if conv:
            self.encoder = ConvEncoder(z_dim * self.q_dist.nparams, n_chan)
            self.decoder = ConvDecoder(z_dim, n_chan)
        else:
            self.encoder = MLPEncoder(z_dim * self.q_dist.nparams)
            self.decoder = MLPDecoder(z_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
    # return prior parameters wrapped in a suitable Variable
    def _get_prior_params(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params.size()
        prior_params = Variable(self.prior_params.expand(expanded_size))
        return prior_params

    # samples from the model p(x|z)p(z)
    def model_sample(self, batch_size=1):
        # sample from prior (value will be sampled by guide when computing the ELBO)
        prior_params = self._get_prior_params(batch_size)
        zs = self.prior_dist.sample(params=prior_params)
        # decode the latent code z
        x_params = self.decoder.forward(zs)
        return x_params

    # define the guide (i.e. variational distribution) q(z|x)
    def encode(self, x):
        x = x.view(x.size(0), self.n_chan, 64, 64)
        # use the encoder to get the parameters used to define q(z|x)
        z_params = self.encoder.forward(x).view(x.size(0), self.z_dim, self.q_dist.nparams)
        logvars = z_params.select(-1, 1)
        print('logvar means {}, min {}'.format(
            logvars.mean().item(), logvars.min().item()),
            file=open('logvars-after.log', 'a'))
        # sample the latent code z
        zs = self.q_dist.sample(params=z_params)
        return zs, z_params

    def decode(self, z, z_params):
        x_params = self.decoder.forward(z).view(z.size(0), self.n_chan, 64, 64)
        sens_idx_mus = [2*i for i in self.sens_idx]
        a_params = z_params.select(-1, 0)[:, self.sens_idx]
        a_recon = a_params.sigmoid()
        if isinstance(self.x_dist, dist.Normal):
            x_params = x_params.sigmoid()  # model logit(img) as Normal
            x_params = torch.stack([x_params, torch.zeros_like(x_params)], -1)  # logsigma = 0
            #xs = self.x_dist.sample(params=x_params).sigmoid()
            xs = x_params
        return xs, x_params, a_recon, a_params

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        zs, z_params = self.encode(x)
        xs, x_params, a_recon, a_params = self.decode(zs, z_params)
        return xs, x_params, zs, z_params, a_recon, a_params

    def _log_importance_weight_matrix(self, batch_size, dataset_size):
        N = dataset_size
        M = batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
        W.view(-1)[::M+1] = 1 / N
        W.view(-1)[1::M+1] = strat_weight
        W[M-1, 0] = strat_weight
        return W.log()

    def elbo(self, x, a, dataset_size):
        metrics = {}
        # log p(x|z) + log p(z) - log q(z|x)
        batch_size = x.size(0)
        x = x.view(batch_size, self.n_chan, 64, 64)
        prior_params = self._get_prior_params(batch_size)
        x_recon, x_params, zbs, zb_params, a_recon, a_params = self.reconstruct_img(x)
        #logpx = self.x_dist.log_density(utils.logit(x), params=x_params).view(batch_size, -1).sum(1)
        logpx = self.x_dist.log_density(x, params=x_params).view(batch_size, -1).sum(1)
        a_ = a[:, self.sens_idx]  # the attributes we care about modeling
        if self.clf_samps:  # use samples for classification
            a_logit = zbs[:, self.sens_idx]
        else:
            a_logit = a_params
        with torch.no_grad():
            clf_accs = (a_logit > 0.).float().eq(a_).float().mean(0)
            for i, s in enumerate(self.sens_idx):
                metrics.update({'clf_acc{}'.format(s): clf_accs[i]})
        logpa = self.a_dist.log_density(a_, params=a_logit).view(batch_size, -1).sum(1)
        logpzb = self.prior_dist.log_density(zbs, params=prior_params).view(batch_size, -1).sum(1)
        logqzb_condx = self.q_dist.log_density(zbs, params=zb_params).view(batch_size, -1).sum(1)

        elbo = logpx + logpa + logpzb - logqzb_condx

        if self.beta == 1 and self.include_mutinfo and self.lamb == 0:
            return elbo, elbo.detach(), metrics

        # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
        _logqzb = self.q_dist.log_density(
            zbs.view(batch_size, 1, self.z_dim),
            zb_params.view(1, batch_size, self.z_dim, self.q_dist.nparams)
        )

        if not self.mss:
            # minibatch weighted sampling
            logqzb = (logsumexp(_logqzb.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))  # = log q(z,b)
            n_sens = len(self.sens_idx)
            n_nonsens = self.z_dim - n_sens
            mask = torch.zeros(1, 1, 100).byte()
            mask[:, :, self.sens_idx] = 1
            mask = mask.cuda(async=True)
            _logqb = _logqzb.masked_select(mask) \
                    .reshape(batch_size, batch_size, n_sens)  # sens latents
            _logqz = _logqzb.masked_select(1 - mask) \
                    .reshape(batch_size, batch_size, n_nonsens)  # nonsens latents
            logqb_prodmarginals = (logsumexp(_logqb, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
            logqz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))
            logqzb_prodmarginals = logqz + logqb_prodmarginals # = log q(z) + sum_k log q(b_k)
        else:
            # minibatch stratified sampling
            raise ValueError('minibatch stratified sampling not supported in this fork')
            logiw_matrix = Variable(self._log_importance_weight_matrix(batch_size, dataset_size).type_as(_logqz.data))
            logqz = logsumexp(logiw_matrix + _logqz.sum(2), dim=1, keepdim=False)
            logqz_prodmarginals = logsumexp(
                logiw_matrix.view(batch_size, batch_size, 1) + _logqz, dim=1, keepdim=False).sum(1)

        if not self.tcvae:
            if self.include_mutinfo:
                modified_elbo = logpx + self.beta_sens * logpa - self.beta * (
                    (logqzb_condx - logpzb) -
                    self.lamb * (logqzb_prodmarginals - logpzb)
                )
            else:
                modified_elbo = logpx + self.beta_sens * logpa - self.beta * (
                    (logqzb - logqzb_prodmarginals) +
                    (1 - self.lamb) * (logqzb_prodmarginals - logpzb)
                )
        else:
            if self.include_mutinfo:
                modified_elbo = logpx + self.beta_sens * logpa - \
                    (logqzb_condx - logqzb) - \
                    self.beta * (logqzb - logqzb_prodmarginals) - \
                    (1 - self.lamb) * (logqzb_prodmarginals - logpzb)
            else:
                modified_elbo = logpx + self.beta_sens * logpa - \
                    self.beta * (logqzb - logqzb_prodmarginals) - \
                    (1 - self.lamb) * (logqzb_prodmarginals - logpzb)

        metrics.update(tc=(logqzb - logqzb_prodmarginals).detach().mean())  # total correlation
        return modified_elbo, elbo.detach(), metrics


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


# for loading and batching datasets
def setup_data_loaders(args, use_cuda=False):
    if args.dataset == 'celeba':
        datasets = {
                'train': dset.CelebA(mode='train'),
                'validation': dset.CelebA(mode='validation', train=False),
                'test': dset.CelebA(mode='test', train=False),
                }
        train_set = dset.CelebA(mode='train')
    else:
        raise ValueError('Unknown dataset ' + str(args.dataset))
    kwargs = {'num_workers': 4, 'pin_memory': use_cuda}
    loaders = {k: DataLoader(dataset=v,
        batch_size=args.batch_size, shuffle=True, **kwargs)
        for k, v in datasets.items()}

    return loaders


win_samples = 'samples'
win_test_reco = 'test_reco'
win_latent_walk = 'latent_walk'
win_train_elbo = 'train_elbo'
win_train_tc = 'train_tc'


def display_samples(model, x, vis):
    global win_samples, win_test_reco, win_latent_walk

    # plot random samples
    sample_mu = model.model_sample(batch_size=100).sigmoid()
    sample_mu = sample_mu
    #images = list(sample_mu.view(-1, model.n_chan, 64, 64).data.cpu())
    images = sample_mu.view(-1, model.n_chan, 64, 64).data.cpu()
    win_samples = vis.images(images, 10, 2, 
            opts={'caption': 'samples', 'title': win_samples}, win=win_samples)

    # plot the reconstructed distribution for the first 50 test images
    test_imgs = x[:50, :]
    _, reco_imgs, zs, _, _, _ = model.reconstruct_img(test_imgs)
    if isinstance(model.x_dist, dist.Normal):  # we plot the means only
        reco_imgs = reco_imgs.select(-1, 0)
    #reco_imgs = reco_imgs.sigmoid()
    #test_reco_imgs = torch.cat([
        #test_imgs.view(1, -1, 64, 64), reco_imgs.view(1, -1, 64, 64)], 0).transpose(0, 1)

    test_reco_imgs = torch.cat([
        test_imgs.transpose(0, 1), reco_imgs.transpose(0, 1)], 0
        ).transpose(0, 1)
    win_test_reco = vis.images(
        #list(test_reco_imgs.contiguous().view(-1, model.n_chan, 64, 64).data.cpu()), 10, 2,
        test_reco_imgs.contiguous().view(-1, model.n_chan, 64, 64).data.cpu(), 10, 2,
        opts={'caption': 'test reconstruction image', 'title': win_test_reco}, win=win_test_reco)
    
    def red_frame(imgs):
        n_pixels = 3  # width of frame
        imgs[:, 0, :n_pixels, :] = 1.
        imgs[:, 1, :n_pixels, :] = 0.
        imgs[:, 2, :n_pixels, :] = 0.
        imgs[:, 0, -n_pixels:, :] = 1.
        imgs[:, 1, -n_pixels:, :] = 0.
        imgs[:, 2, -n_pixels:, :] = 0.
        imgs[:, 0, :, :n_pixels] = 1.
        imgs[:, 1, :, :n_pixels] = 0.
        imgs[:, 2, :, :n_pixels] = 0.
        imgs[:, 0, :, -n_pixels:] = 1.
        imgs[:, 1, :, -n_pixels:] = 0.
        imgs[:, 2, :, -n_pixels:] = 0.
        return imgs

    # plot latent walks (change one variable while all others stay the same)
    #zs = zs[0:3]
    zs = zs[0].unsqueeze(0)
    batch_size, z_dim = zs.size()
    xs = []
    delta = torch.autograd.Variable(torch.linspace(-2, 2, 7), volatile=True).type_as(zs)
    for i in range(z_dim):
        vec = Variable(torch.zeros(z_dim)).view(1, z_dim).expand(7, z_dim).contiguous().type_as(zs)
        vec[:, i] = 1
        vec = vec * delta[:, None]
        zs_delta = zs.clone().view(batch_size, 1, z_dim)
        zs_delta[:, :, i] = 0
        zs_walk = zs_delta + vec[None]
        xs_walk = model.decoder.forward(zs_walk.view(-1, z_dim)).sigmoid()
        if i in SENS_IDX:
            xs_walk = red_frame(xs_walk)
        xs.append(xs_walk)

    #xs = list(torch.cat(xs, 0).data.cpu())
    xs = torch.cat(xs, 0).data.cpu()
    win_latent_walk = vis.images(xs, 7, 2, 
            opts={'caption': 'latent walk', 'title': win_latent_walk}, 
            win=win_latent_walk)


def plot_elbo(train_elbo, vis):
    global win_train_elbo
    win_train_elbo = vis.line(torch.Tensor(train_elbo), 
            opts={'markers': True, 'title': win_train_elbo, 'caption': win_train_elbo}, 
            win=win_train_elbo)

def plot_tc(train_tc, vis):
    global win_train_tc
    win_train_tc = vis.line(torch.Tensor(train_tc), 
            opts={'markers': True, 'title': win_train_tc, 'caption': win_train_tc}, 
            win=win_train_tc)


def anneal_kl(args, vae, iteration):
    if args.dataset == 'celeba':
        warmup_iter = 2500

    if args.lambda_anneal:
        vae.lamb = max(0, 0.95 - 1 / warmup_iter * iteration)  # 1 --> 0
    else:
        vae.lamb = 0
    if args.beta_anneal:
        vae.beta = min(args.beta, args.beta / warmup_iter * iteration)  # 0 --> 1
    else:
        vae.beta = args.beta


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-d', '--dataset', default='celeba', type=str, help='dataset name',
        choices=['celeba'])
    parser.add_argument('-dist', default='normal', type=str, choices=['normal', 'laplace', 'flow'])
    parser.add_argument('-n', '--num-epochs', default=50, type=int, help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=2048, type=int, help='batch size')
    parser.add_argument('-l', '--learning-rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('-z', '--latent-dim', default=100, type=int, help='size of latent dimension')
    parser.add_argument('--beta', default=1, type=float, help='ELBO penalty term')
    parser.add_argument('--beta_sens', default=20, type=float, help='Relative importance of predicting sensitive attributes')
    #parser.add_argument('--sens_idx', default=[13, 15, 20], type=list, help='Relative importance of predicting sensitive attributes')
    parser.add_argument('--tcvae', action='store_true')
    parser.add_argument('--exclude-mutinfo', action='store_true')
    parser.add_argument('--beta-anneal', action='store_true')
    parser.add_argument('--lambda-anneal', action='store_true')
    parser.add_argument('--mss', action='store_true', help='use the improved minibatch estimator')
    parser.add_argument('--conv', action='store_true')
    parser.add_argument('--clf_samps', action='store_true')
    parser.add_argument('--clf_means', action='store_false', dest='clf_samps')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--visdom', action='store_true', help='whether plotting in visdom is desired')
    parser.add_argument('--save', default='betatcvae-celeba')
    parser.add_argument('--log_freq', default=200, type=int, help='num iterations per log')
    parser.add_argument('--audit', action='store_true',
            help='after each epoch, audit the repr wrt fair clf task')
    args = parser.parse_args()
    print(args)
    
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    writer = SummaryWriter(args.save)

    log_file = os.path.join(args.save, 'train.log')
    if os.path.exists(log_file):
        os.remove(log_file)

    print(vars(args))
    print(vars(args), file=open(log_file, 'w'))

    torch.cuda.set_device(args.gpu)

    # data loader
    loaders = setup_data_loaders(args, use_cuda=True)

    # setup the VAE
    if args.dist == 'normal':
        prior_dist = dist.Normal()
        q_dist = dist.Normal()
    elif args.dist == 'laplace':
        prior_dist = dist.Laplace()
        q_dist = dist.Laplace()
    elif args.dist == 'flow':
        prior_dist = FactorialNormalizingFlow(dim=args.latent_dim, nsteps=32)
        q_dist = dist.Normal()

    x_dist = dist.Normal() if args.dataset == 'celeba' else dist.Bernoulli()
    a_dist = dist.Bernoulli()
    vae = SensVAE(z_dim=args.latent_dim, use_cuda=True, prior_dist=prior_dist, 
            q_dist=q_dist, include_mutinfo=not args.exclude_mutinfo, 
            tcvae=args.tcvae, conv=args.conv, mss=args.mss, 
            n_chan=3 if args.dataset == 'celeba' else 1, sens_idx=SENS_IDX,
            x_dist=x_dist, a_dist=a_dist, clf_samps=args.clf_samps)

    if args.audit:
        audit_label_fn = get_label_fn(
                dict(data=dict(name='celeba', label_fn='H'))
                )
        audit_repr_fns = dict()
        audit_attr_fns = dict()
        audit_models = dict()
        audit_train_metrics = dict()
        for attr_fn_name in CELEBA_SENS_IDX.keys():
            model = MLPClassifier(args.latent_dim, 1000, 2)
            model.cuda()
            audit_models[attr_fn_name] = model
            audit_repr_fns[attr_fn_name] = get_repr_fn(
                dict(data=dict(
                    name='celeba', repr_fn='remove_all', attr_fn=attr_fn_name))
                )
            audit_attr_fns[attr_fn_name] = get_attr_fn(
                dict(data=dict(name='celeba', attr_fn=attr_fn_name))
                )

    # setup the optimizer
    optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate)
    if args.audit:
        Adam = optim.Adam
        audit_optimizers = dict()
        for k, v in audit_models.items():
            audit_optimizers[k] = Adam(v.parameters(), lr=args.learning_rate)


    # setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom(env=args.save, port=3776)

    train_elbo = []
    train_tc = []

    # training loop
    dataset_size = len(loaders['train'].dataset)
    num_iterations = len(loaders['train']) * args.num_epochs
    iteration = 0
    # initialize loss accumulator
    elbo_running_mean = utils.RunningAverageMeter()
    tc_running_mean = utils.RunningAverageMeter()
    clf_acc_meters = {'clf_acc{}'.format(s): utils.RunningAverageMeter() for s in vae.sens_idx}

    while iteration < num_iterations:
        bar = tqdm(range(len(loaders['train'])))
        for i, (x, a) in enumerate(loaders['train']):
            bar.update()
            iteration += 1
            batch_time = time.time()
            vae.train()
            if args.audit:
                for model in audit_models.values():
                    model.train()
            #anneal_kl(args, vae, iteration)  # TODO try annealing beta/beta_sens
            vae.beta = args.beta
            vae.beta_sens = args.beta_sens
            optimizer.zero_grad()
            if args.audit:
                for opt in audit_optimizers.values():
                    opt.zero_grad()
            # transfer to GPU
            x = x.cuda(async=True)
            a = a.float()
            a = a.cuda(async=True)
            # wrap the mini-batch in a PyTorch Variable
            x = Variable(x)
            a = Variable(a)
            # do ELBO gradient and accumulate loss
            obj, elbo, metrics = vae.elbo(x, a, dataset_size)
            if utils.isnan(obj).any():
                raise ValueError('NaN spotted in objective.')
            obj.mean().mul(-1).backward()
            elbo_running_mean.update(elbo.mean().data.item())
            tc_running_mean.update(metrics['tc'])
            for (s, meter), (_, acc) in zip(clf_acc_meters.items(), metrics.items()):
                clf_acc_meters[s].update(acc.data.item())
            optimizer.step()

            if args.audit:
                # now re-encode x and take a step to train each audit classifier
                with torch.no_grad():
                    zs, z_params = vae.encode(x)
                    if args.clf_samps:
                        z = zs
                    else:
                        z_mu = z_params.select(-1, 0)
                        z = z_mu
                    a_all = a
                for subgroup, model in audit_models.items():
                    metrics_dict = {}
                    # noise out sensitive dims of latent code
                    z_ = z.clone()
                    a_all_ = a_all.clone()
                    # subsample to just sens attr of interest for this subgroup
                    a_ = audit_attr_fns[subgroup](a_all_)
                    # noise out sensitive dims for this subgroup
                    z_ = audit_repr_fns[subgroup](z_, None, None)
                    y_ = audit_label_fn(a_all_).long()

                    loss, _, metrics = model(z_, y_, a_)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    metrics_dict.update(loss=loss.detach().item())
                    for k, v in metrics.items():  # hopefully this works...
#                        print(k)
#                        import pdb
#                        pdb.set_trace()
                        #if k == 'ypred':
                        if v.numel() > 1:
                            k += '-avg'
                            v = v.float().mean()
                        metrics_dict.update({k:v.detach().item()})
#                        if not k in monitoring_tensors:
#                            metrics_dict.update({k:v.detach().item()})
#                        else:
#                            metrics_dict.update({k:v.detach()})
                    audit_train_metrics[subgroup] = metrics_dict



            # report training diagnostics
            if iteration % args.log_freq == 0:
                if args.audit:
                    for subgroup, metrics in audit_train_metrics.items():
                        for metric_name, metric_value in metrics.items():
                            writer.add_scalar(
                                    '{}/{}'.format(subgroup, metric_name),
                                    metric_value, iteration)

                train_elbo.append(elbo_running_mean.avg)
                writer.add_scalar('train_elbo', elbo_running_mean.avg, iteration)
                train_tc.append(tc_running_mean.avg)
                writer.add_scalar('train_tc', tc_running_mean.avg, iteration)
                msg = '[iteration %03d] time: %.2f \tbeta %.2f \tlambda %.2f training ELBO: %.4f (%.4f) training TC %.4f (%.4f)' % (
                    iteration, time.time() - batch_time, vae.beta, vae.lamb,
                    elbo_running_mean.val, elbo_running_mean.avg,
                    tc_running_mean.val, tc_running_mean.avg)
                for k, v in clf_acc_meters.items():
                    msg += ' {}: {:.2f}'.format(k, v.avg)
                    writer.add_scalar(k, v.avg, iteration)
                print(msg)
                print(msg, file=open(log_file, 'a'))

                vae.eval()

                # plot training and test ELBOs
                if args.visdom:
                    display_samples(vae, x, vis)
                    plot_elbo(train_elbo, vis)
                    plot_tc(train_tc, vis)

                utils.save_checkpoint({
                    'state_dict': vae.state_dict(),
                    'args': args}, args.save, iteration // len(loaders['train']))
                eval('plot_vs_gt_' + args.dataset)(vae, loaders['train'].dataset,
                    os.path.join(args.save, 'gt_vs_latent_{:05d}.png'.format(iteration)))

    # Report statistics after training
    vae.eval()
    utils.save_checkpoint({
        'state_dict': vae.state_dict(),
        'args': args}, args.save, 0)
    dataset_loader = DataLoader(loaders['train'].dataset, batch_size=1000, num_workers=1, shuffle=False)
    if False:
        logpx, dependence, information, dimwise_kl, analytical_cond_kl, marginal_entropies, joint_entropy = \
            elbo_decomposition(vae, dataset_loader)
        torch.save({
            'logpx': logpx,
            'dependence': dependence,
            'information': information,
            'dimwise_kl': dimwise_kl,
            'analytical_cond_kl': analytical_cond_kl,
            'marginal_entropies': marginal_entropies,
            'joint_entropy': joint_entropy
        }, os.path.join(args.save, 'elbo_decomposition.pth'))
    eval('plot_vs_gt_' + args.dataset)(vae, dataset_loader.dataset, os.path.join(args.save, 'gt_vs_latent.png'))
    return vae


if __name__ == '__main__':
    model = main()
