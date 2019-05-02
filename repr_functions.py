import numpy as np
import torch
from scipy.stats import moment, norm
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MAX_REPR_DIMS = 100

def do_nothing(z, a, y):
    return z

def remove_one_correlated(z, a, y):
    corr_vals = []
    for i in range(z.shape[1]):
        col = z[:, i]
        corr = correlation(col, a)
        corr_vals.append(corr)
        # plot_gaussian(col, 'col_{:d}'.format(i))
    print(corr_vals)
    with np.printoptions(precision=3, suppress=True):
        print(np.corrcoef(z.T))
    for m in range(1, 11):
        print('Moment {:d}'.format(m))
        print(moment(z, moment=m))
    most_corr_i = np.argmax(np.abs(corr_vals))
    less_corr_i = list(filter(lambda i: i != most_corr_i, range(z.shape[1])))
    print(most_corr_i, less_corr_i)
    new_z = z[:, less_corr_i]
    print(z.shape, new_z.shape)
    return new_z


def remove_two_correlated(z, a, y):
    corr_vals = []
    for i in range(z.shape[1]):
        col = z[:, i]
        corr = correlation(col, a)
        corr_vals.append(corr)
        # plot_gaussian(col, 'col_{:d}'.format(i))
    print(corr_vals)
    with np.printoptions(precision=3, suppress=True):
        print(np.corrcoef(z.T))
    for m in range(1, 11):
        print('Moment {:d}'.format(m))
        print(moment(z, moment=m))
    ranked_i_by_corr = sorted(range(z.shape[1]), key=lambda i: corr_vals[i],\
                        reverse=True)
    most_corr_i = ranked_i_by_corr[0]
    most_corr_i2 = ranked_i_by_corr[1]
    less_corr_i = list(filter(lambda i: i not in [most_corr_i, most_corr_i2],\
                                        range(z.shape[1])))
    print(most_corr_i, most_corr_i2, less_corr_i)
    new_z = z[:, less_corr_i]
    print(z.shape, new_z.shape)
    return new_z

def correlation(x, y):
    x, y = torch.Tensor(x), torch.Tensor(y)
    x_mn, x_std = torch.mean(x), torch.std(x)
    y_mn, y_std = torch.mean(y), torch.std(y)
    centred_x = x - x_mn
    centred_y = y - y_mn
    cov = torch.mean(torch.mul(centred_x, centred_y))
    return cov / (x_std * y_std)

def plot_gaussian(x, name):
    figdir = 'figs/normal_vars'
    norm_params = norm.fit(x)
    plt.clf()
    plt.hist(x, density=True, alpha=0.2, label='hist', bins=50)
    xl = np.linspace(np.min(x), np.max(x), 100)
    y = norm.pdf(xl, *norm_params)
    plt.plot(xl, y, 'r-', lw=5, alpha=0.6, label='pdf')
    plt.axis([None, None, 0., max(y) * 1.1])
    plt.legend()
    plt.savefig(os.path.join(figdir, '{}.png'.format(name)))

def get_remove_numbered_dimension_fn(inds):
    def remove_numbered_dimension(z, a, y):
        z = z.clone()
        for i in inds:
            z[:, i] = torch.randn(len(z))
        return z
    return remove_numbered_dimension

for i in range(MAX_REPR_DIMS):
    exec('remove_dimension_{:d} = get_remove_numbered_dimension_fn([{:d}])'.format(i, i))

def define_remove_fn(fn_name):
    fn_info = fn_name.split('_')
    if fn_info[0] == 'remove' and fn_info[1] == 'dimension':
        dim_nums = [int(i) for i in fn_info[2].split('-')]
        return get_remove_numbered_dimension_fn(dim_nums)
    else:
        raise Exception("don't know how to create the repr function {}".format(fn_name))

if __name__ == '__main__':
    z = np.random.randint(10, size=(4,6))
    a = np.array([1., 1., 0., 0.])
    y = np.array([0.,0.,0.,0.])
    z, a, y = torch.tensor(z, dtype=torch.float), torch.tensor(a), torch.tensor(y)
    print(z.shape, a.shape, y.shape)
    print(z)
    # print(remove_one_correlated(z, a, y))
    print(remove_dimension_0(z, a, y))
    print(remove_dimension_1(z, a, y))
    print(remove_dimension_2(z, a, y))
    print(remove_dimension_3(z, a, y))
    print(remove_dimension_4(z, a, y))
    print(remove_dimension_5(z, a, y))


