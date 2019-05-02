import numpy as np

def label_fn_0(y):
    return (y[:,0] >= 1).long()

def label_fn_1(y):
    return (y[:,1] >= 1).long()

def label_fn_2(y):
    return (y[:,2] >= 3).long()

def label_fn_3(y):
    return (y[:,3] >= 20).long()

def label_fn_4(y):
    return (y[:,4] >= 16).long()

def label_fn_5(y):
    return (y[:,5] >= 16).long()

base_label_fns = [label_fn_0, label_fn_1, label_fn_2, label_fn_3, \
                    label_fn_4, label_fn_5]


def remove_ind(y, i):
    ind = np.array(list(filter(lambda x: x != i, range(y.shape[1]))))
    return y[:, ind]

def remove_ind_fn(y, fn):
    i = {'label_fn_0': 0,
            'label_fn_1': 1, 
            'label_fn_2': 2, 
            'label_fn_3': 3, 
            'label_fn_4': 4, 
            'label_fn_5': 5
            }[fn.__name__]
    return remove_ind(y, i) 


# celeb-a stuff
def label_fn_young(a):
    return (a[:,39] > 0).long()

def label_fn_male(a):
    return (a[:,20] > 0).long()

