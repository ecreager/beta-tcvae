"""
Convert ground truth latent classes into binary sensitive attributes
"""

def attr_fn_0(y):
    return y[:,0] >= 1


def attr_fn_1(y):
    return y[:,1] >= 1


def attr_fn_2(y):
    return y[:,2] >= 3


def attr_fn_3(y):
    return y[:,3] >= 20


def attr_fn_4(y):
    return y[:,4] >= 16


def attr_fn_5(y):
    return y[:,5] >= 16


dsprites_attr_fns = [attr_fn_0, attr_fn_1, attr_fn_2, attr_fn_3, attr_fn_4, attr_fn_5]


# celeba stuff
def attr_fn_chubby(a):
    return a[:,13] > 0.

def attr_fn_eyeglasses(a):
    return a[:,15] > 0.

def attr_fn_male(a):
    return a[:,20] > 0.

def attr_fn_heavy_makeup(a):
    return a[:,18] > 0.

CELEBA_SUBGROUPS = {
        'H': attr_fn_heavy_makeup,
        'S': lambda a: a[:,31] > 0.,  # smiling
        'W': lambda a: a[:,36] > 0.,  # wears lipstick
        'A': lambda a: a[:,2] > 0.,  # wears lipstick
        'C': attr_fn_chubby,
        'E': attr_fn_eyeglasses,
        'M': attr_fn_male,
        'C $\land$ E': lambda a: attr_fn_chubby(a) * attr_fn_eyeglasses(a),
        'C $\land$ M': lambda a: attr_fn_chubby(a) * attr_fn_male(a),
        'E $\land$ M': lambda a: attr_fn_eyeglasses(a) * attr_fn_male(a),
        'C $\land$ $\\neg$ E': lambda a: attr_fn_chubby(a) * (1 - attr_fn_eyeglasses(a)),
        'C $\land$ $\\neg$ M': lambda a: attr_fn_chubby(a) * (1 - attr_fn_male(a)),
        'E $\land$ $\\neg$ M': lambda a: attr_fn_eyeglasses(a) * (1 - attr_fn_male(a)),
        '$\\neg$ C $\land$ E': lambda a: (1 - attr_fn_chubby(a)) * attr_fn_eyeglasses(a),
        '$\\neg$ C $\land$ M': lambda a: (1 - attr_fn_chubby(a)) * attr_fn_male(a),
        '$\\neg$ E $\land$ M': lambda a: (1 - attr_fn_eyeglasses(a)) * attr_fn_male(a),
        '$\\neg$ C $\land$ $\\neg$ E': lambda a: (1 - attr_fn_chubby(a)) * (1 - attr_fn_eyeglasses(a)),
        '$\\neg$ C $\land$ $\\neg$ M': lambda a: (1 - attr_fn_chubby(a)) * (1 - attr_fn_male(a)),
        '$\\neg$ E $\land$ $\\neg$ M': lambda a: (1 - attr_fn_eyeglasses(a)) * (1 - attr_fn_male(a)),
        }  # cf. generate_celeba_audit_table.format_subgroups

CELEBA_SENS_IDX = {
        'C': [13],
        'E': [15],
        'M': [20],
        'C $\land$ E': [13, 15],
        'C $\land$ M': [13, 20],
        'E $\land$ M': [15, 20],
        'C $\land$ $\\neg$ E': [13, 15],
        'C $\land$ $\\neg$ M': [13, 20],
        'E $\land$ $\\neg$ M': [15, 20],
        '$\\neg$ C $\land$ E': [13, 15],
        '$\\neg$ C $\land$ M': [13, 20],
        '$\\neg$ E $\land$ M': [15, 20],
        '$\\neg$ C $\land$ $\\neg$ E': [13, 15],
        '$\\neg$ C $\land$ $\\neg$ M': [13, 20],
        '$\\neg$ E $\land$ $\\neg$ M': [15, 20],
        }  # maps named subgroups to the sensitive indices they depend on 
# comcrime stuff

CC_ATTR_STRING = 'cc_attr_fn'
def create_cc_attr_fn(i):
    def f(y):
        # print('column', i)
        return y[:, i] #>= 0.5 - should be already binarized
    return f

cc_attr_fn_0 = create_cc_attr_fn(0)
cc_attr_fn_1 = create_cc_attr_fn(1)
cc_attr_fn_2 = create_cc_attr_fn(2)
cc_attr_fn_3 = create_cc_attr_fn(3)
cc_attr_fn_4 = create_cc_attr_fn(4)
cc_attr_fn_5 = create_cc_attr_fn(5)
cc_attr_fn_6 = create_cc_attr_fn(6)
cc_attr_fn_7 = create_cc_attr_fn(7)
cc_attr_fn_8 = create_cc_attr_fn(8)
cc_attr_fn_9 = create_cc_attr_fn(9)
cc_attr_fn_10 = create_cc_attr_fn(10)
cc_attr_fn_11 = create_cc_attr_fn(11)
cc_attr_fn_12 = create_cc_attr_fn(12)
cc_attr_fn_13 = create_cc_attr_fn(13)
cc_attr_fn_14 = create_cc_attr_fn(14)
cc_attr_fn_15 = create_cc_attr_fn(15)
cc_attr_fn_16 = create_cc_attr_fn(16)
cc_attr_fn_17 = create_cc_attr_fn(17)
cc_attr_fn_18 = create_cc_attr_fn(18)

if __name__ == '__main__':
    import numpy as np
    x = np.zeros((10, 10))
    print('should print 5')
    cc_attr_fn_5(x)
    cc_attr_fn_6(x)
    cc_attr_fn_7(x)
