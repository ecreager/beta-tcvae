import random


from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import ModelCheckpoint, Timer, EarlyStopping
from ignite.metrics import RunningAverage

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

import attr_functions
import label_functions
import repr_functions
from metric_helpers.fair_epoch_metrics import DeltaDP



def check_manual_seed(seed):
    """ 
    If manual seed is not specified, choose a random one and communicate it to the user.
    """

    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    print('Using manual seed: {seed}'.format(seed=seed))


def calculate_delta_dp(yhat, a):
    avg_yhat_a0 = torch.mean(yhat[1 - a].float())
    avg_yhat_a1 = torch.mean(yhat[a].float())
    delta_dp = abs(avg_yhat_a0 - avg_yhat_a1)
    if not torch.isnan(delta_dp):
        return delta_dp
    else:
        return avg_yhat_a0 if not torch.isnan(avg_yhat_a0) else avg_yhat_a1


class MLPClassifier(torch.nn.Module): 
    def __init__(self, n, h, c):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
                nn.Linear(n, h),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(h, int(h / 2)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(int(h / 2), c)
                )

    def forward(self, x, y, a):
        logits = self.model(x)
        logprobs = nn.LogSoftmax(dim=1)(logits)

        criterion = torch.nn.NLLLoss()
        loss = criterion(logprobs, y)

        _, predicted_classes = torch.max(logprobs, 1)
        accuracy = (predicted_classes == y).float().mean()
        delta_dp = calculate_delta_dp(predicted_classes, a)

        probs = torch.exp(logprobs)
        avg_pred = torch.mean(probs, dim=0)[1]

        metrics = {'loss': loss,
                   'acc': accuracy,
                   'est_delta_dp': delta_dp,
                   'avg_pred': avg_pred,
                   'avg_class': torch.mean(predicted_classes.float()),
                   'ypred': predicted_classes,
                   'y': y,
                   'a': a
                  }

        return loss, logprobs, metrics

# from configs/main_autoencoder_from_checkpoint_celeba.json

# TODO(creager): figure out what to do about repr_loaders; could deal with this
# in vae_quant or lib/datasets

def get_device(use_cuda):
    return 'cuda:0' if (torch.cuda.is_available() and use_cuda) else 'cpu' 

def get_label_fn(config):
    if not 'celeba' in config['data']['name']:
        raise ValueError('Only Celeb-A supported')
    try:  
        return attr_functions.CELEBA_SUBGROUPS[config['data']['label_fn']]
    except e:
        assert 'label_fn' in config['data'].keys(), 'please specify an label_fn'
        raise ValueError(
                'failed to retrive label or idx for specified label_fn: {}'
                .format(config['data']['label_fn']))


def get_attr_fn(config):
    if not 'celeba' in config['data']['name']:
        raise ValueError('Only Celeb-A supported')
    try:  
        return attr_functions.CELEBA_SUBGROUPS[config['data']['attr_fn']]
    except e:
        assert 'attr_fn' in config['data'].keys(), 'please specify an attr_fn'
        raise ValueError(
                'failed to retrive label or idx for specified attr_fn: {}'
                .format(config['data']['attr_fn']))

def get_repr_fn(config):
    if config['data']['repr_fn'] == 'do_nothing':
        return repr_functions.do_nothing
    else:
        sens_idx = attr_functions.CELEBA_SENS_IDX[config['data']['attr_fn']]
        return repr_functions.get_remove_numbered_dimension_fn(sens_idx)

def audit(vae, loaders, attr_fn_name, latent_dim, samps):
    if not vae.beta_sens > 0.:
        raise ValueError('Auditing only supported for SensVAE')
    vae.eval()

    config = \
        {
                "optim": {
                        "optimizer": {
                                "name": "adam",
                                "lr": 0.001,
                                "beta1": 0.9,
                                "beta2": 0.99
                        },
                        "patience": 5,
                        "epochs": 1,
                        "batch_size": 256
                },
                "model": {
                        "name": "mlpclassifier",
                        "hidden_dim": 1000,
                        "num_classes": 2
                },
                "data": {
                        "name": "celeba_test",
                        "samps": samps,
                        "ckpt": "/scratch/gobi1/creager/fair-subspaces/ae_train_factorsensvae4_celeba_20190116_16-57-12-937344/networks_model_199-copy.pth",
                        "attr_fn": attr_fn_name,
                        "input_size": latent_dim,
                        "use_x": False,
                        "use_y": False,
                        "label_fn": "H",
                        "repr_fn": "remove_all",
                        "overwrite_results_json": False,
                        "num_workers": 4,
                        "validation": True
                },
                "meters": {
                        "alpha": 0.98
                },
                "json_model_key": "FFVAE (Ours)",
                "clf_problem": True,
                "name": "temp",
                "cuda": True,
                "plot": False,
                "port": 8097,
                "output_root": "/scratch/gobi1/creager/fair-subspaces",
                "seed": None
        }

    # build auditor model
    input_size = config['data']['input_size']
    hidden_dim = config['model']['hidden_dim']
    num_classes = config['model']['num_classes']
    model = MLPClassifier(input_size, hidden_dim, num_classes)
    model.to(get_device(config['cuda']))

    # get helper functions
    repr_fn = get_repr_fn(config)
    attr_fn = get_attr_fn(config)
    label_fn = get_label_fn(config)

#    zs, z_params = vae.encode(tensor)
#    if config['data']['samps']:
#        code = zs
#    else:
#        code = z_params
#    data_loader = DataLoader(data_loader)


    #some bookkeeping
    device = get_device(config['cuda'])
    
    #rename loaders for easier access
    train_loader = loaders['train']
    valid_loader = loaders['validation']
    test_loader = loaders['test']

    # check manual seed
    check_manual_seed(config['seed'])
    print('seed', config['seed'])

    # 6) build optimizer
    model_params = model.parameters()
    optimizer = torch.optim.Adam(model_params, \
            lr=config['optim']['optimizer']['lr'])

    monitoring_tensors = ['ypred', 'y', 'a']    
    # 7) define step function
    def step(engine, batch):
        if model:
            model.train()
        return_dict = {}
        x, a_all = batch
        # TODO(creager): make sure modifications from load_reprs are respected
        #a = a.float()
        label_fn = get_label_fn(config)
        y = label_fn(a_all.clone()).long()
        attr_fn = get_attr_fn(config)
        a = attr_fn(a_all)
        #y = y.float().unsqueeze(-1)
        #y = y.long().unsqueeze(-1)
        #y = y.long().unsqueeze(-1)
        x, y, a = x.to(device), y.to(device), a.to(device)

        # compute latent code by encoding images
        with torch.no_grad():
            zs, z_params = vae.encode(x)
            if config['data']['samps']:
                z = zs
            else:
                z = z_params

        # noise out sensitive dims of latent code
        z = repr_fn(z, None, None)

        loss, _, metrics = model(z, y, a)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return_dict.update(loss=loss.detach().item())
        for k, v in metrics.items():  # hopefully this works...
            if not k in monitoring_tensors:
                return_dict.update({k:v.detach().item()})
            else:
                return_dict.update({k:v.detach()})

        return return_dict

    # 8) ignite objects 
    trainer = Engine(step)
    timer = Timer(average=True)

    # attach running average metrics
    monitoring_metrics = ['loss', 'acc', 'est_delta_dp', 'avg_pred', 'avg_class'] \
                if config['clf_problem'] else ['loss']
    val_monitoring_metrics = ['val_{}'.format(m) for m in monitoring_metrics]
    for metric in monitoring_metrics:
        RunningAverage(alpha=config['meters']['alpha'],
                output_transform=(lambda m: lambda x: x[m])(metric)
                ).attach(trainer, metric)
    delta_dp_name = 'delta_dp'
    monitoring_metrics_no_run_avg = [delta_dp_name]
    val_monitoring_metrics_no_run_avg = ['val_{}'.format(m) \
                                for m in monitoring_metrics_no_run_avg]

    # attach progress bar
    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=monitoring_metrics)

    def create_my_evaluator(model, device=None):
        '''Hacky, mostly copying code for create_supervised_evaluator.'''
        if device:
            model.to(device)

        def _step(engine, batch):
            return_dict = {}
            if 'celeba' in config['data']['name']:
                x, a_all = batch
                #a = a.float()
                label_fn = get_label_fn(config)
                y = label_fn(a_all.clone()).long()
                attr_fn = get_attr_fn(config)
                a = attr_fn(a_all)
                #y = y.float().unsqueeze(-1)
                #y = y.long().unsqueeze(-1)
                #y = y.long().unsqueeze(-1)
            else:
                x, y, a = batch 
            x, y, a = x.to(device), y.to(device), a.to(device)

            # compute latent code by encoding images
            with torch.no_grad():
                zs, z_params = vae.encode(x)
                if config['data']['samps']:
                    z = zs
                else:
                    z = z_params

            # noise out sensitive dims of latent code
            z = repr_fn(z, None, None)

            model.eval()
            with torch.no_grad():
                loss, _, metrics = model(z, y, a)
                return_dict.update(loss=loss.detach().item())
                for k, v in metrics.items():  # hopefully this works...
                    if not k in monitoring_tensors:
                        return_dict.update({'val_{}'.format(k):v.detach().item()})
                    else:
                        return_dict.update({'{}'.format(k):v.detach()})
                return return_dict

        engine = Engine(_step)
        # note: defining RunningAverage within a for-loop doesn't work...
        for metric in val_monitoring_metrics:
            RunningAverage(alpha=config['meters']['alpha'],
                    output_transform=(lambda m: lambda x: x[m])(metric)
                    ).attach(engine, metric)
        DeltaDP().attach(engine, delta_dp_name)
        return engine

    #early stopping
    evaluator = create_my_evaluator(model, device)
    pbar.attach(evaluator, metric_names=val_monitoring_metrics)

    def score_function(engine):
        val_loss = engine.state.metrics['val_loss']
        return -val_loss

    handler = EarlyStopping(patience=config['optim']['patience'],
                            score_function=score_function, \
                            trainer=trainer)
    # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset)
    evaluator.add_event_handler(Events.COMPLETED, handler)

#    def do_printing(eng, log_fn, loader):
#        fname = os.path.join(output_dir, log_fn)
#        columns = eng.state.metrics.keys()
#        print('metrics', eng.state.metrics)
#        values = [str(round(v, 5)) for v in eng.state.metrics.values()]
#
#        #if log_fn == LOGS_FNAME_VALID:
#            #import pdb
#            #pdb.set_trace()
#
#        with open(fname, 'a') as f:
#            if f.tell() == 0:
#                f.write('\t'.join(columns))
#            f.write('\t'.join(values))
#
#        message = '[{epoch}/{max_epoch}][{i}/{max_i}]'.format(
#                epoch=eng.state.epoch,
#                max_epoch=config['optim']['epochs'],
#                i=(eng.state.iteration % len(loader)),
#                max_i=len(loader))
#        for name, value in zip(columns, values):
#            message += ' | {name}: {value}'.format(name=name, value=value)
#
#        pbar.log_message(message)

#    # TODO(creager): look through do_printing; grab code for returning metrics
#    @trainer.on(Events.ITERATION_COMPLETED)
#    def print_logs_train(engine):
#        if (trainer.state.iteration - 1) % PRINT_FREQ == 0:
#            do_printing(trainer, LOGS_FNAME, train_loader)
#
#    @trainer.on(Events.EPOCH_COMPLETED)
#    def print_logs_valid(trainer):
#        evaluator.run(valid_loader)
#        do_printing(evaluator, LOGS_FNAME_VALID, valid_loader)
#
#    @trainer.on(Events.COMPLETED)
#    def print_logs_test(trainer):
#        #do I want to reload an old model first?
#        evaluator.run(test_loader)
#        do_printing(evaluator, LOGS_FNAME_TEST, test_loader)

    # automatically adding handlers via a special `attach` method of `Timer` handler
    timer.attach(trainer, start=Events.EPOCH_STARTED,
            resume=Events.ITERATION_STARTED, pause=Events.ITERATION_COMPLETED,
            step=Events.ITERATION_COMPLETED)

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        message_fmt = 'Epoch {} done. Time per batch: {:.3f}[s]'
        pbar.log_message(message_fmt.format(engine.state.epoch, timer.value()))
        timer.reset()

    @trainer.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine, e):
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            engine.terminate()
            warnings.warn('KeyboardInterrupt caught. Exiting gracefully.')

            if 'plot' in config.keys() and config['plot']:
                plot_learning_curves(engine)

        else:
            raise e

    # 9) run trainer
    train_state = trainer.run(train_loader, config['optim']['epochs'])
    validation_state = evaluator.run(valid_loader, 1)
    validation_state.metrics['val_delta_dp'] \
            = validation_state.metrics.pop('delta_dp')
    return {**train_state.metrics, **validation_state.metrics}

