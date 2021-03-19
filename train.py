import random

import torch
import wandb
import numpy as np

from models.lstm import LSTM
from utils.plotter import plot_last_n, upload_to_wandb
from agents.semi_gradient_td import train_td
from env.classical_conditioning_benchmarks import TraceConditioning, compute_return_error

wandb.init(project='animal-state')
wandb.config.update({'GAMMA': 1-1/np.mean(wandb.config['ISI_interval'])},
                    allow_val_change=True)
cfg = wandb.config

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

set_seed(cfg['SEED'])

device = torch.device("cpu")
if torch.cuda.is_available() and cfg['USE_GPU']:
    device = torch.device("cuda:0")

tc = TraceConditioning(seed=cfg['SEED'],
                       ISI_interval=cfg['ISI_interval'],
                       ITI_interval=cfg['ITI_interval'],
                       gamma=cfg['GAMMA'],
                       num_distractors=cfg['NUM_DIST'],
                       activation_lengths={'CS': cfg['LEN_CS'], 'US': cfg['LEN_US'], 'distractor': cfg['LEN_DIST']})

model = LSTM(device, cfg).to(device)
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=float(cfg['STEP_SIZE']),
                             betas=[cfg['ADAMB1'],
                                    cfg['ADAMB2']],
                             eps=float(cfg['ADAME']))

obsall, predall = train_td(tc, model, loss, optimizer, device, cfg)
errors = compute_return_error(obsall[:,0], predall, cfg['GAMMA'])
wandb.run.summary['MSRE'] = errors[0]
upload_to_wandb(obsall, predall, errors, n=min(cfg['N_TRAIN_STEPS'], 1200), nobs=cfg['NUM_CS'] + cfg['NUM_US'] + cfg['NUM_DIST'])
fig = plot_last_n(obsall, predall, errors, n=500, nobs=cfg['NUM_CS'] + cfg['NUM_US'] + cfg['NUM_DIST'])
wandb.log({'plot': wandb.Image(fig)})
#fig.savefig('latest.pdf')
