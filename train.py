import random

import torch
import numpy as np

from models.lstm import LSTM
from utils.plotter import plot_last_n
from agents.semi_gradient_td import train_td
from env.classical_conditioning_benchmarks import TraceConditioning, compute_return_error

cfg = {
    'SEED' : 0,
    'USE_GPU': True,
    'NUM_CS' : 1,
    'NUM_US' : 1,
    'NUM_DIST' : 10,
    'ISI_interval' : (7,13),
    'ITI_interval' : (80,120),
    'LEN_CS' : 4,
    'LEN_US' : 2,
    'LEN_DIST' : 4,
    'ADAMB1' : 0.9,
    'ADAMB2' : 0.999,
    'ADAME' : 10e-8,
    'LAMBDA' : 0,
    'N_TRAIN_STEPS' : 1000,
    'TBPTT_T' : 10,
    'HIDDEN_L_SIZE' : 10,
    'STEP_SIZE' : 1e-4,
    'INITIAL_STEPS': 100,
}
cfg['GAMMA'] = 1-1/np.mean(cfg['ISI_interval'])

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
                             lr=cfg['STEP_SIZE'],
                             betas=[cfg['ADAMB1'],
                                    cfg['ADAMB2']],
                             eps=cfg['ADAME'])

obsall, predall = train_td(tc, model, loss, optimizer, device, cfg)
errors = compute_return_error(obsall[:,0], predall, cfg['GAMMA'])
fig = plot_last_n(obsall, predall, errors, n=500, nobs=cfg['NUM_CS'] + cfg['NUM_US'] + cfg['NUM_DIST'])
fig.savefig('latest.pdf')
