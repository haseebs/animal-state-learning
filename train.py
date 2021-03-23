import random

import torch
import configs.animal_learning as reg_parser
from experiment.experiment import experiment
import numpy as np

from models.lstm import LSTM
from utils.plotter import plot_last_n
from agents.semi_gradient_td import train_td
from env.classical_conditioning_benchmarks import TraceConditioning, compute_return_error
import utils
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger('experiment')




p = reg_parser.Parser()
total_seeds = len(p.parse_known_args()[0].seed)
rank = p.parse_known_args()[0].rank
all_args = vars(p.parse_known_args()[0])

args = utils.get_run(all_args, rank)
args["ISI_interval"] = [int(x) for x in args["ISI_interval"].split(",")]
args["ITI_interval"] = [int(x) for x in args["ITI_interval"].split(",")]
args["GAMMA"] =  1-1/np.mean(args['ISI_interval'])

my_experiment = experiment(args["name"], args, args["output_dir"], sql=True,
                           rank=int(rank / total_seeds),
                           seed=total_seeds)


my_experiment.make_table("metrics", {"rank": 0, "TD_error": 0.0, "step": 0, "V(t)":0.0}, ("rank", "step"))
my_experiment.make_table("msre", {"rank": 0, "msre": 0.0}, ("rank"))

my_experiment.results["all_args"] = all_args
utils.set_seed(args["seed"])

gpu_to_use = rank % args["gpus"]
if torch.cuda.is_available():
    device = torch.device('cuda:' + str(gpu_to_use))
    logger.info("Using gpu : %s", 'cuda:' + str(gpu_to_use))
else:
    device = torch.device('cpu')

tc = TraceConditioning(seed=args['seed'],
                       ISI_interval=args['ISI_interval'],
                       ITI_interval=args['ITI_interval'],
                       gamma=args['GAMMA'],
                       num_distractors=args['NUM_DIST'],
                       activation_lengths={'CS': args['LEN_CS'], 'US': args['LEN_US'], 'distractor': args['LEN_DIST']})

model = LSTM(device, args).to(device)
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=float(args['STEP_SIZE']),
                             betas=(args['ADAMB1'],
                                    args['ADAMB2']),
                             eps=float(args['ADAME']))

obsall, predall, list_of_results = train_td(tc, model, loss, optimizer, device, args)
errors = compute_return_error(obsall[:,0], np.append(predall, 0)[1:], args['GAMMA'])
keys = ["TD_error", "V(t)", "step", "rank"]
my_experiment.insert_values("metrics", keys, list_of_results)

fig = plot_last_n(obsall, np.append(predall, 0)[1:], errors, n=500, nobs=args['NUM_CS'] + args['NUM_US'] + args['NUM_DIST'])
plt.savefig(my_experiment.path + "result.pdf", format="pdf")
