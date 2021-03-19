import torch
import wandb
import numpy as np
from tqdm import tqdm

def train_td(env, model, loss, optimizer, device, cfg):
    wandb.watch(model, log='all')#, log_freq=1)
    state = env.reset()
    obsall = np.zeros((cfg['N_TRAIN_STEPS'], state.observation.shape[0],))
    predall = np.zeros((cfg['N_TRAIN_STEPS']))
    t = -1
    # take 100 steps at start
    for i in range(cfg['INITIAL_STEPS']):
        t += 1
        step = env.step(None)
        obsall[t] = step.observation
        predall[t] = 0
        wandb.log({'TD Error': 0, 'V(t)': 0, 'MSRE': 1})

    for i in tqdm(range(cfg['N_TRAIN_STEPS'] - cfg['INITIAL_STEPS'])):
        t += 1
        step_new = env.step(None)
        obsall[t] = step_new.observation

        # get V_t-T .... V_t.
        V_t = model(torch.FloatTensor(obsall[t-cfg['TBPTT_T']: t+1]).to(device))

        # get the next step prediction (V_t+1) for each step (V_t)
        V_tp1 = V_t[1:]
        # get the reward obtained (US_t+1) for going to next step
        US_tp1 = torch.tensor(
            obsall[t-cfg['TBPTT_T']+1: t+1][:,0]
        ).view(-1,1).to(device)

        # dont use gradient when calculating the TD target
        # For each value prediction V_t at step t, we use US_t+1 and V_t+1 to calculate TD target
        with torch.no_grad():
            td_target = US_tp1 + cfg['GAMMA'] * V_tp1

        # V_t also includes the prediction for o_t, so we remove it from here.
        # It is only used to calculate the TD target for o_t-1 previously
        td_error = loss(V_t[:-1], td_target.float())
        td_error.backward()
        optimizer.step()

        if i % 5000 == 0:
            print(f'TD Error at t:{i} = {td_error}')
        # Store the value prediction belonging to o_t
        predall[t] = V_tp1[-1].detach().item()

        wandb.log({'TD Error': td_error, 'V(t)': predall[t]})
    #wandb.log({'observations': wandb.Table(data=obsall, columns=list(range(obsall.shape[1])))})
    return obsall, predall
