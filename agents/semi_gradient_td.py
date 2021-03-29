import torch
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger('experiment')

def train_td(env, model, loss, optimizer, device, cfg, my_experiment):
    state = env.reset()
    obsall = np.zeros((cfg['N_TRAIN_STEPS'], state.observation.shape[0],))
    predall = np.zeros((cfg['N_TRAIN_STEPS']))
    t = -1
    # take 100 steps at start
    results_list = []
    global_step = 0;
    for i in range(cfg['INITIAL_STEPS']):
        t += 1
        step = env.step(None)
        obsall[t] = step.observation
        predall[t] = 0
        # results_list.append([0, 0, global_step, cfg["rank"]])
        global_step+=1

    running_error = 0
    for i in range(cfg['N_TRAIN_STEPS'] - cfg['INITIAL_STEPS']):
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
        optimizer.zero_grad()
        td_error.backward()
        optimizer.step()
        #
        if i % 5000 == 0:
            logger.info("TD Error at t:%d = %f", i, running_error)
            keys = ["TD_error", "V", "step", "run"]
            if len(results_list) > 0:
                my_experiment.insert_values("metrics", keys, results_list)
                results_list = []
            # print(f'TD Error at t:{i} = {running_error}')
        running_error = running_error*0.99 + td_error.item()*0.01
        # Store the value prediction belonging to o_t
        predall[t] = V_tp1[-1].detach().item()
        if i % 100 == 0:
            results_list.append([running_error, predall[t], global_step, cfg["run"]])
        global_step += 1
        # results_list.append([td_error.item(), predall[t], global_step  ,cfg["rank"]])

    if len(results_list) > 0:
        my_experiment.insert_values("metrics", keys, results_list)
        results_list = []
    return obsall, predall, results_list
