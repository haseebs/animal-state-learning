import argparse
import datetime
import json
import logging
import os
import subprocess
from logging import handlers
import mysql.connector
import sys
import time
import random
import json
import sys
import wandb
import numpy as np
from experiment.experiment import experiment




logger = logging.getLogger('experiment')



# How to use this class (See the main function at the end of this file for an actual example)

## 1. Create an experiment object in the main file (the one used to run the experiment)
## 2. Pass a name and args_parse objecte. Output_dir corresponds to directly where all the results will be stored
## 3. Use experiment.path to get path to the output_dir to store any other results (such as saving the model)
## 4. You can also store results in experiment.result dictionary (Only add objects which are json serializable)
## 5. Call experiment.store_json() to store/update the json file (I just call it periodically in the training loop)

class experimentWandb(experiment):
    '''
    Subclass for experiment to use the wandb instead. Why not use this?
    '''

    def __init__(self, project='animal_state_learning', entity='nolife'):
        super().__init__(None, None, None, None, None, None)
        wandb.init(project='animal_state_learning', entity='nolife')

        self.type = 'wandb'
        self.results = {} #backward compt
        self.cfg = wandb.config
        self.run = wandb.run.name
        self.path = wandb.run.dir + '/'

    def make_table(self, table_name, data_dict, primary_key):
        # can use wandb.Table here to get the same behavior as the
        # base class's make_table()
        pass

    def insert_value(self, table_name, data_dict):
        pass

    def insert_values(self, table_name, keys, value_list):
        # shouldnt do this here but I am just making this backwards compatible
        # without having to change the rest of the code

        # Theres only 1 value, if we removed the 'run'
        # this is some hardcoding masterpiece right here
        # this should be in insert_value instead...
        if len(keys)  == 2:
            wandb.run.summary[keys[1]] = value_list[0][1]

        else:
            value_list = np.asarray(value_list)
            steps = value_list[:, keys.index('step')]

            for step_idx, single_step in enumerate(value_list):
                for idx in range(len(keys)):
                    if keys[idx] in ['run', 'step']:
                        continue
                    wandb.log({ keys[idx]: single_step[idx] }, step=int(steps[step_idx]), commit=False)

            #overwrite last update and make them persist
            wandb.log({ keys[0]: single_step[0] }, step=int(steps[step_idx]))

    def add_result(self, key, value):
        pass

    def store_json(self):
        pass

    def get_json(self):
        return json.dumps(self.__dict__, indent=4, sort_keys=True)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='iCarl2.0')
#     parser.add_argument('--batch-size', type=int, default=50, metavar='N',
#                         help='input batch size for training (default: 64)')
#     parser.add_argument('--epochs', type=int, default=200, metavar='N',
#                         help='input batch size for training (default: 64)')
#     parser.add_argument('--epochs2', type=int, default=10, metavar='N',
#                         help='input batch size for training (default: 64)')
#     parser.add_argument('--lrs', type=float, nargs='+', default=[0.00001],
#                         help='learning rate (default: 2.0)')
#     parser.add_argument('--decays', type=float, nargs='+', default=[0.99, 0.97, 0.95],
#                         help='learning rate (default: 2.0)')
#     # Tsdsd
#
#     args = parser.parse_args()
#     e = experiment("TestExperiment", args, "../../")
#     e.add_result("Test Key", "Test Result")
#     e.store_json()
