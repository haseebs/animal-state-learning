import configargparse

class Parser(configargparse.ArgParser):
    def __init__(self):
        """
        #
        Returns:
            object:
        """
        super().__init__()


        self.add('--gpus', type=int, help='epoch number', default=1)
        self.add('--name', help='Name of experiment', default="oml_regression")
        self.add('--output-dir', help='Name of experiment', default="../results/")
        self.add('--seed', nargs='+', help='Seed', default=[90], type=int)
        self.add('--run', type=int, help='meta batch size, namely task num', default=0)
        self.add("--HIDDEN_L_SIZE", nargs='+', type=int, default=[10, 20, 40])
        self.add("--STEP_SIZE", nargs='+', type=float, default=[1e-4])
        self.add("--TBPTT_T", nargs='+', type=int, default=[20])
        self.add("--NUM_CS", nargs='+', type=int, default=[1])
        self.add("--NUM_US", nargs='+', type=int, default=[1])
        self.add("--NUM_DIST", nargs='+', type=int, default=[10])
        self.add("--ISI_interval", nargs='+', type=str, default="7,13")
        self.add("--ITI_interval", nargs='+', type=str, default="80,120")
        self.add("--LEN_CS", nargs='+', type=int, default=[4])
        self.add("--LEN_US", nargs='+', type=int, default=[2])
        self.add("--LEN_DIST", nargs='+', type=int, default=[4])
        self.add("--ADAMB1", nargs='+', type=int, default=[0.9])
        self.add("--ADAMB2", nargs='+', type=int, default=[0.999])
        self.add("--ADAME", nargs='+', type=int, default=[1e-8])
        self.add("--LAMBDA", nargs='+', type=float, default=[0])
        self.add("--N_TRAIN_STEPS", nargs='+', type=int, default=[2000000])
        self.add("--INITIAL_STEPS", nargs='+', help="Number of env steps to take and store before starting training", type=int, default=[100])
        # self.add("--GAMMA", nargs='+', type=int, help="This is set in train.py", default=[0])

