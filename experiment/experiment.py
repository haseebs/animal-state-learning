import argparse
import datetime
import json
import logging
import os
import subprocess
from logging import handlers
import sqlite3

logger = logging.getLogger('experiment')


# How to use this class (See the main function at the end of this file for an actual example)

## 1. Create an experiment object in the main file (the one used to run the experiment)
## 2. Pass a name and args_parse objecte. Output_dir corresponds to directly where all the results will be stored
## 3. Use experiment.path to get path to the output_dir to store any other results (such as saving the model)
## 4. You can also store results in experiment.result dictionary (Only add objects which are json serializable)
## 5. Call experiment.store_json() to store/update the json file (I just call it periodically in the training loop)

class experiment:
    '''
    Class to create directory and other meta information to store experiment results.
    A directory is created in output_dir/DDMMYYYY/name_0
    In-case there already exists a folder called name, name_1 would be created.

    Race condition:
    '''

    def __init__(self, name, args, output_dir="../", sql=True, rank=None, seed=None):
        import sys
        if name[-1] != "/":
            name += "/"

        self.command_args = "python " + " ".join(sys.argv)
        self.rank = rank
        self.name_initial = name

        if not args is None:
            if rank is not None:
                self.name = name + str(rank) + "/" + str(seed)
            else:
                self.name = name
            self.params = args
            print(self.params)
            self.results = {}
            self.dir = output_dir

            root_folder = datetime.datetime.now().strftime("%d%B%Y")

            if not os.path.exists(output_dir + root_folder):
                try:
                    os.makedirs(output_dir + root_folder)
                except:
                    assert (os.path.exists(output_dir + root_folder))

            self.root_folder = output_dir + root_folder
            full_path = self.root_folder + "/" + self.name

            ver = 0

            while True:
                ver += 1
                if not os.path.exists(full_path + "_" + str(ver)):
                    try:
                        os.makedirs(full_path +  "_" + str(ver))
                        break
                    except:
                        pass
            self.path = full_path + "_" + str(ver) + "/"

            if sql:
                self.database_path =  os.path.join(self.root_folder, self.name_initial, "results.db")

                self.conn = sqlite3.connect(self.database_path, timeout=300)
                self.sql_run = self.conn.cursor()

                ret = self.make_table("runs", args, ["rank"])
                self.insert_value("runs", args)
                self.conn.close()
                if ret:
                    print("Table created")
                else:
                    print("Table already exists")

            fh = logging.FileHandler(self.path + "log.txt")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(
                logging.Formatter('rank:' + str(args['rank']) + ' ' + name + ' %(levelname)-8s %(message)s'))
            logger.addHandler(fh)

            ch = logging.handlers.logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(
                logging.Formatter('rank:' + str(args['rank']) + ' ' + name + ' %(levelname)-8s %(message)s'))
            logger.addHandler(ch)
            logger.setLevel(logging.DEBUG)
            logger.propagate = False



            self.store_json()

    def get_connection(self):
        return

    def is_jsonable(self, x):
        try:
            json.dumps(x)
            return True
        except:
            return False


    def make_table(self, table_name, data_dict, primary_key):

        self.conn = sqlite3.connect(self.database_path, timeout=300)
        self.sql_run = self.conn.cursor()

        table = "CREATE TABLE " + table_name + " ("
        counter = 0
        for a in data_dict:
            if type(data_dict[a]) is int or type(data_dict[a]) is float:
                table = table + a + " real"
            else:
                table = table + a + " text"

            counter += 1
            if counter != len(data_dict):
                table += ", "
        if primary_key is not None:
            table += " ".join([",", "PRIMARY KEY(", ",".join(primary_key)]) + ")"
        table = table + ")"
        print(table)
        try:
            self.sql_run.execute(table)
            self.conn.close()
            return True
        except:
            self.conn.close()
            return False

    def insert_value(self, table_name, data_dict):
        self.conn = sqlite3.connect(self.database_path, timeout=300)
        self.sql_run = self.conn.cursor()
        query = " ".join(["INSERT INTO", table_name,   str(tuple(data_dict.keys())),   "VALUES", str(tuple(data_dict.values()))])
        self.sql_run.execute(query)
        self.conn.commit()
        self.conn.close()

    def insert_values(self, table_name, keys, value_list):
        self.conn = sqlite3.connect(self.database_path, timeout=300)
        self.sql_run = self.conn.cursor()
        strin = "("
        counter = 0
        for a in value_list[0]:
            counter+=1
            strin += "?"
            if counter != len(value_list[0]):
                strin +=","
        strin += ")"

        query = " ".join(
            ["INSERT INTO", table_name, str(tuple(keys)), "VALUES", strin])
        # print(query)
        # print(value_list)
        self.sql_run.executemany(query, value_list)
        self.conn.commit()
        self.conn.close()

    def commit_changes(self):
        self.conn.commit()



    def add_result(self, key, value):
        assert (self.is_jsonable(key))
        assert (self.is_jsonable(value))
        self.results[key] = value

    def store_json(self):
        pass
        # self.conn.commit()
        # with open(self.path + "metadata.json", 'w') as outfile:
        #     json.dump(self.__dict__, outfile, indent=4, separators=(',', ': '), sort_keys=True)
        #     outfile.write("")

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
