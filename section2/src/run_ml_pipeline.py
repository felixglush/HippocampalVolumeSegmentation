"""
This file contains code that will kick off training and testing processes
"""
import os, sys
import argparse
import json
import numpy as np
from experiments.UNetExperiment import UNetExperiment
from data_prep.HippocampusDatasetLoader import LoadHippocampusData
from torch.utils.data import random_split

class Config:
    """
    Holds configuration parameters
    """
    def __init__(self):
        self.name = "Basic_unet"
        self.root_dir = r"data/"
        self.n_epochs = 10
        self.learning_rate = 0.0002
        self.batch_size = 8
        self.patch_size = 64
        self.test_results_dir = "out/results"
        self.model_name = "" # the command line provided model name to save network weights in
        self.weights_name = ""  # the command line provided weights file name to load network weights from 
        self.test = False
        
    def set_model_name(self, m):
        self.model_name = m
        
    def set_weights_name(self, w):
        self.weights_name = w
    
    def set_test(self, t):
        self.test = t
        
if __name__ == "__main__":
    # Get configuration

    # TASK: Fill in parameters of the Config class and specify directory where the data is stored and 
    # directory where results will go
    c = Config()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", "-w", help="file name for saved model weights", action="store")
    parser.add_argument("--modelname", "-m", help="model weights filename used for saving this model", action="store")
    parser.add_argument("--testonly", "-t", help="test only, no training", action="store_true")
    args = parser.parse_args()
    
    if args.weights:
        print("Will load model weights from", args.weights)
        c.set_weights_name(args.weights)
    else:
        print("No pretrained model weights given. Will train a new model.")
        
    if args.modelname:
        print("Will store model weights in", args.modelname)
        c.set_model_name(args.modelname)
    
    if args.testonly:
        # need to also provide a weights filename if we're only testing
        print("Testing mode.")
        c.set_test(True)
        if not args.weights:
            print("Please also provide a weights filename through -w")
            sys.exit()
    
    # Load data
    print("Loading data...")

    # TASK: LoadHippocampusData is not complete. Go to the implementation and complete it. 
    data = LoadHippocampusData(c.root_dir + "TrainingSet/", y_shape = c.patch_size, z_shape = c.patch_size)

    # Create test-train-val split
    # In a real world scenario you would probably do multiple splits for 
    # multi-fold training to improve your model quality
    data_len = len(data)
    keys = range(data_len)
    
    # Here, random permutation of keys array would be useful in case if we do something like 
    # a k-fold training and combining the results. 


    # TASK: create three keys in the dictionary: "train", "val" and "test". In each key, store
    # the array with indices of training volumes to be used for training, validation 
    # and testing respectively.
    
    train_proportion = 0.7
    val_proportion = 0.2
    test_proportion = 0.1
    splits = [int(np.floor(train_proportion * data_len)), 
              int(np.floor(val_proportion * data_len)), 
              int(np.floor(test_proportion * data_len))]
    train, val, test = random_split(keys, splits)
    
    split = {"train": train,
             "val": val, 
             "test": test}
    
    # Set up and run experiment
    
    # TASK: Class UNetExperiment has missing pieces. Go to the file and fill them in
    exp = UNetExperiment(c, split, data)

    # You could free up memory by deleting the dataset
    # as it has been copied into loaders
    del data

    if not args.testonly:
        # run training and validation
        exp.run()

    # prep and run testing

    # TASK: Test method is not complete. Go to the method and complete it
    results_json = exp.run_test()

    results_json["config"] = vars(c)

    with open(os.path.join(exp.out_dir, "results.json"), 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))

