"""
This script saves and loads neural network parameters

@author: Kirk Hovell
"""

import os
import tensorflow as tf

from settings import Settings

class Saver:

    def __init__(self, sess, filename):
        self.sess = sess
        self.filename = filename

    def save(self, n_iteration, policy_input, policy_output):
        # Save all the tensorflow parameters from this session into a file
        # The file is saved to the directory Settings.MODEL_SAVE_DIRECTORY.
        # It uses the n_iteration in the file name
        if Settings.ENVIRONMENT != 'fixedICs':            
            print("Saving neural networks at iteration number " + str(n_iteration) + "...")
    
            os.makedirs(os.path.dirname(Settings.MODEL_SAVE_DIRECTORY + self.filename), exist_ok = True)
            self.saver.save(self.sess, Settings.MODEL_SAVE_DIRECTORY + self.filename + "/Iteration_" + str(n_iteration) + ".ckpt")
        else:
            print("Skipping saving the networks since we are simulating initial conditions")

    def load(self):
        # Try to load in weights to the networks in the current Session.
        # If it fails, or we don't want to load (Settings.RESUME_TRAINING = False)
        # then we start from scratch

        self.saver = tf.train.Saver(max_to_keep = Settings.NUM_CHECKPOINT_MODELS_TO_SAVE) # initialize the tensorflow Saver()

        if Settings.RESUME_TRAINING:
            print("\nAttempting to load in the most recent previously-trained model")
            try:
                # Finding the most recent checkpoint file
                most_recent_checkpoint_filename = [i for i in sorted(os.listdir('..')) if i.endswith('.index')][-1].rsplit('.',1)[0]
                self.saver.restore(self.sess, '../' + most_recent_checkpoint_filename)
                print("Model successfully loaded!\n")
                return True

            except (ValueError, AttributeError):
                print("Model: ", most_recent_checkpoint_filename, " not found... :(")
                return False
        else:
            return False

    def initialize(self):
        self.saver = tf.train.Saver(max_to_keep = Settings.NUM_CHECKPOINT_MODELS_TO_SAVE) # initialize the tensorflow Saver() without trying to load in parameters