from abc import ABCMeta
import torch
import numpy as np

class Base(object):
    """
    Defense base class.
    """


    __metaclass__ = ABCMeta

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def parse_params(self,
                     epsilon = 0.2,
                     epoch_num = 100,
                     lr_train = 0.005,
                     momentum = 0.1,
                     order= np.inf,
                     clip_max= 8/255,
                     clip_min = -8/255):
        """
        Parse user defined parameters
        """
        return True

    def generate_wm(self, train_loader, test_loader, data_adv):
        """generate.
        Parameters
        ----------
        train_loader :
            training data
        test_loader :
            testing data
        kwargs :
            user defined parameters
        """
        self.train_loader = train_loader
        self.test_loader = test_loader
        return 
    
    def test(self, X_wm, test_loader):
        """train.
        Parameters
        ----------
        train_loader :
            training data
        optimizer :
            training optimizer
        epoch :
            training epoch
        """
        return True

    def calculate_loss(self, output, target, redmode = 'mean'):
        """
        Calculate training loss. 
        Overide this function to customize loss.
        
        Parameters
        ----------
        output :
            model outputs
        target :
            true labels
        """
        return True

    def train_model(self, X_wm,data,target):
        return True
    
