import torch
import numpy as np

class Evaluator(object):
    def __init__(self, num_classes, gt, mask):
        self.num_classes = num_classes
        self.gt = gt
        self.mask = mask

    def _generate_confusion_matrix(self):
        pass
