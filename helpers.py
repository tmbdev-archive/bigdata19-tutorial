import time
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from IPython import display
from imp import reload
from torch import nn, optim
from torch.nn import functional as F
from torchmore import layers, flex
import torch
from torchvision import datasets, transforms
from torchvision.datasets import imagenet
import os.path
from torch.utils import data as torchdata


class Timer(object):
    def __init__(self, name=""):
        self.values = []
        self.name = name
    def add(self, x):
        self.values.append(x)
    def value(self):
        return np.mean(self.values) if len(self.values)>0 else -1
    def __truediv__(self, n):
        return self.value() / n
    def __repr__(self):
        value = self.value()
        return f"{self.name}:{value:.2e}"
    def __enter__(self):
        self.start = time.time()
    def __exit__(self, *args):
        self.add(time.time() - self.start)
        self.start = None

        
class Timers(object):
    def __init__(self):
        self.timers = {}
    def __getattr__(self, key):
        if key[0]=="_": raise AttributeError
        return self.timers.setdefault(key, Timer(key))
    def __repr__(self):
        return "\n".join(str(x) for x in self.timers.values())
    

class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.losses = []
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.display_time = 5
    def reset_times(self):
        self.timers = Timers()
    def set_last(self, *args):
        self.last = [x.detach().cpu() for x in args]
    def set_lr(self, lr):
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
    def train_batch(self, images, targets):
        self.optimizer.zero_grad()
        outputs = self.model.forward(images.cuda())
        loss = self.criterion(outputs, targets.cuda())
        loss.backward()
        #nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        self.optimizer.step()
        self.set_last(images, outputs, targets)
        return float(loss)
    def train_for(self, nsamples, loader, quiet=False):
        total = self.losses[-1][0] if len(self.losses)>1 else 0
        last = time.time()
        self.reset_times()
        while total < nsamples:
            src = iter(loader)
            try:
                while total < nsamples:
                    with self.timers.loading:
                        images, classes = next(src)
                    with self.timers.training:
                        l = self.train_batch(images, classes)
                    total += images.size(0)
                    self.losses.append((total, l))
                    if not quiet and time.time() - last > self.display_time:
                        self.update_plot()
                        last = time.time()
            except StopIteration:
                pass
    def update_plot(self):
            if len(self.losses)<10: return
            plt.close("all")
            fig = plt.figure(figsize=(12, 3))
            fig.add_subplot(1, 1, 1)
            ax1, = fig.get_axes()
            ax1.set_title(f"loss (times for {self.timers.loading} sec {self.timers.training} sec)")
            ax1.set_yscale("log")
            #ax1.set_xscale("log")
            losses = np.array(self.losses)
            ax1.plot(losses[:,0][::10], ndi.gaussian_filter(losses[:,1], 10.0, mode="nearest")[::10])
            display.clear_output(wait=True)
            display.display(fig)
