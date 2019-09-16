import torch
import os
from collections import OrderedDict
import copy

# Default parameter decay=0.9999
# This value is used in QANet
# https://arxiv.org/pdf/1804.09541.pdf  Training Details.   Training also involved learning rate warm-up.


class EMA(object):    # This is a EMA tracker for the model. Remember this is not the model.
    def __init__(self, model, parameters, decay=0.9999, device_num=-1, warmup=True):
        self.decay = decay
        self.steps = 0
        self.shadow = OrderedDict()
        if device_num == -1:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{device_num}")
        self.warmup = warmup

        for _name, _parameter in parameters:
            if _parameter.requires_grad:
                self.shadow[_name] = _parameter.detach().clone().to(self.device)

        self.back_up_model = None
        self.set_back_up_model(model)

    def set_back_up_model(self, model):
        self.back_up_model = copy.deepcopy(model)
        self.back_up_model = self.back_up_model.to(self.device)

    def __call__(self, parameters):
        self.steps += 1
        if self.warmup:
            decay = min((self.steps + 1) / (10 + self.steps), self.decay)
        else:
            decay = self.decay

        for _name, _parameter in parameters:
            with torch.no_grad():
                if _parameter.requires_grad:
                    self.shadow[_name].add_((1. - decay) * (_parameter.data.to(self.device) - self.shadow[_name]))
                    # new_average = (1.0 - decay) * _parameter.detach().clone() + decay * self.shadow[_name]
                    # self.shadow[_name] = new_average
        return self.shadow

    def get_inference_model(self, model_instance=None):
        if model_instance is None:
            model_instance = self.back_up_model

        state_dict = model_instance.state_dict()
        state_dict.update(self.shadow)
        model_instance.load_state_dict(state_dict)
        return model_instance

    def load_ema_state_dict(self, state_dict, back_up_model_instance=None):
        self.shadow = state_dict
        if back_up_model_instance is not None:
            self.set_back_up_model(back_up_model_instance)

    @classmethod
    def save_ema_to_file(cls, ema_model, filename):
        torch.save(ema_model.shadow, filename)

    @classmethod
    def load_ema_to_model(cls, model, ema_tracker_instance):    # instance is a file or a EMA object
        if not isinstance(ema_tracker_instance, EMA):
            # If ema_model is a filename
            ema_shadow = torch.load(ema_tracker_instance)
        else:
            ema_shadow = ema_tracker_instance.shadow

        state_dict = model.state_dict()
        state_dict.update(ema_shadow)
        model.load_state_dict(state_dict)


def get_ema_gpu_id_list(master_device_num=1):
    parallel_id_list = [master_device_num]
    for i in range(torch.cuda.device_count()):
        if i not in parallel_id_list:
            parallel_id_list.append(i)
    return parallel_id_list


class EMAGPU:
    def __init__(self, decays):
        self.decays = list(decays)
        self.devices = []
        for i, decay in enumerate(self.decays):
            if type(decay) is int:
                self.decays[i] = 1. - (1. / decay)
            else:
                self.decays[i] = decay
            self.devices.append(torch.device("cuda:%d" % (i+1)))
        print("Decays in EMA, ", self.decays)
        self.shadow = {}
        self.backup = {}

    def register(self, name, val):
        self.shadow[name] = {}
        self.backup[name] = val.detach().clone().to(self.devices[0])
        for decay, device in zip(self.decays, self.devices):
            self.shadow[name][decay] = val.detach().clone().to(device)

    def update(self, name, x):
        for decay, device in zip(self.decays, self.devices):
            x_gpu = x.data.to(device)
            assert name in self.shadow
            self.shadow[name][decay].add_((1. - decay) * (x_gpu - self.shadow[name][decay]))

    def apply(self, name, x, decay=None):
        if decay is None:
            decay = self.decays[0]
        x.data.copy_(self.shadow[name][decay])

    def make_backup(self, name, x):
        self.backup[name].data.copy_(x.data)

    def recover(self, name, x):
        x.data.copy_(self.backup[name].data)