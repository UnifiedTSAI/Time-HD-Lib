import os
import torch
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, TimeXer, MultiPatchFormer,\
    ModernTCN, CCM, UCast, PDF, DUET

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            "SCINet": SCINet,
            'PAttn': PAttn,
            'TimeXer': TimeXer,
            'MultiPatchFormer': MultiPatchFormer,
            'ModernTCN': ModernTCN,
            'CCM': CCM,
            'UCast': UCast,
            'PDF': PDF,
            'DUET': DUET,
        }
        if args.model == 'Mamba':
            # print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        # Initialize Accelerator; fp16 mode is enabled if self.args.use_amp is True.
        self.accelerator = args.accelerator
        self.device = self.accelerator.device
        
        # Print all devices that Accelerator is using
        if torch.cuda.is_available():
            self.accelerator.print("\n=== Accelerator Device Information ===")
            self.accelerator.print(f"Number of processes: {self.accelerator.num_processes}")
            self.accelerator.print(f"Distributed type: {self.accelerator.distributed_type}")
            self.accelerator.print("\nDevice details for all processes:")
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                self.accelerator.print(f"  GPU #{i}: {device_props.name} - Total memory: {device_props.total_memory / 1024**3:.2f} GB")
            self.accelerator.print("=======================================\n")
        
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError
        return None

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
