__all__ = ['PDF']

# Cell
from typing import Optional

from torch import nn
from torch import Tensor
import torch.nn.functional as F
from layers.PDF_backbone import PDF_backbone


class Model(nn.Module):
    def __init__(self, configs, max_seq_len: Optional[int] = 1024, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 norm: str = 'BatchNorm', attn_dropout: float = 0.,
                 act: str = "gelu", key_padding_mask: bool = 'auto', padding_var: Optional[int] = None,
                 attn_mask: Optional[Tensor] = None, res_attention: bool = True,
                 pre_norm: bool = False, store_attn: bool = False, pe: str = 'zeros', learn_pe: bool = True,
                 pretrain_head: bool = False, head_type='flatten', verbose: bool = False, **kwargs):

        super().__init__()

        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len

        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = 0.35
        fc_dropout = 0.15
        head_dropout = 0

        individual = 0

        add = True
        wo_conv = False
        serial_conv = False

        kernel_list = [3, 7, 9, 11]
        patch_len = [1]
        period = [24]
        stride = [1]

        padding_patch = "end"

        revin = 1
        affine = 0
        subtract_last = 0

        # models
        self.model = PDF_backbone(c_in=c_in, context_window=context_window, target_window=target_window,
                                           wo_conv=wo_conv, serial_conv=serial_conv, add=add,
                                           patch_len=patch_len, kernel_list=kernel_list, period=period, stride=stride,
                                           max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                           n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                           attn_dropout=attn_dropout,
                                           dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                           padding_var=padding_var,
                                           attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                           store_attn=store_attn,
                                           pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
                                           head_dropout=head_dropout,
                                           padding_patch=padding_patch,
                                           pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                                           revin=revin, affine=affine,
                                           subtract_last=subtract_last, verbose=verbose, **kwargs)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec):  # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)  # x: [Batch, Channel, Input length]
        x = self.model(x)
        x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]
        return x, 0