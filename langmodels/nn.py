from torch.nn import Embedding, ModuleList

from typing import List, Tuple, Optional

import torch
from torch import Tensor
from fastai.text import SequentialRNN, AWD_LSTM, WeightDropout, EmbeddingDropout, RNNDropout, one_param, to_detach, \
    Module

TORCH_LONG_MIN_VAL = -2 ** 63


def to_test_mode(model: SequentialRNN) -> None:
    # Set batch size to 1
    model[0].bs = 1
    # Turn off dropout
    model.eval()
    # Reset hidden state
    model.reset()


def save_hidden_states(model: SequentialRNN) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    return [(hl[0].clone(), hl[1].clone()) if isinstance(hl, (tuple, list)) else hl.clone() for hl in model[0].hidden]


def get_last_layer_activations(model: SequentialRNN, input: torch.FloatTensor) -> Optional[torch.FloatTensor]:
    tensor_rank = len(input.size())
    if tensor_rank != 2:
        if tensor_rank == 0:
            error_msg = 'Your input tensor is a scalar. If you want to use batch size = 1 and feed only one token to the model, pass input[None, None] to this method.'
        elif tensor_rank == 1:
            error_msg = 'You input tensor has rank 1. If you were intending to use batch size = 1, please pass input[None, :] to this method. If you however were intending to use multiple batches but feed only one element to the model pass input[:, None]'
        else:
            error_msg = f'Your tensor has rank {tensor_rank}.'

        raise ValueError(f'This method accepts tensors of rank 2. {error_msg}')

    if input.nelement() == 0:
        return None

    last_layer_activations, *_ = model(input)
    return last_layer_activations


class GRU(Module):

    initrange=0.1

    def __init__(self, vocab_sz:int, emb_sz:int, n_hid:int, n_layers:int, pad_token:int=1, hidden_p:float=0.2,
                 input_p:float=0.6, embed_p:float=0.1, weight_p:float=0.5, qrnn:bool=False, bidir:bool=False):
        self.bs,self.qrnn,self.emb_sz,self.n_hid,self.n_layers = 1,qrnn,emb_sz,n_hid,n_layers
        self.n_dir = 2 if bidir else 1
        self.encoder = Embedding(vocab_sz, emb_sz, padding_idx=pad_token)
        self.encoder_dp = EmbeddingDropout(self.encoder, embed_p)

        self.rnns = [torch.nn.GRU(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz)//self.n_dir, 1,
                             batch_first=True, bidirectional=bidir) for l in range(n_layers)]
        self.rnns = [WeightDropout(rnn, weight_p) for rnn in self.rnns]
        self.rnns = ModuleList(self.rnns)
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.input_dp = RNNDropout(input_p)
        self.hidden_dps = ModuleList([RNNDropout(hidden_p) for l in range(n_layers)])

    def forward(self, input:Tensor, from_embeddings:bool=False)->Tuple[Tensor,Tensor]:
        if from_embeddings: bs,sl,es = input.size()
        else: bs,sl = input.size()
        if bs!=self.bs:
            self.bs=bs
            self.reset()
        raw_output = self.input_dp(input if from_embeddings else self.encoder_dp(input))
        new_hidden,raw_outputs,outputs = [],[],[]
        for l, (rnn,hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            raw_output, new_h = rnn(raw_output, self.hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.n_layers - 1: raw_output = hid_dp(raw_output)
            outputs.append(raw_output)
        self.hidden = to_detach(new_hidden, cpu=False)
        return raw_outputs, outputs

    def _one_hidden(self, l:int)->Tensor:
        "Return one hidden state."
        nh = (self.n_hid if l != self.n_layers - 1 else self.emb_sz) // self.n_dir
        return one_param(self).new(self.n_dir, self.bs, nh).zero_()

    def select_hidden(self, idxs):
        self.hidden = [h[:,idxs,:] for h in self.hidden]
        self.bs = len(idxs)

    def reset(self):
        "Reset the hidden states."
        [r.reset() for r in self.rnns if hasattr(r, 'reset')]
        self.hidden = [self._one_hidden(l) for l in range(self.n_layers)]


def add_gru_to_model_data():
    from fastai.text.learner import _model_meta
    gru_meta_data = {k:v for k, v in _model_meta[AWD_LSTM].items()}
    _model_meta[GRU] = gru_meta_data