from collections import OrderedDict


def weights_to_v1(old_weights: OrderedDict):
    new_weights = OrderedDict()
    new_weights['0.encoder.weight']=old_weights['0.encoder.weight']
    new_weights['0.encoder_dp.emb.weight']=old_weights['0.encoder_with_dropout.embed.weight']
    new_weights['0.rnns.0.weight_hh_l0_raw']=old_weights['0.rnns.0.module.weight_hh_l0_raw']
    new_weights['0.rnns.0.module.weight_ih_l0']=old_weights['0.rnns.0.module.weight_ih_l0']
    new_weights['0.rnns.0.module.weight_hh_l0']=old_weights['0.rnns.0.module.weight_hh_l0_raw']
    new_weights['0.rnns.0.module.bias_ih_l0']=old_weights['0.rnns.0.module.bias_ih_l0']
    new_weights['0.rnns.0.module.bias_hh_l0']=old_weights['0.rnns.0.module.bias_hh_l0']
    new_weights['0.rnns.1.weight_hh_l0_raw']=old_weights['0.rnns.1.module.weight_hh_l0_raw']
    new_weights['0.rnns.1.module.weight_ih_l0']=old_weights['0.rnns.1.module.weight_ih_l0']
    new_weights['0.rnns.1.module.weight_hh_l0']=old_weights['0.rnns.1.module.weight_hh_l0_raw']
    new_weights['0.rnns.1.module.bias_ih_l0']=old_weights['0.rnns.1.module.bias_ih_l0']
    new_weights['0.rnns.1.module.bias_hh_l0']=old_weights['0.rnns.1.module.bias_hh_l0']
    new_weights['0.rnns.2.weight_hh_l0_raw']=old_weights['0.rnns.2.module.weight_hh_l0_raw']
    new_weights['0.rnns.2.module.weight_ih_l0']=old_weights['0.rnns.2.module.weight_ih_l0']
    new_weights['0.rnns.2.module.weight_hh_l0']=old_weights['0.rnns.2.module.weight_hh_l0_raw']
    new_weights['0.rnns.2.module.bias_ih_l0']=old_weights['0.rnns.2.module.bias_ih_l0']
    new_weights['0.rnns.2.module.bias_hh_l0']=old_weights['0.rnns.2.module.bias_hh_l0']
    new_weights['1.decoder.weight'] = old_weights['1.decoder.weight']
    return new_weights


def pth_to_torch(old_weights: OrderedDict):
    return OrderedDict(old_weights['model'])
