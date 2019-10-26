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
    new_weights = OrderedDict(old_weights)
    new_weights['0.encoder.weight']=old_weights['model']['0.encoder.weight']
    new_weights['0.encoder_dp.emb.weight']=old_weights['model']['0.encoder_dp.emb.weight']
    try:
        i = 0
        while True:
            new_weights[f'0.rnns.{i}.weight_hh_l0_raw']=old_weights['model'][f'0.rnns.{i}.weight_hh_l0_raw']
            new_weights[f'0.rnns.{i}.module.weight_ih_l0']=old_weights['model'][f'0.rnns.{i}.module.weight_ih_l0']
            new_weights[f'0.rnns.{i}.module.weight_hh_l0']=old_weights['model'][f'0.rnns.{i}.module.weight_hh_l0']
            new_weights[f'0.rnns.{i}.module.bias_ih_l0']=old_weights['model'][f'0.rnns.{i}.module.bias_ih_l0']
            new_weights[f'0.rnns.{i}.module.bias_hh_l0']=old_weights['model'][f'0.rnns.{i}.module.bias_hh_l0']
            i += 1
    except KeyError:
        pass

    new_weights['1.decoder.weight'] = old_weights['model']['1.decoder.weight']
    new_weights['1.decoder.bias'] = old_weights['model']['1.decoder.bias']
    del new_weights['model']
    del new_weights['opt']
    return new_weights