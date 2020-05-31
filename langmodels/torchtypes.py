from typing import Union

import torch

AnyDeviceLongTensor=Union[torch.cuda.LongTensor, torch.LongTensor]
AnyDeviceFloatTensor=Union[torch.cuda.FloatTensor, torch.FloatTensor]