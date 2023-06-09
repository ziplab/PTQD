import sys
sys.path.append(".")
sys.path.append('./taming-transformers')
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from taming.models import vqgan

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.diffusionmodules.openaimodel import ResBlock
from ldm.modules.attention import SpatialTransformer, BasicTransformerBlock
import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid

from quant_scripts.brecq_quant_model import QuantModel
from quant_scripts.brecq_quant_layer import QuantModule
from quant_scripts.brecq_layer_recon import layer_reconstruction
from quant_scripts.brecq_block_recon import block_reconstruction_single_input, block_reconstruction_two_input
from quant_scripts.brecq_adaptive_rounding import AdaRoundQuantizer

from tqdm import tqdm

n_bits_w = 4
n_bits_a = 8

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    # model.cuda()
    model.eval()
    return model

def get_model():
    config = OmegaConf.load("configs/latent-diffusion/cin256-v2.yaml")  
    model = load_model_from_config(config, "models/ldm/cin256-v2/model.ckpt")
    return model

def get_train_samples(train_loader, num_samples):
    image_data, t_data, y_data = [], [], []
    for (image, t, y) in train_loader:
        image_data.append(image)
        t_data.append(t)
        y_data.append(y)
        if len(image_data) >= num_samples:
            break
    return torch.cat(image_data, dim=0)[:num_samples], torch.cat(t_data, dim=0)[:num_samples], torch.cat(y_data, dim=0)[:num_samples]

global cnt 
cnt=0
def count_recon_times(model):
    global cnt
    for name, module in model.named_children():
        if isinstance(module, QuantModule):
            if module.ignore_reconstruction is True:
                print('Ignore reconstruction of layer {}'.format(name))
                continue
            else:
                print('Reconstruction for layer {}'.format(name))
                cnt += 1
                # layer_reconstruction(qnn, module, **kwargs)


        elif isinstance(module, (ResBlock, BasicTransformerBlock)):
            print('Reconstruction for block {}'.format(name))
            cnt += 1
            # block_reconstruction(qnn, module, **kwargs)
        else:
            count_recon_times(module)

if __name__ == '__main__':
    model = get_model()
    model = model.model.diffusion_model
    model.cuda()
    model.eval()
    from quant_scripts.quant_dataset import DiffusionInputDataset
    from torch.utils.data import DataLoader

    dataset = DiffusionInputDataset('imagenet_input_20steps.pth')
    data_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)
    
    wq_params = {'n_bits': n_bits_w, 'channel_wise': False, 'scale_method': 'mse'}
    aq_params = {'n_bits': n_bits_a, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': True}
    qnn = QuantModel(model=model, weight_quant_params=wq_params, act_quant_params=aq_params)
    qnn.cuda()
    qnn.eval()

    print('Setting the first and the last layer to 8-bit')
    qnn.set_first_last_layer_to_8bit()

    cali_images, cali_t, cali_y = get_train_samples(data_loader, num_samples=1024)
    device = next(qnn.parameters()).device
    # Initialize weight quantization parameters
    qnn.set_quant_state(True, False)

    print('First run to init model...')
    with torch.no_grad():
        _ = qnn(cali_images[:32].to(device),cali_t[:32].to(device),cali_y[:32].to(device))

    # Kwargs for weight rounding calibration
    kwargs = dict(cali_images=cali_images, cali_t=cali_t, cali_y=cali_y, iters=10000, weight=0.01, asym=True,
                    b_range=(20, 2), warmup=0.2, act_quant=False, opt_mode='mse', batch_size=32)

    pass_block = 0
    def recon_model(model: nn.Module):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        global pass_block
        for name, module in model.named_children():
            if isinstance(module, (QuantModule)):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    print('Reconstruction for layer {}'.format(name))
                    layer_reconstruction(qnn, module, **kwargs)

            elif isinstance(module, ResBlock):
                pass_block -= 1
                if pass_block < 0 :
                    print('Reconstruction for ResBlock {}'.format(name))
                    block_reconstruction_two_input(qnn, module, **kwargs)
            elif isinstance(module, BasicTransformerBlock):
                pass_block -= 1
                if pass_block < 0 :
                    print('Reconstruction for BasicTransformerBlock {}'.format(name))
                    block_reconstruction_two_input(qnn, module, **kwargs)
            else:
                recon_model(module)
        
    # Start calibration
    for name, module in qnn.named_modules():
        if isinstance(module, QuantModule) and module.ignore_reconstruction is False:
            module.weight_quantizer.soft_targets = False
            module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode='learned_hard_sigmoid', weight_tensor=module.org_weight.data)

    ckpt = torch.load('quantw4_ldm_brecq.pth', map_location='cpu') ## replace first step checkpoint here
    qnn.load_state_dict(ckpt, False)

    qnn.set_quant_state(True, True)
    with torch.no_grad():
        _ = qnn(cali_images[:32].to(device),cali_t[:32].to(device),cali_y[:32].to(device))
    # Disable output quantization because network output
    # does not get involved in further computation
    qnn.disable_network_output_quantization()
    # Kwargs for activation rounding calibration
    kwargs = dict(cali_images=cali_images, cali_t=cali_t, cali_y=cali_y, iters=5000, act_quant=True, opt_mode='mse', lr=4e-4, p=2.4, batch_size=16)
    recon_model(qnn)
    qnn.set_quant_state(weight_quant=True, act_quant=True)
    torch.save(qnn.state_dict(), 'quantw{}a{}_ldm_brecq.pth'.format(n_bits_w, n_bits_a))
