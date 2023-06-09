import sys
sys.path.append(".")
sys.path.append('./taming-transformers')
from taming.models import vqgan
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler_collectQuantError

import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid

from quant_scripts.brecq_quant_model import QuantModel
from quant_scripts.brecq_quant_layer import QuantModule
from quant_scripts.brecq_adaptive_rounding import AdaRoundQuantizer

n_bits_w = 4
n_bits_a = 8

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
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

if __name__ == '__main__':
    fp_model = get_model()
    model = get_model()
    dmodel = model.model.diffusion_model
    dmodel.cuda()
    dmodel.eval()
    from quant_scripts.quant_dataset import DiffusionInputDataset
    from torch.utils.data import DataLoader

    dataset = DiffusionInputDataset('imagenet_input_20steps.pth')
    data_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True) ## each sample is (16,4,32,32)
    
    wq_params = {'n_bits': n_bits_w, 'channel_wise': False, 'scale_method': 'mse'}
    aq_params = {'n_bits': n_bits_a, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': True}
    qnn = QuantModel(model=dmodel, weight_quant_params=wq_params, act_quant_params=aq_params)
    qnn.cuda()
    qnn.eval()

    print('Setting the first and the last layer to 8-bit')
    qnn.set_first_last_layer_to_8bit()
    # Disable output quantization because network output
    # does not get involved in further computation
    qnn.disable_network_output_quantization()

    cali_images, cali_t, cali_y = get_train_samples(data_loader, num_samples=1024)
    device = next(qnn.parameters()).device
    # Initialize weight quantization parameters
    qnn.set_quant_state(True, True)

    print('First run to init model...')
    with torch.no_grad():
        _ = qnn(cali_images[:32].to(device),cali_t[:32].to(device),cali_y[:32].to(device))
        
    # Start calibration
    for name, module in qnn.named_modules():
        if isinstance(module, QuantModule) and module.ignore_reconstruction is False:
            module.weight_quantizer.soft_targets = False
            module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode='learned_hard_sigmoid', weight_tensor=module.org_weight.data)

    ckpt = torch.load('quantw{}a{}_ldm_brecq.pth'.format(n_bits_w, n_bits_a), map_location='cpu')
    qnn.load_state_dict(ckpt)
    qnn.cuda()
    qnn.eval()

    setattr(model.model, 'diffusion_model', qnn)

    sampler = DDIMSampler_collectQuantError(fp_model, model)


    classes = torch.randint(0, 1000, size=(1024,))
    n_samples_per_class = 1

    ddim_steps = 20
    ddim_eta = 0.0
    scale = 3.0


    all_samples = list()

    with torch.no_grad():
        with model.ema_scope():
            uc = model.get_learned_conditioning(
                {model.cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.device)}
                )
            
            for class_label in classes:
                print(f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
                xc = torch.tensor(n_samples_per_class*[class_label])
                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                conditioning=c,
                                                batch_size=n_samples_per_class,
                                                shape=[3, 64, 64],
                                                verbose=False,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc, 
                                                eta=ddim_eta)

    import ldm.globalvar as globalvar
    data_error_t = globalvar.getList()
    torch.save(data_error_t, 'data_error_t_w{}a{}_scale{}_eta{}_step{}.pth'.format(n_bits_w, n_bits_a, scale, ddim_eta, ddim_steps))


