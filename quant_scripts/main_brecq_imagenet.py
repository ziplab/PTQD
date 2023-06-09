import sys, time
sys.path.append(".")
sys.path.append('./taming-transformers')
from taming.models import vqgan

import torch
torch.cuda.manual_seed(3407)
import torch.nn as nn
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler, DDIMSampler_quantCorrection_imagenet
from ldm.models.diffusion.ddpm import DDPM

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
    pl_sd = torch.load(ckpt, map_location="cpu")
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
    model = get_model()
    dmodel = model.model.diffusion_model
    dmodel.cuda()
    dmodel.eval()
    from quant_scripts.quant_dataset import DiffusionInputDataset
    from torch.utils.data import DataLoader

    dataset = DiffusionInputDataset('imagenet_input_20steps.pth')
    data_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True) 
    
    wq_params = {'n_bits': n_bits_w, 'channel_wise': False, 'scale_method': 'mse'}
    aq_params = {'n_bits': n_bits_a, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': True}
    qnn = QuantModel(model=dmodel, weight_quant_params=wq_params, act_quant_params=aq_params)
    qnn.cuda()
    qnn.eval()

    print('Setting the first and the last layer to 8-bit')
    qnn.set_first_last_layer_to_8bit()

    cali_images, cali_t, cali_y = get_train_samples(data_loader, num_samples=1024)
    device = next(qnn.parameters()).device
    # Initialize weight quantization parameters
    qnn.set_quant_state(True, True)
    # Disable output quantization because network output
    # does not get involved in further computation
    qnn.disable_network_output_quantization()

    print('First run to init model...')
    with torch.no_grad():
        _ = qnn(cali_images[:32].to(device),cali_t[:32].to(device),cali_y[:32].to(device))

    # Kwargs for weight rounding calibration
    kwargs = dict(cali_images=cali_images, cali_t=cali_t, cali_y=cali_y, iters=10000, weight=0.01, asym=True,
                    b_range=(20, 2), warmup=0.2, act_quant=False, opt_mode='mse', batch_size=8)
        
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

    sampler = DDIMSampler_quantCorrection_imagenet(model, num_bit=4, correct=True)


    classes = [25, 187, 448, 992]   # define classes to be sampled here
    n_samples_per_class = 6

    ## Quality, sampling speed and diversity are best controlled via the `scale`, `ddim_steps` and `ddim_eta` variables
    ddim_steps = 20
    ddim_eta = 0.0
    scale = 3.0   # for  guidance


    all_samples = list()

    with torch.no_grad():
        with model.ema_scope():
            uc = model.get_learned_conditioning(
                {model.cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.device)}
                )
            
            for class_label in classes:
                t0 = time.time()
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

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                            min=0.0, max=1.0)
                all_samples.append(x_samples_ddim)

                t1 = time.time()
                print('throughput : {}'.format(x_samples_ddim.shape[0] / (t1 - t0)))
                
    # display as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=n_samples_per_class)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    image_to_save = Image.fromarray(grid.astype(np.uint8))
    image_to_save.save("imagenet_brecq_w{}a{}_step{}_eta{}_scale{}_corrected.jpg".format(n_bits_w,n_bits_a,ddim_steps, ddim_eta, scale))

