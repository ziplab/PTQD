"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import sys, time
sys.path.append(".")
sys.path.append('./taming-transformers')
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6'
import time
import logging

import numpy as np
import torch.distributed as dist

import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler, DDIMSampler_quantCorrection_imagenet

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=50000)
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--out_dir', default='./generated')
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument("--resume", action='store_true')
    args = parser.parse_args()
    print(args)
    # init ddp
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    rank = torch.distributed.get_rank()
    ## for debug, not use ddp
    # rank=0
    # local_rank=0
    # Setup PyTorch:
    logging.basicConfig(level=logging.INFO if rank in [-1, 0] else logging.WARN)
    if args.resume:
        seed = int(time.time())
        torch.manual_seed(seed + rank)
    else:
        torch.manual_seed(0 + rank)

    torch.set_grad_enabled(False)
    device = torch.device("cuda", local_rank)

    ddim_steps = 20
    ddim_eta = 0.0
    scale = 3.0

    # Load model:
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

    # Disable output quantization because network output
    # does not get involved in further computation
    qnn.disable_network_output_quantization()

    ckpt = torch.load('quantw{}a{}_ldm_brecq.pth'.format(n_bits_w, n_bits_a), map_location='cpu')
    qnn.load_state_dict(ckpt)
    qnn.cuda()
    qnn.eval()
    setattr(model.model, 'diffusion_model', qnn)
    sampler = DDIMSampler_quantCorrection_imagenet(model, num_bit=4, correct=True)

    out_path = os.path.join(args.out_dir, f"brecq_w{n_bits_w}a{n_bits_a}_{args.num_samples}steps{ddim_steps}eta{ddim_eta}scale{scale}_0504.npz")

    logging.info("sampling...")
    generated_num = torch.tensor(0, device=device)
    if rank == 0:
        all_images = []
        all_labels = []
        if args.resume:
            if os.path.exists(out_path):
                ckpt = np.load(out_path)
                all_images = ckpt['arr_0']
                all_labels = ckpt['arr_1']
                assert all_images.shape[0] % args.batch_size == 0, f'Wrong resume checkpoint shape {all_images.shape}'
                all_images = np.split(all_images,
                                      all_images.shape[0] // args.batch_size,
                                      0)
                all_labels = np.split(all_labels,
                                      all_labels.shape[0] // args.batch_size,
                                      0)

                logging.info('successfully resume from the ckpt')
                logging.info(f'Current number of created samples: {len(all_images) * args.batch_size}')
        generated_num = torch.tensor(len(all_images) * args.batch_size, device=device)
    dist.barrier()
    dist.broadcast(generated_num, 0)
    n_samples_per_class = args.batch_size
    while generated_num.item() < args.num_samples:
        class_labels = torch.randint(low=0,
                                     high=args.num_classes,
                                     size=(args.batch_size,),
                                     device=device)
        uc = model.get_learned_conditioning(
            {model.cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.device)}
            )
        
        for class_label in class_labels:
            t0 = time.time()
            xc = torch.tensor(n_samples_per_class*[class_label]).to(model.device)
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
            
            x_samples_ddim = ((x_samples_ddim + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            x_samples_ddim = x_samples_ddim.permute(0, 2, 3, 1)
            samples = x_samples_ddim.contiguous()

            t1 = time.time()
            print('throughput : {}'.format(x_samples_ddim.shape[0] / (t1 - t0)))
            
            gathered_samples = [torch.zeros_like(samples) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, samples)  # gather not supported with NCCL

            gathered_labels = [
                torch.zeros_like(xc) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, xc)

            if rank == 0:
                all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
                all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
                logging.info(f"created {len(all_images) * n_samples_per_class} samples")
                generated_num = torch.tensor(len(all_images) * n_samples_per_class, device=device)
                if args.resume:
                    if generated_num % 1024 == 0:
                        arr = np.concatenate(all_images, axis=0)
                        arr = arr[: args.num_samples]

                        label_arr = np.concatenate(all_labels, axis=0)
                        label_arr = label_arr[: args.num_samples]
                        logging.info(f"intermediate results saved to {out_path}")
                        np.savez(out_path, arr, label_arr)
                        del arr
                        del label_arr
            torch.distributed.barrier()
            dist.broadcast(generated_num, 0)

    if rank == 0:
        arr = np.concatenate(all_images, axis=0)
        arr = arr[: args.num_samples]

        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]

        logging.info(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

    dist.barrier()
    logging.info("sampling complete")


if __name__ == "__main__":
    main()