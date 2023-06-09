# PTQD
This is the official implementation of PTQD: Accurate Post-Training Quantization for Diffusion Models. If you find any bugs, please do not hesitate to contact me.
## Getting Started
### Requirements
Establish a virtual environment and install dependencies as referred to [LDM](https://github.com/CompVis/latent-diffusion).
### Usage
Here we detail how to quantize models and collect quantization noise for correction and taking ImageNet as example. 

First, downloading the pretrained models provided by LDM:
    
    mkdir -p models/ldm/cin256-v2/
    wget -O models/ldm/cin256-v2/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt 
Now you should be able to sample images with the following script:
```bash
python3 quant_scripts/main_imagenet.py
```
Then collecting input data of the model, which is required for calibration. You can modify the amount of data by setting different classes and n_samples_per_class in the script.
```bash
python3 quant_scripts/collect_diffusion_input_imagenet.py
```
Then you can quantize and calibrate the model by running:
```bash
python3 quant_scripts/quantize_ldm_brecq.py ## for weights
python3 quant_scripts/quantize_ldm_brecqA.py ## for activations
```
With quantized model and full-precision model, you can collect quantization noise by:
```bash
python3 quant_scripts/collect_quant_error_brecq.py
```
Also, you can modify the amount of data you want to collect.

Then you can analyze the quantization noise to get statistics (correlation, noise mean and variance) for correction:
```bash
python3 quant_scripts/analyze_error.py
```

Now sampling with quantized model!
```bash
python3 quant_scripts/main_brecq_imagenet.py
```
You can set correct=False in DDIMSampler_quantCorrection_imagenet to disable correction.
## Acknowledgement

This repository is built upon [LDM](https://github.com/CompVis/latent-diffusion) and [BRECQ](https://github.com/yhhhli/BRECQ). We thank the authors for their open-sourced code.
