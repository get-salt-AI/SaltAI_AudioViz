import torch

import comfy.conds
import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils

LEGACY = False
try:
    import comfy.sampler_helpers
except Exception:
    LEGACY = True

def prepare_mask(noise_mask, shape, device):
    """ensures noise mask is of proper dimensions"""
    print(noise_mask.shape)
    noise_mask = torch.nn.functional.interpolate(noise_mask.reshape((-1, 1, noise_mask.shape[-2], noise_mask.shape[-1])), size=(shape[2], shape[3]), mode="bilinear")
    noise_mask = torch.cat([noise_mask] * shape[1], dim=1)
    noise_mask = comfy.utils.repeat_to_batch_size(noise_mask, shape[0])
    noise_mask = noise_mask.to(device)
    return noise_mask

def get_models_from_cond(cond, model_type):
    models = []
    for c in cond:
        if model_type in c:
            models += [c[model_type]]
    return models

def convert_cond(cond):
    out = []
    for c in cond:
        temp = c[1].copy()
        model_conds = temp.get("model_conds", {})
        if c[0] is not None:
            model_conds["c_crossattn"] = comfy.conds.CONDCrossAttn(c[0]) #TODO: remove
            temp["cross_attn"] = c[0]
        temp["model_conds"] = model_conds
        out.append(temp)
    return out

def get_additional_models(positive, negative, dtype):
    """loads additional models in positive and negative conditioning"""
    control_nets = set(get_models_from_cond(positive, "control") + get_models_from_cond(negative, "control"))

    inference_memory = 0
    control_models = []
    for m in control_nets:
        control_models += m.get_models()
        inference_memory += m.inference_memory_requirements(dtype)

    gligen = get_models_from_cond(positive, "gligen") + get_models_from_cond(negative, "gligen")
    gligen = [x[1] for x in gligen]
    models = control_models + gligen
    return models, inference_memory

def cleanup_additional_models(models):
    """cleanup additional models that were loaded"""
    for m in models:
        if hasattr(m, 'cleanup'):
            m.cleanup()


def unsample(model, seed, cfg, sampler_name, steps, end_at_step, scheduler, normalize, positive, negative, latent_image):
    input_model = model
    device = comfy.model_management.get_torch_device()
    end_at_step = steps - min(end_at_step, steps - 1)

    latent = latent_image
    latent_image = latent["samples"].to(device)
    latent_mask = latent.get("noise_mask", None)
    
    noise_shape = latent_image.size()
    noise = torch.zeros(noise_shape, dtype=latent_image.dtype, layout=latent_image.layout, device=device)
    noise_mask = prepare_mask(latent.get("noise_mask"), latent_image.shape, device) if "noise_mask" in latent else None

    if LEGACY:
        positive = convert_cond(positive)
        negative = convert_cond(negative)

    models, inference_memory = get_additional_models(positive, negative, model.model_dtype())
    comfy.model_management.load_models_gpu([model] + models, model.memory_required(noise.shape) + inference_memory)

    model_options = model.model_options
    if LEGACY:
        model = model.model
        
    try:
        sampler = comfy.samplers.KSampler(model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=1.0, model_options=model_options)
    except Exception as e:
        raise e
    
    sigmas = sampler.sigmas.flip(0) + 0.0001

    pbar = comfy.utils.ProgressBar(steps)
    def callback(step, x0, x, total_steps): pbar.update_absolute(step + 1, total_steps)

    samples = sampler.sample(noise, positive, negative, cfg=cfg, latent_image=latent_image, force_full_denoise=False, denoise_mask=noise_mask, sigmas=sigmas, start_step=0, last_step=end_at_step, callback=callback, seed=seed)
    
    if normalize == "enable":
        samples = (samples - samples.mean()) / samples.std()
    
    cleanup_additional_models(models)
    
    out = latent.copy()
    out["samples"] = samples.cpu()
    if isinstance(latent_mask, torch.Tensor):
        out["noise_mask"] = latent_mask
        
    return (out,)