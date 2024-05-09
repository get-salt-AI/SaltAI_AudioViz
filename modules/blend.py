import math
import torch
import torch.nn.functional as F

def slerp(strength, tensor_from, tensor_to, epsilon=1e-6):
    low_norm = F.normalize(tensor_from, p=2, dim=-1, eps=epsilon)
    high_norm = F.normalize(tensor_to, p=2, dim=-1, eps=epsilon)
    
    dot_product = torch.clamp((low_norm * high_norm).sum(dim=-1), -1.0, 1.0)
    omega = torch.acos(dot_product)
    so = torch.sin(omega)
    zero_so_mask = torch.isclose(so, torch.tensor([0.0], device=so.device), atol=epsilon)
    so = torch.where(zero_so_mask, torch.tensor([1.0], device=so.device), so)
    sin_omega_minus_strength = torch.sin((1.0 - strength) * omega) / so
    sin_strength_omega = torch.sin(strength * omega) / so
    
    res = sin_omega_minus_strength.unsqueeze(-1) * tensor_from + sin_strength_omega.unsqueeze(-1) * tensor_to
    res = torch.where(zero_so_mask.unsqueeze(-1), 
                      tensor_from if strength < 0.5 else tensor_to, 
                      res)
    return res

# from  https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475
def slerp_latents(val, low, high):
    dims = low.shape

    #flatten to batches
    low = low.reshape(dims[0], -1)
    high = high.reshape(dims[0], -1)

    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)

    # in case we divide by zero
    low_norm[low_norm != low_norm] = 0.0
    high_norm[high_norm != high_norm] = 0.0

    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res.reshape(dims)

def blend_latents(alpha, latent_1, latent_2):
    if not isinstance(alpha, torch.Tensor):
        alpha = torch.tensor([alpha], dtype=latent_1.dtype, device=latent_1.device)
    
    blended_latent = (1 - alpha) * latent_1 + alpha * latent_2
    
    return blended_latent

def cosine_interp_latents(val, low, high):
    if not isinstance(val, torch.Tensor):
        val = torch.tensor([val], dtype=low.dtype, device=low.device)        
    t = (1 - torch.cos(val * math.pi)) / 2
    return (1 - t) * low + t * high