import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .utils import tensor2pil

class VoronoiNoise(nn.Module):
    def __init__(self, width, height, scale, detail, seed, randomness, X=[0], Y=[0], distance_metric='euclidean', batch_size=1, device='cpu'):
        super(VoronoiNoise, self).__init__()
        # Flip the assignments here to correctly reflect their intended uses
        self.width, self.height = width, height
        self.scale = torch.tensor(self._adjust_list_length(scale, batch_size), dtype=torch.float, device=device)
        self.detail = self._adjust_list_length(detail, batch_size)
        self.seed = self._adjust_list_length(seed, batch_size)
        self.randomness = torch.tensor(self._adjust_list_length(randomness, batch_size), dtype=torch.float, device=device)
        self.X = torch.tensor(self._adjust_list_length(X, batch_size), dtype=torch.float, device=device)
        self.Y = torch.tensor(self._adjust_list_length(Y, batch_size), dtype=torch.float, device=device)
        self.distance_metric = distance_metric
        self.batch_size = batch_size
        self.device = device

    @staticmethod
    def _adjust_list_length(lst, length):
        return lst + [lst[-1]] * (length - len(lst)) if len(lst) < length else lst

    def forward(self):
        noise_batch = []
        for b in range(self.batch_size):
            torch.manual_seed(self.seed[b])

            center_x = self.width // 2
            center_y = self.height // 2

            sqrt_detail = int(np.sqrt(self.detail[b]))
            spacing = max(self.width, self.height) / sqrt_detail
            offsets_x = torch.arange(-sqrt_detail // 2, sqrt_detail // 2 + 1, device=self.device) * spacing
            offsets_y = torch.arange(-sqrt_detail // 2, sqrt_detail // 2 + 1, device=self.device) * spacing

            grid_x, grid_y = torch.meshgrid(offsets_x, offsets_y, indexing='xy')
            points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)

            random_offsets = (torch.rand_like(points) * 2 - 1) * self.randomness[b] * spacing / 2
            points += random_offsets
            points[len(points) // 2] = torch.tensor([0, 0], device=self.device)

            points *= self.scale[b]

            points += torch.tensor([self.X[b], self.Y[b]], device=self.device)
            points += torch.tensor([center_x, center_y], device=self.device)

            x_coords = torch.arange(self.width, device=self.device)
            y_coords = torch.arange(self.height, device=self.device)
            grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
            grid = torch.stack([grid_x, grid_y], dim=-1).float()

            if self.distance_metric == 'euclidean':
                distances = torch.sqrt(((grid.unsqueeze(2) - points) ** 2).sum(dim=-1))
            elif self.distance_metric == 'manhattan':
                distances = torch.abs(grid.unsqueeze(2) - points).sum(dim=-1)
            elif self.distance_metric == 'chebyshev':
                distances = torch.abs(grid.unsqueeze(2) - points).max(dim=-1).values
            elif self.distance_metric == 'minkowski':
                p = 3
                distances = (torch.abs(grid.unsqueeze(2) - points) ** p).sum(dim=-1) ** (1/p)
            else:
                raise ValueError("Unsupported distance metric")

            min_distances, _ = distances.min(dim=-1)

            single_noise = min_distances
            single_noise_flat = single_noise.view(-1)
            local_min = single_noise_flat.min()
            local_max = single_noise_flat.max()
            normalized_noise = (single_noise - local_min) / (local_max - local_min)

            final_output = normalized_noise.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3).cpu()

            noise_batch.append(final_output)

        return torch.cat(noise_batch, dim=0)
    
    
class PerlinPowerFractalNoise(nn.Module):
    def __init__(self, width, height, scale, octaves, persistence, lacunarity, exponent, seeds, clamp_min, clamp_max, batch_size=1, device='cpu'):
        super(PerlinPowerFractalNoise, self).__init__()
        self.width = width
        self.height = height
        self.device = device
        self.scale = self._adjust_list_length(scale, batch_size)
        self.octaves = self._adjust_list_length(octaves, batch_size)
        self.persistence = self._adjust_list_length(persistence, batch_size)
        self.lacunarity = self._adjust_list_length(lacunarity, batch_size)
        self.exponent = self._adjust_list_length(exponent, batch_size)
        self.seeds = self._adjust_list_length(seeds, batch_size)
        self.clamp_min = self._adjust_list_length(clamp_min, batch_size)
        self.clamp_max = self._adjust_list_length(clamp_max, batch_size)
        self.batch_size = batch_size

    @staticmethod
    def _adjust_list_length(lst, length):
        return lst + [lst[-1]] * (length - len(lst)) if len(lst) < length else lst

    @staticmethod
    def fade(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    @staticmethod
    def lerp(t, a, b):
        return a + t * (b - a)

    @staticmethod
    def grad(hash, x, y, z):
        h = hash & 15
        u = torch.where(h < 8, x, y)
        v = torch.where(h < 4, y, torch.where((h == 12) | (h == 14), x, z))
        return torch.where(h & 1 == 0, u, -u) + torch.where(h & 2 == 0, v, -v)

    def noise(self, x, y, z, p):
        X = (x.floor() % 255).to(torch.int32)
        Y = (y.floor() % 255).to(torch.int32)
        Z = (z.floor() % 255).to(torch.int32)

        x -= x.floor()
        y -= y.floor()
        z -= z.floor()

        u = self.fade(x)
        v = self.fade(y)
        w = self.fade(z)

        A = p[X] + Y
        AA = p[A] + Z
        AB = p[A + 1] + Z
        B = p[X + 1] + Y
        BA = p[B] + Z
        BB = p[B + 1] + Z

        return self.lerp(w, self.lerp(v, self.lerp(u, self.grad(p[AA], x, y, z), self.grad(p[BA], x - 1, y, z)),
                                    self.lerp(u, self.grad(p[AB], x, y - 1, z), self.grad(p[BB], x - 1, y - 1, z))),
                         self.lerp(v, self.lerp(u, self.grad(p[AA + 1], x, y, z - 1), self.grad(p[BA + 1], x - 1, y, z - 1)),
                                    self.lerp(u, self.grad(p[AB + 1], x, y - 1, z - 1), self.grad(p[BB + 1], x - 1, y - 1, z - 1))))

    def perlin_noise(self, x, y, z, batch_index, p):
        total = torch.zeros_like(x)
        max_value = 0.0
        amplitude = 1.0

        for octave in range(self.octaves[batch_index]):
            frequency = self.lacunarity[batch_index] ** octave
            n = self.noise(x * frequency / self.scale[batch_index], y * frequency / self.scale[batch_index], z * frequency / self.scale[batch_index], p)
            total += n * amplitude
            max_value += amplitude
            amplitude *= self.persistence[batch_index]

        total = (total / max_value) ** self.exponent[batch_index]
        return total

    def forward(self):
        images = torch.zeros((self.batch_size, self.height, self.width, 3), device=self.device)
        for i in range(self.batch_size):
            torch.manual_seed(self.seeds[i])
            p = torch.randperm(256, device=self.device).repeat(2)  # Ensure permutation is appropriate
            #p = torch.cat((p, p))

            x_coords = torch.arange(self.width, device=self.device).unsqueeze(0).repeat(self.height, 1) / self.scale[i]
            y_coords = torch.arange(self.height, device=self.device).unsqueeze(1).repeat(1, self.width) / self.scale[i]
            z_coords = torch.full((self.height, self.width), i, device=self.device) / self.scale[i]  # not implemented

            noise_result = self.perlin_noise(x_coords, y_coords, z_coords, i, p)
            noise_result_clamped = noise_result.clamp(min=self.clamp_min[i], max=self.clamp_max[i])
            normalized_noise = (noise_result_clamped - noise_result_clamped.min()) / (noise_result_clamped.max() - noise_result_clamped.min())
            images[i, :, :, 0] = normalized_noise
            images[i, :, :, 1] = normalized_noise
            images[i, :, :, 2] = normalized_noise

        return images.cpu()
    
    def save_images(self, noise_tensor, path='./'):
        if not os.path.exists(path):
            os.makedirs(path)
        for i, img_tensor in enumerate(noise_tensor):
            img = tensor2pil(img_tensor.squeeze(0))
            filename = os.path.join(path, f"perlin_power_fractal_{i+1:04d}.png")
            img.save(filename)


class DisplacementLayer(nn.Module):
    def __init__(self, device):
        super(DisplacementLayer, self).__init__()
        self.device = device

    def _create_meshgrid(self, height, width, device):
        meshgrid_y, meshgrid_x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=device), 
            torch.linspace(-1, 1, width, device=device), indexing='ij'
        )
        return torch.stack((meshgrid_x, meshgrid_y), dim=2)[None, :, :, :]

    def forward(self, image, displacement_map, amplitude, angle):
        N, H, W, C = image.shape
        
        meshgrid = self._create_meshgrid(H, W, self.device)

        displacement_map_resized = F.interpolate(
            displacement_map.permute(0, 3, 1, 2), 
            size=(H, W), mode='bilinear', align_corners=False
        )

        if displacement_map_resized.shape[1] == 3:
            displacement_map_resized = torch.mean(displacement_map_resized, dim=1, keepdim=True)

        displacement_map_modulated = displacement_map_resized * amplitude.view(N, 1, 1, 1)
        dx = torch.cos(angle.view(N, 1, 1, 1)) * displacement_map_modulated
        dy = torch.sin(angle.view(N, 1, 1, 1)) * displacement_map_modulated

        dx_norm = dx / (W - 1) * 2
        dy_norm = dy / (H - 1) * 2
        dx_norm = dx_norm.expand(-1, -1, H, W)
        dy_norm = dy_norm.expand(-1, -1, H, W)

        displacements = torch.stack((dx_norm, dy_norm), dim=4).squeeze(1)
        displaced_grid = meshgrid + displacements
        displaced_image = F.grid_sample(
            image.permute(0, 3, 1, 2).float(), 
            displaced_grid, mode='bilinear', padding_mode='reflection', align_corners=False
        ).permute(0, 2, 3, 1)

        return displaced_image.cpu()