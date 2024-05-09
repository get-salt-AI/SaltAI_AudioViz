import torch
import time
import tqdm
import numpy as np
import math
import pilgram

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter

from ..modules.noise import VoronoiNoise, PerlinPowerFractalNoise, DisplacementLayer
from ..modules.utils import pil2tensor, pil2mask, tensor2pil, mask2pil

from comfy.utils import ProgressBar

from ..modules.transform import (
    generate_frame, 
    movement_modes, 
    edge_modes,
    ImageBatchTransition
)
from ..modules.easing import easing_functions
from ..modules.utils import pil2mask, pil2tensor, tensor2pil, mask2pil

from nodes import MAX_RESOLUTION

# TODO - Fix frame-rate issue
class OPACTransformImages:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
                "depth_maps": ("IMAGE", ),
                                
                "displacement_amplitude": ("FLOAT", {"min": 0, "max": 1024, "default": 60, "step": 0.1}),
                "displacement_angle_increment": ("INT", {"min": 1, "max": 24, "default": 3, "step": 1}),
                "displacement_start_frame": ("INT", {"min": 0, "max": 4096, "default": 0, "step": 1}),
                "displacement_end_frame": ("INT", {"min": 2, "max": 4096, "default": 60, "step": 1}),
                "displacement_preset": (list(movement_modes),),
                "displacement_easing": (list(easing_functions.keys()),),
                "displacement_tremor_scale": ("FLOAT", {"min": 0, "max": 100.0, "default": 0.02, "step": 0.01}),

                "edge_mode": (list(edge_modes.keys()),),

                "zoom_factor": ("FLOAT", {"min": 1, "max": 16, "default": 1, "step": 1}),
                "zoom_increment": ("FLOAT", {"min": 0, "max": 16, "default": 0, "step": 0.01}),
                "zoom_easing": (["ease-in", "ease-out", "ease-in-out", "bounce-in", "bounce-out", "bounce-in-out"],),
                "zoom_coordinate_x": ("INT", {"min": -1, "max": 8196, "default": -1, "step": 1}),
                "zoom_coordinate_y": ("INT", {"min": -1, "max": 8196, "default": -1, "step": 1}),
                "zoom_start_frame": ("INT", {"min": 0, "max": 4096, "default": 0, "step": 1}),
                "zoom_end_frame": ("INT", {"min": 2, "max": 4096, "default": 60, "step": 1}),
                "zoom_tremor_scale": ("FLOAT", {"min": 0, "max": 100.0, "default": 0.02, "step": 0.01}),

                "tremor_octaves": ("INT", {"min": 1, "max": 6, "default": 1}),
                "tremor_persistence": ("FLOAT", {"min": 0.01, "max": 1.0, "default": 0.5}),
                "tremor_lacunarity": ("FLOAT", {"min": 0.1, "max": 4.0, "default": 2, "step": 0.01}),

                "create_masks": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")

    CATEGORY = "SALT/Animation/Transform"
    FUNCTION = "transform"

    def transform(
            self,
            images,
            depth_maps,
            displacement_amplitude,
            displacement_angle_increment,
            displacement_start_frame,
            displacement_end_frame,
            displacement_preset,
            displacement_easing,
            displacement_tremor_scale,
            edge_mode,
            zoom_factor,
            zoom_increment,
            zoom_easing,
            zoom_coordinate_x,
            zoom_coordinate_y,
            zoom_start_frame,
            zoom_end_frame,
            zoom_tremor_scale,
            tremor_octaves,
            tremor_persistence,
            tremor_lacunarity,
            create_masks,
    ):
        
        frames = []
        masks = []

        num_frames = images.shape[0]

        start_time = time.time()
        comfy_pbar = ProgressBar(num_frames)

        for frame_number in tqdm(range(num_frames), desc='Generating frames'):
            
            image = images[frame_number]
            depth_map = depth_maps[frame_number] if frame_number <= depth_maps.shape[0] else depth_maps[-1]
            frame, mask = generate_frame(
                num_frames=num_frames,
                frame_number=frame_number,
                amplitude=displacement_amplitude,
                angle_increment=displacement_angle_increment,
                movement_mode=displacement_preset,
                movement_tremor_scale=displacement_tremor_scale,
                easing_function=displacement_easing,
                texture_image=tensor2pil(image.unsqueeze(0)),
                displacement_image=tensor2pil(depth_map.unsqueeze(0)),
                edge_mode=edge_mode,
                zoom_factor=zoom_factor,
                zoom_increment=zoom_increment,
                zoom_easing=zoom_easing,
                zoom_coordinates=(zoom_coordinate_x, zoom_coordinate_y),
                create_mask=create_masks,
                repetition=False,
                zoom_start_frame=zoom_start_frame,
                zoom_end_frame=zoom_end_frame,
                zoom_tremor_scale=zoom_tremor_scale,
                displacement_start_frame=displacement_start_frame,
                displacement_end_frame=displacement_end_frame,
                tremor_octaves=tremor_octaves,
                tremor_lacunarity=tremor_lacunarity,
                tremor_persistence=tremor_persistence
            )

            if frame:
                frames.append(pil2tensor(frame.convert("RGB")))
                if create_masks and mask is not None:
                    masks.append(pil2mask(mask.convert("L")))

            comfy_pbar.update(1)
        
        elapsed_time = time.time() - start_time
        
        print("Frames generated.")
        print(f"Transform Animation Completed in {elapsed_time:.2f} seconds")

        return (torch.cat(frames, dim=0), torch.cat(masks, dim=0))

class SaltScheduledImageAdjust:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "masks": ("MASK",),
                "brightness_schedule": ("LIST", ),
                "contrast_schedule": ("LIST", ),
                "saturation_schedule": ("LIST", ),
                "sharpness_schedule": ("LIST", ),
                "gaussian_blur_schedule": ("LIST", ),
                "edge_enhance_schedule": ("LIST", ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "images_adjust"

    CATEGORY = "SALT/Scheduling/Image"

    def float_value(self, adj_list, idx):
        if isinstance(adj_list, list) and adj_list:
            return adj_list[idx] if idx < len(adj_list) else adj_list[-1]
        else:
            return 1.0

    def images_adjust(
            self, 
            images, 
            masks=[], 
            brightness_schedule=[1.0], 
            contrast_schedule=[1.0], 
            saturation_schedule=[1.0], 
            sharpness_schedule=[1.0], 
            gaussian_blur_schedule=[0.0], 
            edge_enhance_schedule=[0.0]
        ):
        
        adjusted_images = []
        for idx, img in enumerate(images):
            original_pil_image = tensor2pil(img.unsqueeze(0))
            pil_image = original_pil_image.copy() 
            if isinstance(masks, torch.Tensor):
                pil_mask = mask2pil(masks[idx].unsqueeze(0)) if idx < len(masks) else mask2pil(masks[-1].unsqueeze(0))
                pil_mask = pil_mask.resize(original_pil_image.size).convert('L')
            else:
                pil_mask = None

            brightness = self.float_value(brightness_schedule, idx)
            contrast = self.float_value(contrast_schedule, idx)
            saturation = self.float_value(saturation_schedule, idx)
            sharpness = self.float_value(sharpness_schedule, idx)
            gaussian_blur = self.float_value(gaussian_blur_schedule, idx)
            edge_enhance = self.float_value(edge_enhance_schedule, idx)

            if brightness != 1.0:
                pil_image = ImageEnhance.Brightness(pil_image).enhance(brightness)
            
            if contrast != 1.0:
                pil_image = ImageEnhance.Contrast(pil_image).enhance(contrast)

            if saturation != 1.0:
                pil_image = ImageEnhance.Color(pil_image).enhance(saturation)

            if sharpness != 1.0:
                pil_image = ImageEnhance.Sharpness(pil_image).enhance(sharpness)

            if gaussian_blur > 0.0:
                pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=gaussian_blur))

            if edge_enhance > 0.0:
                edge_enhanced_img = pil_image.filter(ImageFilter.EDGE_ENHANCE_MORE)
                blend_mask = Image.new("L", pil_image.size, color=int(round(edge_enhance * 255)))
                pil_image = Image.composite(edge_enhanced_img, pil_image, blend_mask)

            if pil_mask:
                pil_image = Image.composite(pil_image, original_pil_image, pil_mask)

            adjusted_images.append(pil2tensor(pil_image))

        return (torch.cat(adjusted_images, dim=0), )


class SaltScheduledShapeTransformation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_frames": ("INT", {"min": 1}),
                "image_width": ("INT", {"default": 512, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION}),
                "image_height": ("INT", {"default": 512, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION}),
                "initial_width": ("INT", {"default": 256, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION}),
                "initial_height": ("INT", {"default": 256, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION}),
                "initial_x_coord": ("INT", {"default": 256, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION}),
                "initial_y_coord": ("INT", {"default": 256, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION}),
                "initial_rotation": ("FLOAT", {"min": 0, "max": 360, "step": 0.01}),
                "shape_mode": (["circle", "diamond", "triangle", "square", "hexagon", "octagon"], ),
            },
            "optional": {
                "shape": ("MASK", ),
                "width_schedule": ("LIST", ),
                "height_schedule": ("LIST", ),
                "x_schedule": ("LIST", ),
                "y_schedule": ("LIST", ),
                "rotation_schedule": ("LIST", ),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("images", )
    FUNCTION = "transform_shape"
    CATEGORY = "SALT/Scheduling/Image"

    def apply_multiply_operation(self, initial_value, schedule, frame_idx):
        factor = schedule[min(frame_idx, len(schedule) - 1)]
        return initial_value * factor

    def draw_shape(self, draw, shape_mode, center, width, height):
        if shape_mode == "circle":
            draw.ellipse([(center[0] - width / 2, center[1] - height / 2), (center[0] + width / 2, center[1] + height / 2)], fill="white")
        elif shape_mode == "square":
            draw.rectangle([(center[0] - width / 2, center[1] - height / 2), (center[0] + width / 2, center[1] + height / 2)], fill="white")
        elif shape_mode == "diamond":
            half_width = width / 2
            half_height = height / 2
            draw.polygon([center[0], center[1] - half_height, center[0] + half_width, center[1], center[0], center[1] + half_height, center[0] - half_width, center[1]], fill="white")
        elif shape_mode == "triangle":
            draw.polygon([(center[0], center[1] - height / 2), (center[0] + width / 2, center[1] + height / 2), (center[0] - width / 2, center[1] + height / 2)], fill="white")
        elif shape_mode == "hexagon":
            angle = 2 * np.pi / 6
            points = [(center[0] + math.cos(i * angle) * width / 2, center[1] + math.sin(i * angle) * height / 2) for i in range(6)]
            draw.polygon(points, fill="white")
        elif shape_mode == "octagon":
            angle = 2 * np.pi / 8
            points = [(center[0] + math.cos(i * angle) * width / 2, center[1] + math.sin(i * angle) * height / 2) for i in range(8)]
            draw.polygon(points, fill="white")

    def transform_shape(self, max_frames, image_width, image_height, initial_width, initial_height, initial_x_coord, initial_y_coord, initial_rotation, shape_mode, shape=None, width_schedule=[1.0], height_schedule=[1.0], x_schedule=[1.0], y_schedule=[1.0], rotation_schedule=[1.0]):
        frames = []
        for frame_idx in range(max_frames):
            width = self.apply_multiply_operation(initial_width, width_schedule, frame_idx)
            height = self.apply_multiply_operation(initial_height, height_schedule, frame_idx)
            x_coord = self.apply_multiply_operation(initial_x_coord, x_schedule, frame_idx)
            y_coord = self.apply_multiply_operation(initial_y_coord, y_schedule, frame_idx)
            rotation_fraction = rotation_schedule[min(frame_idx, len(rotation_schedule) - 1)]
            rotation_degree = rotation_fraction * 360
            
            img = Image.new('RGB', (image_width, image_height), 'black')
            if isinstance(shape, torch.Tensor):
                shape_image = mask2pil(shape)
                shape_image = shape_image.resize((max(int(width), 1), max(int(height), 1)), resample=Image.LANCZOS)
                rotated_shape_image = shape_image.rotate(rotation_degree, expand=True, fillcolor=(0), resample=Image.BILINEAR)
                paste_x = int(x_coord - rotated_shape_image.width / 2)
                paste_y = int(y_coord - rotated_shape_image.height / 2)
                img.paste(rotated_shape_image, (paste_x, paste_y), rotated_shape_image)
            else:
                shape_img = Image.new('RGBA', (max(int(width), 1), max(int(height), 1)), (0, 0, 0, 0))
                shape_draw = ImageDraw.Draw(shape_img)
                self.draw_shape(shape_draw, shape_mode, (shape_img.width / 2, shape_img.height / 2), width, height)
                rotated_shape_img = shape_img.rotate(rotation_degree, expand=True, fillcolor=(0), resample=Image.BILINEAR)
                paste_x = int(x_coord - rotated_shape_img.width / 2)
                paste_y = int(y_coord - rotated_shape_img.height / 2)
                img.paste(rotated_shape_img, (paste_x, paste_y), rotated_shape_img)

            frames.append(img)
            
        if frames:
            tensor = [pil2tensor(img) for img in frames]
            tensor = torch.cat(tensor, dim=0)
        else:
            raise ValueError("No frames were generated!")

        return (tensor, )


class SaltScheduledVoronoiNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_size": ("INT", {"min": 1, "max": 4096}),
                "width": ("INT", {"default": 64, "min": 64, "max": MAX_RESOLUTION}),
                "height": ("INT", {"default": 64, "min": 64, "max": MAX_RESOLUTION}),
            }, 
            "optional": {
                "distance_metric": (["euclidean", "manhattan", "chebyshev", "minkowski"],),
                "x_schedule": ("LIST",),
                "y_schedule": ("LIST",),
                "scale_schedule": ("LIST",),
                "detail_schedule": ("LIST",),
                "randomness_schedule": ("LIST",),
                "seed_schedule": ("LIST", ),
                "device": (["cuda", "cpu"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "batch_size")

    FUNCTION = "generate"
    CATEGORY = "SALT/Scheduling/Image"

    def generate(self, batch_size, width, height, distance_metric="euclidean", x_schedule=[0], y_schedule=[0], z_schedule=[0], scale_schedule=[1.0], detail_schedule=[100], randomness_schedule=[1], seed_schedule=[0], device="cuda"):
        voronoi = VoronoiNoise(width=width, height=height, scale=scale_schedule, detail=detail_schedule, seed=seed_schedule, 
                            X=x_schedule, Y=y_schedule, 
                            randomness=randomness_schedule, distance_metric=distance_metric, batch_size=batch_size, device=device)
        voronoi = voronoi.to(device)
        tensors = voronoi()
        return(tensors, batch_size)


class SaltScheduledPerlinPowerFractalNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_size": ("INT", {"min": 1, "max": 4096}),
                "width": ("INT", {"default": 256, "min": 64, "max": MAX_RESOLUTION}),
                "height": ("INT", {"default": 256, "min": 64, "max": MAX_RESOLUTION}),
            },
            "optional": {
                "scale_schedule": ("LIST",),
                "octaves_schedule": ("LIST",),
                "persistence_schedule": ("LIST",),
                "lacunarity_schedule": ("LIST",),
                "exponent_schedule": ("LIST",),
                "seed_schedule": ("LIST",),
                "clamp_min_schedule": ("LIST",),
                "clamp_max_schedule": ("LIST",),
                "device": (["cuda", "cpu"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "batch_size")

    FUNCTION = "generate"
    CATEGORY = "SALT/Scheduling/Image"

    def generate(self, batch_size, width, height, scale_schedule=[1.0], octaves_schedule=[4], persistence_schedule=[0.5], lacunarity_schedule=[2.0], exponent_schedule=[1.0], seed_schedule=[0], clamp_min_schedule=[-0.5], clamp_max_schedule=[1.5], device="cuda"):
        octaves_schedule = [int(octave) for octave in octaves_schedule]
        ppfn = PerlinPowerFractalNoise(
            width, height, 
            scale=scale_schedule, 
            octaves=octaves_schedule, 
            persistence=persistence_schedule, 
            lacunarity=lacunarity_schedule, 
            exponent=exponent_schedule, 
            seeds=seed_schedule, 
            clamp_min=clamp_min_schedule, 
            clamp_max=clamp_max_schedule, 
            batch_size=batch_size, 
            device=device
        )
        noise_tensor = ppfn.forward()
        return (noise_tensor, batch_size)


class SaltScheduledImageDisplacement:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "displacement_images": ("IMAGE",),
            },
            "optional": {
                "amplitude_schedule": ("LIST",),
                "angle_schedule": ("LIST",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    FUNCTION = "apply_displacement"
    CATEGORY = "SALT/Scheduling/Image"

    def apply_displacement(self, images, displacement_images, amplitude_schedule=None, angle_schedule=None):
        batch_size, height, width, _ = images.shape
        device = images.device

        # Initialize the optimized displacement layer
        displacement_layer = DisplacementLayer(device)

        displaced_images = []
        for i in range(batch_size):
            # Default amplitude and angle to 0 if their schedules are not provided or are too short
            amplitude_value = amplitude_schedule[i] if amplitude_schedule and i < len(amplitude_schedule) else 0
            angle_value = angle_schedule[i] if angle_schedule and i < len(angle_schedule) else 0

            amplitude = torch.tensor([amplitude_value], dtype=torch.float, device=device)
            angle = torch.tensor([angle_value], dtype=torch.float, device=device)
            
            image = images[i:i+1].to(device)
            displacement_image = displacement_images[i:i+1].to(device)
            
            # Use the displacement layer
            displaced_image = displacement_layer(image, displacement_image, amplitude, angle)
            displaced_images.append(displaced_image)

        # Combine the batch of displaced images
        displaced_images = torch.cat(displaced_images, dim=0)

        return (displaced_images,)
    

class SaltScheduledBinaryComparison:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "comparison_schedule": ("LIST",),
            },
            "optional": {
                "epsilon_schedule": ("LIST", {}),
                "use_epsilon": ("BOOLEAN", {"default": True})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    FUNCTION = "binary_threshold"
    CATEGORY = "SALT/Scheduling/Image"

    def binary_threshold(self, images, comparison_schedule, epsilon_schedule=[0.1], use_epsilon=True):
        batch_size = images.shape[0]

        if len(comparison_schedule) < batch_size:
            last_val = comparison_schedule[-1]
            comparison_schedule.extend([last_val] * (batch_size - len(comparison_schedule)))
        comparison_schedule = comparison_schedule[:batch_size]

        thresholds_tensor = torch.tensor(comparison_schedule, dtype=images.dtype).view(batch_size, 1, 1, 1)
        
        if use_epsilon:
            if epsilon_schedule is None:
                epsilon_schedule = [0.1] * batch_size
            if len(epsilon_schedule) < batch_size:
                last_eps = epsilon_schedule[-1]
                epsilon_schedule.extend([last_eps] * (batch_size - len(epsilon_schedule)))
            epsilon_schedule = epsilon_schedule[:batch_size]
            epsilon_tensor = torch.tensor(epsilon_schedule, dtype=images.dtype).view(batch_size, 1, 1, 1)
            
            condition_met = ((images == thresholds_tensor) |
                             (torch.abs(images - thresholds_tensor) <= epsilon_tensor))
        else:
            condition_met = images >= thresholds_tensor

        thresholded_images = torch.where(condition_met, torch.tensor(1.0, dtype=images.dtype), torch.tensor(0.0, dtype=images.dtype))

        return (thresholded_images, )

class SaltImageComposite:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images_a": ("IMAGE",),
                "images_b": ("IMAGE",),
                "mode": ([
                    "add",
                    "color",
                    "color_burn",
                    "color_dodge",
                    "darken",
                    "difference",
                    "exclusion",
                    "hard_light",
                    "hue",
                    "lighten",
                    "multiply",
                    "overlay",
                    "screen",
                    "soft_light"
                ],),
            },
            "optional": {
                "masks": ("MASK",),
                "blend_schedule": ("LIST", ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "blend"
    CATEGORY = f"SALT/Scheduling/Image"

    def blend(self, images_a, images_b, mode, blend_schedule=[1.0], masks=None):
        blended_images = []
        min_length = min(len(images_a), len(images_b))

        if len(blend_schedule) < min_length:
            blend_schedule += [blend_schedule[-1]] * (min_length - len(blend_schedule))

        for i in range(min_length):
            img_a = tensor2pil(images_a[i].unsqueeze(0))
            img_b = tensor2pil(images_b[i].unsqueeze(0))
            img_b_resized = img_b.resize(img_a.size, Image.LANCZOS).convert(img_a.mode)

            if mode == "add":
                base_image = ImageChops.add(img_a, img_b_resized, scale=2.0, offset=int(255 * (1 - blend_schedule[i])))
            else:
                base_image = getattr(pilgram.css.blending, mode)(img_a, img_b_resized)

            blend_mask = Image.new("L", img_a.size, int(255 * blend_schedule[i]))
            out_image = Image.composite(base_image, img_a, blend_mask)

            if isinstance(masks, torch.Tensor):
                mask = mask2pil(masks[i if len(masks) > i else -1].unsqueeze(0)).resize(img_a.size, Image.LANCZOS).convert("L")
                final_image = Image.composite(out_image, img_a, mask)
            else:
                final_image = out_image

            blended_images.append(pil2tensor(final_image))

        blended_images_batch = torch.cat(blended_images, dim=0)
        return (blended_images_batch, )
    

class SaltFilmicTransitions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images_a": ("IMAGE", ),
                "images_b": ("IMAGE", ),
                "mode": (["sipe_left", "swipe_right", "swipe_up", "swipe_down", "diagonal_tl_br", "diagonal_tr_bl", "diagonal_bl_tr", "diagonal_br_tl", "circle_expand", "circle_contract"],),
                "transition_frames": ("INT", {"min": 2, "max": 1024, "default": 10}),
            },
            "optional": {
                "mask_blur_schedule": ("LIST", ),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    FUNCTION = "generate_transition"
    CATEGORY = "SALT/Scheduling/Image"

    def generate_transition(self, images_a, images_b, mode, transition_frames, mask_blur_schedule=[0]):
        mask_blur_schedule = [float(val) for val in mask_blur_schedule]
        img_list_a = [tensor2pil(img) for img in images_a]
        img_list_b = [tensor2pil(img) for img in images_b]
        transition = ImageBatchTransition(img_list_a, img_list_b, frames_per_transition=int(transition_frames), blur_radius=mask_blur_schedule, mode=mode)
        result_images = transition.create_transition()
        result_images = [pil2tensor(img) for img in result_images]
        return (torch.cat(result_images, dim=0), )


NODE_DISPLAY_NAME_MAPPINGS = {
    "SaltScheduledImageAdjust": "Batch Image Scheduled Adjustments",
    "SaltScheduledShapeTransformation": "Scheduled Shape Transform",
    "SaltScheduledVoronoiNoise": "Scheduled Voronoi Noise Generator",
    "SaltScheduledPerlinPowerFractalNoise": "Scheduled Perline Power Fractal Generator",
    "SaltScheduledImageDisplacement": "Scheduled Image Displacement",
    "SaltScheduledBinaryComparison": "Scheduled Binary Comparison",
    "SaltImageComposite": "Scheduled Image Composite",
    "SaltFilmicTransitions": "Image Batch Filmic Transitions"
}

NODE_CLASS_MAPPINGS = {
    key: globals()[key] for key in NODE_DISPLAY_NAME_MAPPINGS.keys()
}