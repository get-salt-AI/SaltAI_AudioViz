import json
import os
import matplotlib.pyplot as plt
import random
import math
import numpy as np
import uuid
import hashlib

import numpy as np
import torch
from pprint import pprint

import folder_paths
import comfy.sample
import comfy.samplers
import comfy.model_management
import nodes

from .. import MENU_NAME, SUB_MENU_NAME, logger
from ..modules.transform import (
    PerlinNoise, 
    OrganicPerlinCameraScheduler, 
    zoom_presets,
    horizontal_pan_presets,
    vertical_pan_presets
)
from ..modules.easing import easing_functions, KeyframeScheduler, safe_eval
from ..modules.unsample import unsample
from ..modules.blend import blend_latents, cosine_interp_latents, slerp_latents


INPUT = folder_paths.get_input_directory()
TEMP = folder_paths.get_temp_directory()

CLIPTextEncode = nodes.CLIPTextEncode()
USE_BLK, BLK_ADV = (False, None)
if "BNK_CLIPTextEncodeAdvanced" in nodes.NODE_CLASS_MAPPINGS:
    BLK_ADV = nodes.NODE_CLASS_MAPPINGS['BNK_CLIPTextEncodeAdvanced']
    USE_BLK = True

if USE_BLK:
    logger.info(f"\n[Animation-Keyframing] Found `\33[1mComfyUI_ADV_CLIP_emb\33[0m`. Using \33[93mBLK Advanced CLIPTextEncode\33[0m for Conditioning Sequencing\n")
    blk_adv = BLK_ADV()

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False
        
WILDCARD = AnyType("*")

def log_curve(label, value):
    logger.info(f"\t\033[1m\033[93m{label}:\033[0m")
    logger.data(value, indent=4)

class SaltOPAC:
    """
        Generates semi-random keyframes for zoom, spin, and translation based on specified start and end ranges,
        with individual tremor scale controls for each parameter, allowing for organic variation using Perlin noise.
    """
    def __init__(self):
        self.noise_base = random.randint(0, 1000)
        self.perlin_noise = PerlinNoise()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_wiggle": ("BOOLEAN", {"default": True}),
                "frame_count": ("INT", {"default": 48, "min": 1, "max": 500}),
                "zoom_range": ("STRING", {"default": "0.95,1.05"}),
                "zoom_tremor_scale": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 1.0}),
                "angle_range": ("STRING", {"default": "-5,5"}),
                "angle_tremor_scale": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 1.0}),
                "trx_range": ("STRING", {"default": "-10,10"}),
                "trx_tremor_scale": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 1.0}),
                "try_range": ("STRING", {"default": "-10,10"}),
                "try_tremor_scale": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 1.0}),
                "trz_range": ("STRING", {"default": "-10,10"}),
                "trz_tremor_scale": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 1.0}),
                "rotx_range": ("STRING", {"default": "-5,5"}),
                "rotx_tremor_scale": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 1.0}),
                "roty_range": ("STRING", {"default": "-5,5"}),
                "roty_tremor_scale": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 1.0}),
                "rotz_range": ("STRING", {"default": "-5,5"}),
                "rotz_tremor_scale": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 1.0}),
            },
            "optional": {
                "opac_perlin_settings": ("DICT", {})
            }
        }

    RETURN_TYPES = ("LIST", "LIST", "LIST", "LIST", "LIST", "LIST", "LIST", "LIST")
    RETURN_NAMES = ("zoom_schdule_list", "angle_schdule_list", "translation_x_schdule_list", "translation_y_schdule_list", "translation_z_schdule_list", "rotation_3d_x_schdule_list", "rotation_3d_y_schdule_list", "rotation_3d_z_schdule_list")
    FUNCTION = "execute"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling"

    def process_kwargs(self, **kwargs):
        self.use_wiggle = kwargs.get('use_wiggle', True)
        self.frame_count = kwargs.get('frame_count', 48)
        self.zoom_range = tuple(map(float, kwargs.get('zoom_range', "0.95,1.05").split(',')))
        self.zoom_tremor_scale = kwargs.get('zoom_tremor_scale', 0.05)
        self.angle_range = tuple(map(float, kwargs.get('angle_range', "-5,5").split(',')))
        self.angle_tremor_scale = kwargs.get('angle_tremor_scale', 0.05)
        self.trx_range = tuple(map(float, kwargs.get('trx_range', "-10,10").split(',')))
        self.trx_tremor_scale = kwargs.get('trx_tremor_scale', 0.05)
        self.try_range = tuple(map(float, kwargs.get('try_range', "-10,10").split(',')))
        self.try_tremor_scale = kwargs.get('try_tremor_scale', 0.05)
        self.trz_range = tuple(map(float, kwargs.get('trz_range', "-10,10").split(',')))
        self.trz_tremor_scale = kwargs.get('trz_tremor_scale', 0.05)
        self.rotx_range = tuple(map(float, kwargs.get('rotx_range', "-5,5").split(',')))
        self.rotx_tremor_scale = kwargs.get('rotx_tremor_scale', 0.05)
        self.roty_range = tuple(map(float, kwargs.get('roty_range', "-5,5").split(',')))
        self.roty_tremor_scale = kwargs.get('roty_tremor_scale', 0.05)
        self.rotz_range = tuple(map(float, kwargs.get('rotz_range', "-5,5").split(',')))
        self.rotz_tremor_scale = kwargs.get('rotz_tremor_scale', 0.05)

        # Zoom Perlin settings
        self.zoom_octaves = kwargs.get('zoom_octaves', 1)
        self.zoom_persistence = kwargs.get('zoom_persistence', 0.5)
        self.zoom_lacunarity = kwargs.get('zoom_lacunarity', 2.0)
        self.zoom_repeat = kwargs.get('zoom_repeat', 1024)
            
        # Angle Perlin settings
        self.angle_octaves = kwargs.get('angle_octaves', 1)
        self.angle_persistence = kwargs.get('angle_persistence', 0.5)
        self.angle_lacunarity = kwargs.get('angle_lacunarity', 2.0)
        self.angle_repeat = kwargs.get('angle_repeat', 1024)
            
        # Translation Perlin settings (trx, try, trz)
        self.trx_octaves = kwargs.get('trx_octaves', 1)
        self.trx_persistence = kwargs.get('trx_persistence', 0.5)
        self.trx_lacunarity = kwargs.get('trx_lacunarity', 2.0)
        self.trx_repeat = kwargs.get('trx_repeat', 1024)
            
        self.try_octaves = kwargs.get('try_octaves', 1)
        self.try_persistence = kwargs.get('try_persistence', 0.5)
        self.try_lacunarity = kwargs.get('try_lacunarity', 2.0)
        self.try_repeat = kwargs.get('try_repeat', 1024)
            
        self.trz_octaves = kwargs.get('trz_octaves', 1)
        self.trz_persistence = kwargs.get('trz_persistence', 0.5)
        self.trz_lacunarity = kwargs.get('trz_lacunarity', 2.0)
        self.trz_repeat = kwargs.get('trz_repeat', 1024)
            
        # Rotation Perlin settings (rotx, roty, rotz)
        self.rotx_octaves = kwargs.get('rotx_octaves', 1)
        self.rotx_persistence = kwargs.get('rotx_persistence', 0.5)
        self.rotx_lacunarity = kwargs.get('rotx_lacunarity', 2.0)
        self.rotx_repeat = kwargs.get('rotx_repeat', 1024)
        
        self.roty_octaves = kwargs.get('roty_octaves', 1)
        self.roty_persistence = kwargs.get('roty_persistence', 0.5)
        self.roty_lacunarity = kwargs.get('roty_lacunarity', 2.0)
        self.roty_repeat = kwargs.get('roty_repeat', 1024)
            
        self.rotz_octaves = kwargs.get('rotz_octaves', 1)
        self.rotz_persistence = kwargs.get('rotz_persistence', 0.5)
        self.rotz_lacunarity = kwargs.get('rotz_lacunarity', 2.0)
        self.rotz_repeat = kwargs.get('rotz_repeat', 1024)

    #def sample_perlin(self, base, scale, x, min_val, max_val, octaves=1, persistence=0.5, lacunarity=2.0, repeat=1024):
    #    noise_val = self.perlin_noise.sample(base + x * scale, scale=1.0, octaves=octaves, persistence=persistence, lacunarity=lacunarity)
    #    return noise_val * (max_val - min_val) + min_val


    def sample_perlin(self, frame_index, range_min, range_max, tremor_scale, octaves, persistence, lacunarity, repeat):
        # Prepare noise correctly with normalization
        t = frame_index / (self.frame_count - 1 if self.frame_count > 1 else 1)
        linear_value = (range_max - range_min) * t + range_min
        noise = self.perlin_noise.sample(self.noise_base + frame_index * 0.1, scale=1.0, octaves=octaves, persistence=persistence, lacunarity=lacunarity)
        noise_adjustment = 1 + noise * tremor_scale
        interpolated_value = linear_value * noise_adjustment
        return interpolated_value

    def execute(self, **kwargs):

        if kwargs.__contains__("opac_perlin_settings"):
            perlin_settings = kwargs.pop("opac_perlin_settings")
            kwargs.update(perlin_settings)
            logger.info("\033[1m\033[94mOPAC Perlin Settings applied!:\033[0m")

        # Process the input values
        self.process_kwargs(**kwargs)

        if not self.use_wiggle:
            return ([0] * self.frame_count,) * 8

        # More dynamic implementation this time
        zoom, angle, translation_x, translation_y, translation_z, rotation_3d_x, rotation_3d_y, rotation_3d_z = (
            [self.sample_perlin(i, *param) for i in range(self.frame_count)]
            for param in [
                (self.zoom_range[0], self.zoom_range[1], self.zoom_tremor_scale, self.zoom_octaves, self.zoom_persistence, self.zoom_lacunarity, self.zoom_repeat),
                (self.angle_range[0], self.angle_range[1], self.angle_tremor_scale, self.angle_octaves, self.angle_persistence, self.angle_lacunarity, self.angle_repeat),
                (self.trx_range[0], self.trx_range[1], self.trx_tremor_scale, self.trx_octaves, self.trx_persistence, self.trx_lacunarity, self.trx_repeat),
                (self.try_range[0], self.try_range[1], self.try_tremor_scale, self.try_octaves, self.try_persistence, self.try_lacunarity, self.try_repeat),
                (self.trz_range[0], self.trz_range[1], self.trz_tremor_scale, self.trz_octaves, self.trz_persistence, self.trz_lacunarity, self.trz_repeat),
                (self.rotx_range[0], self.rotx_range[1], self.rotx_tremor_scale, self.rotx_octaves, self.rotx_persistence, self.rotx_lacunarity, self.rotx_repeat),
                (self.roty_range[0], self.roty_range[1], self.roty_tremor_scale, self.roty_octaves, self.roty_persistence, self.roty_lacunarity, self.roty_repeat),
                (self.rotz_range[0], self.rotz_range[1], self.rotz_tremor_scale, self.rotz_octaves, self.rotz_persistence, self.rotz_lacunarity, self.rotz_repeat)
            ]
        )
            
        logger.info("\033[1m\033[94mOPAC Schedule Curves:\033[0m")

        log_curve("zoom", zoom)
        log_curve("angle", angle)
        log_curve("translation_x", translation_x)
        log_curve("translation_y", translation_y)
        log_curve("translation_z", translation_z)
        log_curve("rotation_3d_x", rotation_3d_x)
        log_curve("rotation_3d_y", rotation_3d_y)
        log_curve("rotation_3d_z", rotation_3d_z)

        logger.info("")

        return zoom, angle, translation_x, translation_y, translation_z, rotation_3d_x, rotation_3d_y, rotation_3d_z

class SaltOPACPerlinSettings:
    """
        Configuration node for Perlin noise sampling in OPAC node
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "zoom_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "zoom_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "zoom_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "zoom_repeat": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "angle_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "angle_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "angle_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "angle_repeat": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "trx_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "trx_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "trx_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "trx_repeat": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "try_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "try_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "try_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "try_repeat": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "trz_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "trz_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "trz_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "trz_repeat": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "rotx_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "rotx_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "rotx_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "rotx_repeat": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "roty_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "roty_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "roty_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "roty_repeat": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "rotz_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "rotz_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "rotz_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "rotz_repeat": ("INT", {"default": 1024, "min": 256, "max": 4096}),
            }
        }
    
    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("opac_perlin_settings",)
    FUNCTION = "process"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling"

    def process(self, **kwargs):
        return (kwargs, )


class SaltScheduleConverter:
    """
        Converts a LIST input to FLOATS or FLOAT type
        Makes schedule lists compatible with MTB, IPAdapter, and other modules that use false types.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule_list": ("LIST",)
            },
        }
    
    RETURN_TYPES = ("FLOATS", "FLOAT", "INT")
    RETURN_NAMES = ("floats", "float", "int")
    FUNCTION = "convert"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling"

    def convert(self, schedule_list):
        int_schedule = [int(round(val)) for val in schedule_list]
        return (schedule_list, schedule_list, int_schedule)
    
class SaltScheduleVariance:
    """
    Applies Perlin noise and optional easing curves to each value in a list to create an OPAC Schedule out of it,
    while aiming to preserve the original distribution of the input values.
    """
    def __init__(self):
        self.noise_base = random.randint(0, 1000)
        self.perlin_noise = PerlinNoise()

    @classmethod
    def INPUT_TYPES(cls):
        easing_fn = list(easing_functions.keys())
        easing_fn.insert(0, "None")
        return {
            "required": {
                "schedule_list": ("LIST", {}),
            },
            "optional": {
                "curves_mode": (easing_fn,),
                "use_perlin_tremors": ("BOOLEAN", {"default": True}),
                "tremor_scale": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 1.0}),
                "octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("schedule_list",)
    FUNCTION = "opac_variance"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling"

    def sample_perlin(self, frame_index, value, tremor_scale, octaves, persistence, lacunarity):
        noise = self.perlin_noise.sample(self.noise_base + frame_index * 0.1, scale=1.0, octaves=octaves, persistence=persistence, lacunarity=lacunarity)
        noise_adjustment = 1 + noise * tremor_scale
        return max(0, min(value * noise_adjustment, 1))

    def opac_variance(self, schedule_list, curves_mode="None", use_perlin_tremors=True, tremor_scale=0.05, octaves=1, persistence=0.5, lacunarity=2.0):
        self.frame_count = len(schedule_list)
        varied_list = schedule_list.copy()

        if use_perlin_tremors:
            for i, value in enumerate(schedule_list):
                noise_adjusted_value = self.sample_perlin(i, value, tremor_scale, octaves, persistence, lacunarity)
                varied_list[i] = round(noise_adjusted_value, 2)

        if curves_mode != "None" and curves_mode in easing_functions:
            for i, value in enumerate(varied_list):
                curve_adjustment = easing_functions[curves_mode](i / max(1, (self.frame_count - 1)))
                # Apply curve adjustment to the original value, not to the noise-adjusted value
                original_value_with_curve = curve_adjustment * schedule_list[i]
                # Blend the original value with curves and noise-adjusted value
                varied_list[i] = round((value + original_value_with_curve) / 2, 2)

        return (varied_list,)
    

class SaltSchedule2ExecSchedule:
    """
        Converts a list to a list output (iterative execution list)
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_input": ("LIST", {}), 
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "convert"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling"

    def convert(self, list_input):
        return (list_input, )
    

class SaltLayerScheduler:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_count": ("INT", {"default": 60, "min": 1, "max": 4096}),
                "zoom_speed": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 1.0, "step": 0.001}),
                "pan_speed": ("FLOAT", {"default": 0.5, "min": 0.001, "max": 5.0, "step": 0.001}),
                "pan_directions": ("STRING", {"default": "90,180,270"}),
                "direction_change_frames": ("STRING", {"default": "10,20,40"}),
                "tremor_scale": ("FLOAT", {"default": 64, "min": 0.01, "max": 1024.0, "step": 0.01}),
                "tremor_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "tremor_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01}),
                "tremor_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 3.0, "step": 0.01}),
                "direction_curve": (list(easing_functions.keys()), ),
                "start_x": ("FLOAT", {"default": 0, "min": -nodes.MAX_RESOLUTION, "max": nodes.MAX_RESOLUTION}),
                "start_y": ("FLOAT", {"default": 0}),
                "zoom_mode": (["zoom-in", "zoom-out", "zoom-in-out"], ),
                "layer_offsets": ("STRING", {"default": "1,0.8,0.6"}),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("schedule_list",)
    FUNCTION = "execute"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling/Parallax Motion"

    def __init__(self):
        self.scheduler = None

    def execute(self, **kwargs):

        self.process_kwargs(**kwargs)

        if not self.scheduler:
            raise Exception("Camera Scheduler was not initialized. Perhaps your settings are bugged?")
        
        mode = kwargs.get("zoom_mode", "zoom-in")
        layer_offsets = kwargs.get("layer_offsets", "1")

        animation_data = self.scheduler.animate(mode, layer_offsets)

        return (animation_data, )

    def process_kwargs(self, **kwargs):
        frame_count = kwargs.get("frame_count", 60)
        zoom_speed = kwargs.get("zoom_speed", 0.01)
        pan_speed = kwargs.get("pan_speed", 0.5)
        pan_directions = list(map(float, kwargs.get("pan_directions", "90,180,270").split(",")))
        direction_change_frames = list(map(int, kwargs.get("direction_change_frames", "10,20,40").split(",")))
        tremor_params = {
            "scale": kwargs.get("tremor_scale", 0.1),
            "octaves": kwargs.get("tremor_octaves", 1),
            "persistence": kwargs.get("tremor_persistence", 0.5),
            "lacunarity": kwargs.get("tremor_lacunarity", 2.0),
        }
        direction_curve = kwargs.get("direction_curve", "linear")
        start_x = kwargs.get("start_x", 0)
        start_y = kwargs.get("start_y", 0)

        self.scheduler = OrganicPerlinCameraScheduler(
            frame_count=frame_count,
            zoom_speed=zoom_speed,
            pan_speed=pan_speed,
            pan_directions=pan_directions,
            direction_change_frames=direction_change_frames,
            tremor_params=tremor_params,
            direction_curve=direction_curve,
            start_x=start_x,
            start_y=start_y,
        )

class SaltLayerExtractor:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "float_layers": ("LIST", ),
                "layer_index": ("INT", {"default": 0, "min": 0})
            }
        }

    RETURN_TYPES = ("LIST", "LIST", "LIST")
    RETURN_NAMES = ("zoom_schedule_lsit", "x_schedule_list", "y_schedule_list")
    FUNCTION = "extract"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling/Parallax Motion"

    def extract(self, **kwargs):
        animation_data = kwargs.get("float_layers", [])
        layer_index = kwargs.get("layer_index", 0)

        if layer_index >= len(animation_data):
            raise ValueError("Layer index out of range.")

        selected_layer_data = animation_data[layer_index]
        zoom_values = [frame[0] for frame in selected_layer_data]
        x_values = [frame[1] for frame in selected_layer_data]
        y_values = [frame[2] for frame in selected_layer_data]

        logger.info("\033[1m\033[94mOPAC Schedule Curves:\033[0m")
        log_curve("Zoom Values", zoom_values)
        log_curve("X Values", x_values)
        log_curve("Y Values", y_values)

        return zoom_values, x_values, y_values


class SaltParallaxMotion:
    """
    A node for generating min/max FLOAT values for Front and Back layers across X, Y, and Z axes
    in 2D plane translations to create a parallax effect, based on selected presets or custom input values.
    Adjusts these values based on the parallax intensity.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "zoom_preset": (["None", "Zoom In", "Zoom Out", "Zoom In/Out", "Zoom Out/In", "Custom", "Select Random"], {
                    "default": "None"
                }),
                "horizontal_pan_preset": (["None", "Pan Left → Right", "Pan Right → Left", "Pan Left → Center", 
                                        "Pan Right → Center", "Pan Center → Right", "Pan Center → Left", "Custom", "Select Random"], {
                    "default": "None"
                }),
                "vertical_pan_preset": (["None", "Pan Up → Down", "Pan Down → Up", "Pan Up → Center", 
                                        "Pan Down → Center", "Pan Center → Up", "Pan Center → Down", "Custom", "Select Random"], {
                    "default": "None"
                }),
                "custom_x_min": ("FLOAT", {"default": 0.0}),
                "custom_x_max": ("FLOAT", {"default": 0.0}),
                "custom_y_min": ("FLOAT", {"default": 0.0}),
                "custom_y_max": ("FLOAT", {"default": 0.0}),
                "custom_z_min": ("FLOAT", {"default": 1.0}),
                "custom_z_max": ("FLOAT", {"default": 1.0}),
                "parallax_intensity": ("FLOAT", {"default": 1.0}),
                "zoom_intensity": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0}),
            },
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("front_x_min", "front_x_max", "front_y_min", "front_y_max", "front_z_min", "front_z_max", "back_x_min", "back_x_max", "back_y_min", "back_y_max", "back_z_min", "back_z_max")
    FUNCTION = "generate_parameters"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling/Parallax Motion"

    def generate_parameters(self, zoom_preset, horizontal_pan_preset, vertical_pan_preset, 
                        custom_x_min, custom_x_max, custom_y_min, custom_y_max, 
                        custom_z_min, custom_z_max, parallax_intensity, zoom_intensity):
            # Handle random selection for zoom presets
        if zoom_preset == "Select Random":
            zoom_options = [key for key in zoom_presets.keys() if key not in ["None", "Custom", "Select Random"]]
            zoom_preset = random.choice(zoom_options)
        # Apply the selected or randomly chosen zoom preset
        if zoom_preset in zoom_presets:
            z_min, z_max = zoom_presets[zoom_preset]

        # Handle random selection for horizontal pan presets
        if horizontal_pan_preset == "Select Random":
            horizontal_options = [key for key in horizontal_pan_presets.keys() if key not in ["None", "Custom", "Select Random"]]
            horizontal_pan_preset = random.choice(horizontal_options)
        # Apply the selected or randomly chosen horizontal pan preset
        if horizontal_pan_preset in horizontal_pan_presets:
            x_min, x_max = horizontal_pan_presets[horizontal_pan_preset]

        # Handle random selection for vertical pan presets
        if vertical_pan_preset == "Select Random":
            vertical_options = [key for key in vertical_pan_presets.keys() if key not in ["None", "Custom", "Select Random"]]
            vertical_pan_preset = random.choice(vertical_options)
        # Apply the selected or randomly chosen vertical pan preset
        if vertical_pan_preset in vertical_pan_presets:
            y_min, y_max = vertical_pan_presets[vertical_pan_preset]
        
        # Initialize default axis values
        x_min, x_max = 0, 0
        y_min, y_max = 0, 0
        z_min, z_max = 1, 1  # Default Z values assume no zoom (scale of 1)
        
        # Apply the selected zoom preset
        if zoom_preset in zoom_presets:
            z_min, z_max = zoom_presets[zoom_preset]
        
        # Apply the selected horizontal pan preset
        if horizontal_pan_preset in horizontal_pan_presets:
            x_min, x_max = horizontal_pan_presets[horizontal_pan_preset]
        
        # Apply the selected vertical pan preset
        if vertical_pan_preset in vertical_pan_presets:
            y_min, y_max = vertical_pan_presets[vertical_pan_preset]
        
        # For 'Custom' selections, override with custom values
        if zoom_preset == "Custom":
            z_min, z_max = custom_z_min, custom_z_max
        if horizontal_pan_preset == "Custom":
            x_min, x_max = custom_x_min, custom_x_max
        if vertical_pan_preset == "Custom":
            y_min, y_max = custom_y_min, custom_y_max

        # Calculate the back layer's values based on the parallax intensity
        back_x_min = x_min / parallax_intensity if x_min != 0 else 0
        back_x_max = x_max / parallax_intensity if x_max != 0 else 0
        back_y_min = y_min / parallax_intensity if y_min != 0 else 0
        back_y_max = y_max / parallax_intensity if y_max != 0 else 0
        # Z values are not adjusted by parallax intensity
        back_z_min, back_z_max = z_min, z_max

        # Adjust for zoom intensity for Z axis to increase front zoom relative to back with higher intensity values
        # This approach assumes the front layer's zoom can increase more than the back's as zoom_intensity increases.
        # Adjust the formula as needed to align with your specific vision for the effect.
        adjusted_front_z_min = z_min * (1 + zoom_intensity - 1)  # Example adjustment, enhances the front zoom based on intensity
        adjusted_front_z_max = z_max * (1 + zoom_intensity - 1)  # Similarly for max zoom

        # The back layer's zoom could remain unchanged, or you could apply a different formula if the back layer also needs adjustment.
        back_z_min, back_z_max = z_min, z_max  # Keeps back layer's zoom unchanged; modify as needed.


        return (x_min, x_max, y_min, y_max, adjusted_front_z_min, adjusted_front_z_max, back_x_min, back_x_max, back_y_min, back_y_max, back_z_min, back_z_max)


class SaltFloatScheduler:
    @classmethod
    def INPUT_TYPES(cls):
        easing_fns = list(easing_functions.keys())
        easing_fns.insert(0, "None")
        return {
            "required": {
                "repeat_sequence_times": ("INT", {"default": 0, "min": 0}),
                "curves_mode": (easing_fns, ),
                "use_perlin_tremors": ("BOOLEAN", {"default": True}),
                "tremor_scale": ("FLOAT", {"default": 64, "min": 0.01, "max": 1024.0, "step": 0.01}),
                "tremor_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "tremor_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01}),
                "tremor_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 3.0, "step": 0.01}),
                "sequence": ("STRING", {"multiline": True, "placeholder": "[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]"}),
            },
            "optional": {
                "max_sequence_length": ("INT", {"default": 0, "min": 0, "max": 4096}),
            }
        }

    RETURN_TYPES = ("LIST", "INT")
    RETURN_NAMES = ("schedule_list", "schedule_length")
    FUNCTION = "generate_sequence"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling"

    def apply_curve(self, sequence, mode):
        if mode in easing_functions.keys():
            sequence = [easing_functions[mode](t) for t in sequence]
        else:
            logger.error(f"The easing mode `{mode}` does not exist in the valid easing functions: {', '.join(easing_functions.keys())}")
        return sequence

    def apply_perlin_noise(self, sequence, scale, octaves, persistence, lacunarity):
        perlin = PerlinNoise()
        noise_values = [perlin.sample(i, scale=scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity) for i, _ in enumerate(sequence)]
        sequence = [val + noise for val, noise in zip(sequence, noise_values)]
        return sequence

    def generate_sequence(self, sequence, repeat_sequence_times, curves_mode, use_perlin_tremors, tremor_scale, tremor_octaves, tremor_persistence, tremor_lacunarity, max_sequence_length=0):
        sequence_list = [float(val.strip()) for val in sequence.replace("[", "").replace("]", "").split(',')]
        if use_perlin_tremors:
            sequence_list = self.apply_perlin_noise(sequence_list, tremor_scale, tremor_octaves, tremor_persistence, tremor_lacunarity)
        if curves_mode != "None":
            sequence_list = self.apply_curve(sequence_list, curves_mode)
        sequence_list = sequence_list * (repeat_sequence_times + 1)
        sequence_list = sequence_list[:max_sequence_length] if max_sequence_length != 0 else sequence_list
        return (sequence_list, len(sequence_list))


# CONDITIONING AND SAMPLING
    
class SaltKSamplerSequence:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed_sequence": ("LIST", ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "denoise_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "denoise_sequence": ("LIST", ),
                "positive_sequence": ("CONDITIONING",),
                "negative_sequence": ("CONDITIONING",),
                "use_latent_interpolation": ("BOOLEAN", {"default": False}),
                "latent_interpolation_mode": (["Blend", "Slerp", "Cosine Interp"],),
                "latent_interp_strength_sequence": ("LIST", ),
                "unsample_latents": ("BOOLEAN", {"default": True}),
                "inject_noise": ("BOOLEAN", {"default": True}),
                "noise_strength_sequence": ("LIST", ),
                "latent_image": ("LATENT", ),
            }
        }

    RETURN_TYPES = ("LATENT",)

    FUNCTION = "sample"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling/Sampling"

    def inject_noise(self, latent_image, noise_strength):
        noise = torch.randn_like(latent_image) * noise_strength
        return latent_image + noise
    
    def expand_sequence(self, sequence, target_length):
        if len(sequence) >= target_length:
            return sequence
        repeated_sequence = (sequence * (target_length // len(sequence) + 1))[:target_length]
        return repeated_sequence

    def sample(self, model, seed_sequence, steps, cfg, sampler_name, scheduler, positive_sequence, negative_sequence, latent_image, 
               use_latent_interpolation, latent_interpolation_mode, latent_interp_strength_sequence, unsample_latents, denoise_start, 
               denoise_sequence, inject_noise, noise_strength_sequence):
        
        if not isinstance(positive_sequence, list):
            positive_sequence = [positive_sequence]
        if not isinstance(negative_sequence, list):
            negative_sequence = [negative_sequence]

        sequence_loop_count = len(positive_sequence)

        if len(negative_sequence) < len(positive_sequence):
            raise ValueError(f"`negative_sequence` of size {len(negative_sequence)} does not match `positive_sequence` of size {len(positive_sequence)}. Conditioning sizes must be the same.")

        # Schedule Value Lists
        seed_sequence = [int(seed) for seed in seed_sequence]
        denoise_sequence = [float(denoise_val) for denoise_val in denoise_sequence]
        latent_interp_strength_sequence = [float(latent_strength) for latent_strength in latent_interp_strength_sequence]
        noise_strength_sequence = [float(noise_strength) for noise_strength in noise_strength_sequence]

        # Expanding all sequences if necessary
        if len(denoise_sequence) < sequence_loop_count:
            denoise_sequence = self.expand_sequence(denoise_sequence, sequence_loop_count)
        if len(latent_interp_strength_sequence) < sequence_loop_count:
            latent_interp_strength_sequence = self.expand_sequence(latent_interp_strength_sequence, sequence_loop_count)
        if len(noise_strength_sequence) < sequence_loop_count:
            noise_strength_sequence = self.expand_sequence(noise_strength_sequence, sequence_loop_count)
        if len(seed_sequence) < sequence_loop_count:
            seed_sequence = self.expand_sequence(seed_sequence, sequence_loop_count)

        results = []

        sequence_loop_count = len(positive_sequence)

        logger.info(f"Starting loop sequence with {sequence_loop_count} frames.")

        positive_conditioning = None
        negative_conditioning = None
        latent_mask = latent_image.get("noise_mask", None)

        for loop_count in range(sequence_loop_count):

            positive_conditioning = [positive_sequence[loop_count]]
            negative_conditioning = [negative_sequence[loop_count]]

            if results and len(results) > 0:
                if len(latent_input) == 1 or latent_mask == None:
                    latent_input = {'samples': results[-1]}
                else:
                    latent_input = {'samples': latent_image["samples"][loop_count if loop_count < len(latent_image) else -1].unsqueeze(0)}
                if isinstance(latent_mask, torch.Tensor):
                    latent_input["noise_mask"] = latent_mask
                start_at_step = round((1 - denoise) * steps)
                end_at_step = steps
            else:
                latent_copy = latent_image.copy()
                if isinstance(latent_mask, torch.Tensor):
                    latent_copy["samples"] = latent_copy["samples"][0].unsqueeze(0)
                latent_input = latent_copy

            denoise = denoise_sequence[loop_count] if loop_count > 0 else denoise_start

            if inject_noise and loop_count > 0:
                logger.info(f"Injecting noise at {noise_strength_sequence[loop_count]} strength.")
                latent_input['samples'] = self.inject_noise(latent_input['samples'], noise_strength_sequence[loop_count])

            if unsample_latents and loop_count > 0:
                force_full_denoise = not (loop_count > 0 or loop_count <= steps - 1)
                disable_noise = False
                logger.info("Unsampling latent image.")
                unsampled_latent = unsample(model=model, seed=seed_sequence[loop_count], cfg=cfg, sampler_name=sampler_name, steps=steps+1, end_at_step=steps, scheduler=scheduler, normalize=False, positive=positive_conditioning, negative=negative_conditioning, latent_image=latent_input)[0]
                if inject_noise and loop_count > 0:
                    logger.info(f"Injecting noise at {noise_strength_sequence[loop_count]} strength.")
                    unsampled_latent['samples'] = self.inject_noise(unsampled_latent['samples'], noise_strength_sequence[loop_count])
                logger.info(f"Sampling Denoise: {denoise}")
                logger.info("Sampling.")
                sample = nodes.common_ksampler(model, seed_sequence[loop_count], steps, cfg, sampler_name, scheduler, positive_conditioning, negative_conditioning, unsampled_latent, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)[0]['samples']
            else:

                if inject_noise and loop_count > 0:
                    logger.info(f"Injecting noise at {noise_strength_sequence[loop_count]} strength.")
                    latent_input['samples'] = self.inject_noise(latent_input['samples'], noise_strength_sequence[loop_count])
                sample = nodes.common_ksampler(model, seed_sequence[loop_count], steps, cfg, sampler_name, scheduler, positive_conditioning, negative_conditioning, latent_input, denoise=denoise)[0]['samples']

            if use_latent_interpolation and results and loop_count > 0:
                if latent_interpolation_mode == "Blend":
                    sample = blend_latents(latent_interp_strength_sequence[loop_count], results[-1], sample)
                elif latent_interpolation_mode == "Slerp":
                    sample = slerp_latents(latent_interp_strength_sequence[loop_count], results[-1], sample)
                elif latent_interpolation_mode == "Cosine Interp":
                    sample = cosine_interp_latents(latent_interp_strength_sequence[loop_count], results[-1], sample)

            results.append(sample)

        results = torch.cat(results, dim=0)
        results = {'samples': results}
        if isinstance(latent_mask, torch.Tensor):
            results["noise_mask"] = latent_mask.repeat(len(results["samples"]), 1, 1, 1)
        
        return (results,)
    

class SaltCLIPTextEncodeSequence:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "token_normalization": (["none", "mean", "length", "length+mean"],),
                "weight_interpretation": (["comfy", "A1111", "compel", "comfy++"],),
                "frame_count": ("INT", {"default": 100, "min": 1, "max": 1024, "step": 1}),
                "text": ("STRING", {"multiline": True, "placeholder": '''"0": "A portrait of a rosebud",
"25": "A portrait of a blooming rosebud",
"50": "A portrait of a blooming rose",
"75": "A portrait of a rose"'''}),
            }
        }
        
    RETURN_TYPES = ("CONDITIONING", "INT")
    RETURN_NAMES = ("conditioning_sequence", "frame_count")

    FUNCTION = "encode"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling/Conditioning"

    def encode(self, clip, text, frame_count, token_normalization, weight_interpretation):
        
        try:
            text_dict = json.loads("{"+text+"}")
        except json.JSONDecodeError as e:
            raise ValueError("Unable to decode propmt schedule:", e)

        conditionings = []
        prev_frame_num = 0
        prev_encoded = None
        for frame_num, prompt in sorted(text_dict.items(), key=lambda item: int(item[0])):
            frame_num = int(frame_num)
            if USE_BLK:
                encoded = blk_adv.encode(clip=clip, text=prompt, token_normalization=token_normalization, weight_interpretation=weight_interpretation)
            else:
                encoded = CLIPTextEncode.encode(clip=clip, text=prompt)
            for _ in range(prev_frame_num, frame_num):
                conditionings.append(prev_encoded)
            prev_encoded = [encoded[0][0][0], encoded[0][0][1]]
            prev_frame_num = frame_num
        for _ in range(prev_frame_num, frame_count):
            conditionings.append(prev_encoded)

        conditioning_count = len(conditionings)

        return (conditionings, conditioning_count)

    def cond_easing(self, type, frame_count, conditioning_count):
        if type == "linear":
            return np.linspace(frame_count // conditioning_count, frame_count, conditioning_count, dtype=int).tolist()
        elif type == "sinus":
            t = np.linspace(0, np.pi, conditioning_count)
            sinus_values = np.sin(t)
            normalized_values = (sinus_values - sinus_values.min()) / (sinus_values.max() - sinus_values.min())
            scaled_values = normalized_values * (frame_count - 1) + 1
            unique_keyframes = np.round(scaled_values).astype(int)
            unique_keyframes = np.unique(unique_keyframes, return_index=True)[1]
            return sorted(unique_keyframes.tolist())
        elif type == "sinus_inverted":
            return (np.cos(np.linspace(0, np.pi, conditioning_count)) * (frame_count - 1) + 1).astype(int).tolist()
        elif type == "half_sinus":
            return (np.sin(np.linspace(0, np.pi / 2, conditioning_count)) * (frame_count - 1) + 1).astype(int).tolist()
        elif type == "half_sinus_inverted":
            return (np.cos(np.linspace(0, np.pi / 2, conditioning_count)) * (frame_count - 1) + 1).astype(int).tolist()
        else:
            raise ValueError("Unsupported cond_keyframes_type: " + type)


class SaltConditioningSetMaskAndCombine:
    """
    Based on KJNodes ConditioningSetMaskAndCombine
    https://github.com/kijai/ComfyUI-KJNodes/blob/671af53b34f13d35526a510dfbbaac253ddd52da/nodes.py#L1256
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_schedule_a": ("CONDITIONING", ),
                "negative_schedule_a": ("CONDITIONING", ),
                "positive_schedule_b": ("CONDITIONING", ),
                "negative_schedule_b": ("CONDITIONING", ),
                "mask_a": ("MASK", ),
                "mask_b": ("MASK", ),
            },
            "optional": {
                "mask_strengths_a": ("LIST", {}),
                "mask_strengths_b": ("LIST", {}),
                "set_cond_area": (["default", "mask bounds"],),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    FUNCTION = "process"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling/Conditioning"

    def process(self, positive_schedule_a, negative_schedule_a, positive_schedule_b, negative_schedule_b, mask_a, mask_b, mask_strengths_a=[1], mask_strengths_b=[1], set_cond_area="default"):
        set_area_to_bounds = set_cond_area != "default"

        combined_positive_a = self.apply_masks(positive_schedule_a, mask_a, mask_strengths_a, set_area_to_bounds)
        combined_negative_a = self.apply_masks(negative_schedule_a, mask_a, mask_strengths_a, set_area_to_bounds)
        combined_positive_b = self.apply_masks(positive_schedule_b, mask_b, mask_strengths_b, set_area_to_bounds)
        combined_negative_b = self.apply_masks(negative_schedule_b, mask_b, mask_strengths_b, set_area_to_bounds)

        combined_positive = combined_positive_a + combined_positive_b
        combined_negative = combined_negative_a + combined_negative_b

        return (combined_positive, combined_negative)

    def apply_masks(self, conditionings, mask, strengths, set_area_to_bounds):
        combined = []
        if len(mask.shape) < 3:
            mask = mask.unsqueeze(0)

        for idx, conditioning in enumerate(conditionings):
            strength = strengths[idx % len(strengths)]

            combined.append(self.append_helper(conditioning, mask, set_area_to_bounds, strength))
        return combined

    def append_helper(self, conditioning, mask, set_area_to_bounds, strength):
        conditioned = [conditioning[0], conditioning[1].copy()]
        _, h, w = mask.shape
        conditioned[1]['mask'] = mask
        conditioned[1]['set_area_to_bounds'] = set_area_to_bounds
        conditioned[1]['mask_strength'] = strength
        return conditioned


class SaltThresholdSchedule:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "float_schedule": ("LIST", ),
                "max_frames": ("INT", {"min": 1, "max": 4096, "default": 16}),
                "output_mode": (["prompt_schedule", "float_list", "int_list", "raw"],),
                "schedule_values": ("STRING", {"multiline": True, "default": '''"0.0": "A beautiful forest, (green:1.2) color scheme",
"0.5": "A beautiful forest, (autumn:1.2) color scheme",
"1.0": "A beautiful forest, (winter:1.2) color scheme"'''}),
            }
        }

    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = ("value_schedule_list",)

    FUNCTION = "generate_sequence"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling"

    def generate_sequence(self, float_schedule, schedule_values, max_frames, output_mode):
        try:
            text_dict = json.loads("{" + schedule_values + "}")
        except json.JSONDecodeError as e:
            raise ValueError("Unable to decode prompt schedule:", e)

        prompt_sequence = []
        prompt_dict = dict(sorted({float(key): value for key, value in text_dict.items()}.items(), key=lambda x: x[0]))
        default_prompt = next(iter(prompt_dict.values())) if prompt_dict else None

        adjusted_float_schedule = [float_schedule[i] if i < len(float_schedule) else 0 for i in range(max_frames)]

        for float_val in adjusted_float_schedule:
            closest_prompt = default_prompt
            min_diff = float('inf')
            
            for prompt_key, prompt_val in prompt_dict.items():
                diff = abs(float_val - prompt_key)
                
                if diff < min_diff:
                    closest_prompt = prompt_val
                    min_diff = diff

            prompt_sequence.append(closest_prompt)

        if output_mode == "prompt_schedule":
            output = ", ".join(f'"{i}": "{prompt}"' for i, prompt in enumerate(prompt_sequence))
        elif output_mode == "float_list":
            output = [float(value) for value in prompt_sequence]
        else:
            output = [int(value) for value in prompt_sequence]

        return (output, )  


class SaltListOperation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "operation": (
                    ["add", "subtract", "multiply", "divide", "average", "max", "min", "normalize_a",
                     "logarithmic", "exponential", "sin", "cos", "tan", "arcsin", "arccos", "arctan",
                     "percentage_of", "modulo", "custom_expression"],
                ),
                "output_type": (["default", "int", "float", "boolean"],)
            },
            "optional": {
                "schedule_list_a": (WILDCARD,),
                "schedule_list_b": (WILDCARD,),
                "expression": ("STRING", {"default": "", "placeholder": "Custom expression...", "multiline": True})
            }
        }

    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = ("schedule_list",)
    FUNCTION = "calculate"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling/Filter"

    def calculate(self, operation, output_type, schedule_list_a=[0], schedule_list_b=[0], expression=""):
        if not isinstance(schedule_list_a, list):
            schedule_list_a = [schedule_list_a]
        if not isinstance(schedule_list_b, list):
            schedule_list_b = [schedule_list_b]

        # expand the lists to match the longest list
        longest_list_length = max(len(schedule_list_a), len(schedule_list_b))
        if len(schedule_list_a) < longest_list_length:
            schedule_list_a += [schedule_list_a[-1]] * (longest_list_length - len(schedule_list_a))
        if len(schedule_list_b) < longest_list_length:
            schedule_list_b += [schedule_list_b[-1]] * (longest_list_length - len(schedule_list_b))

        result_list = []

        if operation != "custom_expression":
            for i in range(longest_list_length):
                a = schedule_list_a[i]
                b = schedule_list_b[i] if schedule_list_b is not None else 0

                match operation:
                    case "add":
                        result_list.append(a + b)
                    case "subtract":
                        result_list.append(a - b)
                    case "multiply":
                        result_list.append(a * b)
                    case "divide":
                        result_list.append(a / b if b != 0 else 0)
                    case "average":
                        result_list.append((a + b) / 2)
                    case "max":
                        result_list.append(max(a, b))
                    case "min":
                        result_list.append(min(a, b))
                    case "logarithmic":
                        result_list.append(math.log(a) if a > 0 else 0)
                    case "exponential":
                        result_list.append(math.exp(a))
                    case "sin":
                        result_list.append(math.sin(a))
                    case "cos":
                        result_list.append(math.cos(a))
                    case "tan":
                        result_list.append(math.tan(a))
                    case "arcsin":
                        result_list.append(math.asin(a) if -1 <= a <= 1 else 0)
                    case "arccos":
                        result_list.append(math.acos(a) if -1 <= a <= 1 else 0)
                    case "arctan":
                        result_list.append(math.atan(a))
                    case "percentage_of":
                        result_list.append(a * (b / 100))
                    case "modulo":
                        result_list.append(a % b if b != 0 else 0)
        else:
            try:
                # Evaluate custom expression
                result_list = [
                    safe_eval(expression, custom_vars={"a": a_val, "b": b_val}) 
                    for a_val, b_val in zip(schedule_list_a, schedule_list_b)
                ]
            except Exception as e:
                raise ValueError(f"Unable to evaluate given expression `{expression}`: {e}")

        if operation == "normalize_a":
            max_val = max(schedule_list_a)
            min_val = min(schedule_list_a)
            range_val = max_val - min_val
            if range_val > 0:
                normalized_result_list = []
                for a in schedule_list_a:
                    normalized = (a - min_val) / range_val
                    normalized_result_list.append(normalized)
                result_list = normalized_result_list
            else:
                result_list = schedule_list_a

        # Output type conversion
        match(output_type):
            case "int":
                result_list = [int(val) for val in result_list]
            case "float":
                result_list = [float(val) for val in result_list]
            case "boolean":
                result_list = [bool(val) for val in result_list]

        return (result_list[0],) if len(result_list) == 1 else (result_list,)


class SaltListClamp:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule_list": ("LIST", ),
                "min_value": ("FLOAT", {"step": 0.01}),
                "max_value": ("FLOAT", {"step": 0.01})
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("schedule_list",)
    FUNCTION = "clamp_values"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling/Filter"

    def clamp_values(self, schedule_list, min_value, max_value):
        if min_value > max_value:
            raise ValueError("Schedules min_value cannot be greater than max_value.")

        clamped_list = [max(min(x, max_value), min_value) for x in schedule_list]

        return (clamped_list, )


class SaltListLinearInterpolation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule_list_a": ("LIST", ),
                "schedule_list_b": ("LIST", ),
                "interpolation_factor": ("FLOAT", {"min": 0.0, "max": 1.0}),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("schedule_list",)
    FUNCTION = "lerp"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling/Filter"

    def lerp(self, schedule_list_a, schedule_list_b, interpolation_factor):
        if len(schedule_list_a) != len(schedule_list_b):
            raise ValueError("Schedule lists must have the same length.")
        
        interpolated_list = []
        
        for a, b in zip(schedule_list_a, schedule_list_b):
            interpolated_value = (1 - interpolation_factor) * a + interpolation_factor * b
            interpolated_list.append(interpolated_value)

        return (interpolated_list, )


class SaltScheduleListExponentialFade:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule_list": ("LIST", ),
                "fade_type": (["in", "out", "in-and-out"],),
                "strength": ("FLOAT", {"min": 0.01, "max": 10.0, "default": 1.0}),
            },
            "optional": {
                "start_index": ("INT", {"min": 0, "default": 0}),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("schedule_list", )
    FUNCTION = "exponential_fade"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling/Filter"

    def exponential_fade(self, schedule_list, fade_type, strength, start_index=0):
        length = len(schedule_list)
        faded_schedule = []
        
        for i in range(length):
            if i < start_index:
                faded_schedule.append(schedule_list[i])
                continue

            if fade_type in ["in", "out"]:
                t = (i - start_index) / max(1, (length - 1 - start_index))
                if fade_type == "in":
                    value = t ** strength
                else:
                    value = ((1 - t) ** strength)
            elif fade_type == "in-and-out":
                midpoint = (length - start_index) // 2 + start_index
                if i <= midpoint:
                    t = (i - start_index) / max(1, (midpoint - start_index))
                    value = t ** strength
                else:
                    t = (i - midpoint) / max(1, (length - 1 - midpoint))
                    value = ((1 - t) ** strength)

            faded_schedule.append(value)
        
        faded_schedule = [original * fade for original, fade in zip(schedule_list, faded_schedule)]

        return (faded_schedule, )


class SaltScheduleRandomValues:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "count": ("INT", {"min": 1}),
                "min_value": ("FLOAT", {}),
                "max_value": ("FLOAT", {}),
            },
            "optional": {
                "force_integer": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("schedule_list", )
    FUNCTION = "generate_random"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling"

    def generate_random(self, count, min_value, max_value, force_integer=False):
        if min_value >= max_value:
            raise ValueError("Schedule min_value must be less than max_value.")
        
        output_type = int if force_integer or (isinstance(min_value, int) and isinstance(max_value, int)) else float
        
        if output_type == int:
            random_schedule = [random.randint(int(min_value), int(max_value)) for _ in range(count)]
        else:
            random_schedule = [random.uniform(min_value, max_value) for _ in range(count)]

        return (random_schedule, )
    

class SaltScheduleSmoothing:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule_list": ("LIST", ),
                "smoothing_factor": ("FLOAT", {"min": 0.0, "max": 1.0, "default": 0.5}),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("schedule_list", )
    FUNCTION = "smooth"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling/Filter"

    def smooth(self, schedule_list, smoothing_factor):
        smoothed_schedule = schedule_list[:]
        for i in range(1, len(schedule_list)):
            smoothed_schedule[i] = smoothed_schedule[i-1] * (1 - smoothing_factor) + schedule_list[i] * smoothing_factor
        return (smoothed_schedule, )
    
class SaltCyclicalSchedule:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule_list": ("LIST", ),
                "start_index": ("INT", {"min": 0}),
                "end_index": ("INT", {"min": 0}),
                "repetitions": ("INT", {"min": 1}),
            },
            "optional": {
                "ping_pong": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("schedule_list",)
    FUNCTION = "generate_cyclical"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling/Filter"

    def generate_cyclical(self, schedule_list, start_index, end_index, repetitions, ping_pong=False):
        if end_index < start_index:
            raise ValueError("Schedule end_index must be greater than or equal to start_index.")
        
        if end_index >= len(schedule_list):
            raise ValueError("Schedule end_index must be within the range of the schedule_list.")
        
        loop_segment = schedule_list[start_index:end_index + 1]
        
        cyclical_schedule = []
        for _ in range(repetitions):
            cyclical_schedule.extend(loop_segment)
            if ping_pong:
                cyclical_schedule.extend(loop_segment[-2:0:-1])
        
        return (cyclical_schedule,)
    

class SaltScheduleSplit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule_list": ("LIST", ),
                "split_index": ("INT", {"min": 0}),
            },
        }

    RETURN_TYPES = ("LIST", "LIST")
    RETURN_NAMES = ("schedule_list_a", "schedule_list_b")
    FUNCTION = "split"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling"

    def split(self, schedule_list, split_index):
        if split_index >= len(schedule_list) or split_index < 0:
            raise ValueError("Schedule split_index must be within the range of the schedule_list.")
        first_part = schedule_list[:split_index]
        second_part = schedule_list[split_index:]
        return (first_part, second_part)


class SaltScheduleMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule_list_a": ("LIST", ),
                "schedule_list_b": ("LIST", ),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("schedule_list", )
    FUNCTION = "append"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling/Filter"

    def append(self, schedule_list_a, schedule_list_b):
        appended_list = schedule_list_a + schedule_list_b
        return (appended_list, )


class SaltKeyframeVisualizer:
    @classmethod
    def INPUT_TYPES(cls):
        input_types = {
            "required": {
                "schedule_list": ("LIST",),
            },
            "optional": {
                "start_frame": ("INT", {"min": 0, "default": 0}),
                "end_frame": ("INT", {"min": 0, "default": -1}),
                "simulate_stereo": ("BOOLEAN", {"default": False}),
                "frame_rate": ("INT", {"min": 1, "default": 24}),
                "schedule_list_a": ("LIST", {"default": None}),
                "schedule_list_b": ("LIST", {"default": None}),
                "schedule_list_c": ("LIST", {"default": None}),
            }
        }
        return input_types

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True
    FUNCTION = "visualize_keyframes"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling/Util"

    def visualize_keyframes(self, schedule_list, start_frame=0, end_frame=-1, simulate_stereo=False, frame_rate=24.0, schedule_list_a=None, schedule_list_b=None, schedule_list_c=None):
        TEMP = folder_paths.get_temp_directory()
        os.makedirs(TEMP, exist_ok=True)

        schedule_lists = [schedule_list, schedule_list_a, schedule_list_b, schedule_list_c]
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        schedule_names = ['Schedule List', 'Schedule List A', 'Schedule List B', 'Schedule List C']
        metrics_data = []

        plt.figure(figsize=(10, 14 if simulate_stereo else 8))

        for i, sched_list in enumerate(schedule_lists):
            if sched_list is not None:
                if end_frame == -1 or end_frame > len(sched_list):
                    end_frame = len(sched_list)

                num_frames = max(2, end_frame - start_frame)
                frames = np.linspace(start_frame, end_frame - 1, num=num_frames, endpoint=True)
                keyframe_values = np.array(sched_list[start_frame:end_frame])

                plt.plot(frames, keyframe_values, color=colors[i], linewidth=0.5, label=schedule_names[i] + ' Left')
                if simulate_stereo:
                    plt.plot(frames, -keyframe_values, color=colors[i], linewidth=0.5, linestyle='dashed', label=schedule_names[i] + ' Right')
                    plt.fill_between(frames, keyframe_values, 0, color=colors[i], alpha=0.3)
                    plt.fill_between(frames, -keyframe_values, 0, color=colors[i], alpha=0.3)

                metrics = {
                    "Max": np.round(np.max(keyframe_values), 2),
                    "Min": np.round(np.min(keyframe_values), 2),
                    "Sum": np.round(np.sum(keyframe_values), 2),
                    "Avg": np.round(np.mean(keyframe_values), 2),
                    "Abs Sum": np.round(np.sum(np.abs(keyframe_values)), 2),
                    "Abs Avg": np.round(np.mean(np.abs(keyframe_values)), 2),
                    "Duration": (num_frames / frame_rate)
                }
                metrics_data.append((schedule_names[i], metrics))

        plt.title('Schedule Visualization')
        plt.xlabel('Frame')
        plt.ylabel('Value')
        plt.legend(loc='upper right')

        metrics_text = "Metric Values:\n"
        for name, data in metrics_data:
            metrics_text += f"{name}: "
            metrics_text += ' | '.join([f"{k}: {v}" for k, v in data.items()])
            metrics_text += "\n"

        plt.figtext(0.5, -0.2 if simulate_stereo else -0.1, metrics_text, ha="center", fontsize=12,
                    bbox={"facecolor": "lightblue", "alpha": 0.5, "pad": 5}, wrap=True)

        filename = str(uuid.uuid4()) + "_keyframe_visualization.png"
        file_path = os.path.join(TEMP, filename)

        plt.savefig(file_path, bbox_inches="tight", pad_inches=1 if simulate_stereo else 0.1)
        plt.close()

        return {
            "ui": {
                "images": [
                    {
                        "filename": filename,
                        "subfolder": "",
                        "type": "temp"
                    }
                ]
            }
        }
    
    @staticmethod
    def gen_hash(input_dict):
        sorted_json = json.dumps(input_dict, sort_keys=True)
        hash_obj = hashlib.sha256()
        hash_obj.update(sorted_json.encode('utf-8'))
        return hash_obj.hexdigest()
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return cls.gen_hash(kwargs)
    

class SaltKeyframeMetrics:
    @classmethod
    def INPUT_TYPES(cls):
        input_types = {
            "required": {
                "schedule_list": ("LIST",),
            },
            "optional": {
                "start_frame": ("INT", {"min": 0, "default": 0}),
                "end_frame": ("INT", {"min": 0, "default": -1}),
                "frame_rate": ("FLOAT", {"min": 0.1, "default": 24.0}),
            }
        }
        return input_types

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("value_min", "value_max", "value_sum", "value_avg", "abs_sum", "abs_avg", "duration")

    FUNCTION = "schedule_metrics"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling/Util"

    def schedule_metrics(self, schedule_list, start_frame=0, end_frame=-1, frame_rate=24.0):
        if end_frame == -1 or end_frame > len(schedule_list):
            end_frame = len(schedule_list)

        num_frames = max(2, end_frame - start_frame)
        keyframe_values = schedule_list[start_frame:end_frame]

        value_min = float(np.min(keyframe_values).round(2))
        value_max = float(np.max(keyframe_values).round(2))
        value_sum = float(np.sum(keyframe_values).round(2))
        value_avg = float(np.mean(keyframe_values).round(2))
        abs_sum = float(np.sum(np.abs(keyframe_values)).round(2))
        abs_avg = float(np.mean(np.abs(keyframe_values)).round(2))
        duration = num_frames / frame_rate

        return value_min, value_max, value_sum, value_avg, abs_sum, abs_avg, duration
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    
class SaltKeyframeScheduler:
    @classmethod
    def INPUT_TYPES(cls):
        easing_funcs = list(easing_functions.keys())
        easing_funcs.insert(0, "None")
        return {
            "required": {
                "keyframe_schedule": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "easing_mode": (easing_funcs, )
            },
            "optional": {
                "end_frame": ("INT", {"min": 0}),
                "ndigits": ("INT", {"min": 1, "max": 12, "default": 5}),
                "a": (WILDCARD, {}),
                "b": (WILDCARD, {})
            }
        }

    RETURN_TYPES = ("LIST", )
    RETURN_NAMES = ("schedule_list", )

    FUNCTION = "keyframe_schedule"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling"

    def keyframe_schedule(self, keyframe_schedule, easing_mode, end_frame=0, ndigits=2, a=None, b=None):
        if a and not isinstance(a, (int, float, bool, list)):
            raise ValueError("`a` is not a valid int, float, boolean, or schedule_list")
        if b and not isinstance(b, (int, float, bool, list)):
            raise ValueError("`b` is not a valid int, float, or boolean, or schedule_list")
        
        custom_vars = {}
        if a:
            custom_vars['a'] = a
        if b:
            custom_vars['b'] = b
        
        scheduler = KeyframeScheduler(end_frame=end_frame, custom_vars=custom_vars)
        schedule = scheduler.generate_schedule(keyframe_schedule, easing_mode=easing_mode, ndigits=ndigits)
        return (schedule, )


class SaltKeyframeSchedulerBFN:
    @classmethod
    def INPUT_TYPES(cls):
        easing_funcs = list(easing_functions.keys())
        easing_funcs.insert(0, "None")
        return {
            "required": {
                "keyframe_schedule": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "easing_mode": (easing_funcs, )
            },
            "optional": {
                "end_frame": ("INT", {"min": 0}),
                "ndigits": ("INT", {"min": 1, "max": 12, "default": 5}),
                "a": (WILDCARD, {}),
                "b": (WILDCARD, {}),
                "c": (WILDCARD, {}),
                "d": (WILDCARD, {}),
                "e": (WILDCARD, {}),
                "f": (WILDCARD, {}),
                "g": (WILDCARD, {}),
                "h": (WILDCARD, {}),
            }
        }

    RETURN_TYPES = ("LIST", )
    RETURN_NAMES = ("schedule_list", )

    FUNCTION = "keyframe_schedule"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling"

    def keyframe_schedule(self, keyframe_schedule, easing_mode, end_frame=0, ndigits=2, a=[0], b=[0], c=[0], d=[0], e=[0], f=[0], g=[0], h=[0]):
        logger.info("Received keyframe_schedule:", keyframe_schedule)
        custom_vars = {"a": a, "b": b, "c": c, "d": d, "e": e, "f": f, "g": g, "h": h}
        scheduler = KeyframeScheduler(end_frame=end_frame, custom_vars=custom_vars)
        schedule = scheduler.generate_schedule(keyframe_schedule, easing_mode=easing_mode, ndigits=ndigits)
        return (schedule, )


NODE_DISPLAY_NAME_MAPPINGS = {
    "SaltOPAC": "Perlin Tremor Scheduler",
    "SaltOPACPerlinSettings": "Perlin Tremor Settings",
    "SaltScheduleConverter": "Convert Schedule List",
    "SaltScheduleVariance": "Apply Easing to Schedule",
    "SaltSchedule2ExecSchedule": "Convert Schedule to Iterative Execution List",
    "SaltLayerScheduler": "Parallax Motion Camera Scheduler",
    "SaltLayerExtractor": "Parallax Motion Camera Scheduler Extractor",
    "SaltParallaxMotion": "Parallax Motion Parameter Schedule Generator",
    "SaltFloatScheduler": "Float Schedule",
    "SaltKSamplerSequence": "KSampler Scheduled Sequence",
    "SaltCLIPTextEncodeSequence": "CLIPTextEncode Scheduled Sequence",
    "SaltConditioningSetMaskAndCombine": "Conditioning Schedule Mask and Combine",
    "SaltThresholdSchedule": "Schedule Value Threshold",
    "SaltListOperation": "Schedule Numeric Operation",
    "SaltListClamp": "Schedule Numeric Clamp",
    "SaltListLinearInterpolation": "Schedule Linear Interpolation",
    "SaltScheduleListExponentialFade": "Schedule Exponential Fade",
    "SaltScheduleRandomValues": "Schedule Random Values",
    "SaltScheduleSmoothing": "Schedule Smoothing",
    "SaltCyclicalSchedule": "Schedule Cyclical Loop",
    "SaltScheduleSplit": "Schedule Split",
    "SaltScheduleMerge": "Schedule Merge",
    "SaltKeyframeVisualizer": "Schedule Visualizer",
    "SaltKeyframeMetrics": "Schedule Metrics",
    "SaltKeyframeScheduler": "Keyframe Scheduler",
    "SaltKeyframeSchedulerBFN": "Keyframe Scheduler (BIG)"
}

NODE_CLASS_MAPPINGS = {
    key: globals()[key] for key in NODE_DISPLAY_NAME_MAPPINGS.keys()
}