import math
import random
import traceback
import cv2
from PIL import Image, ImageChops, ImageDraw, ImageFilter
import numpy as np
import torch

from .easing import easing_functions

# Simple perlin noise based on Power Noise Suite
class PerlinNoise:
    def __init__(self, seed=None):
        self.seed = seed if seed is not None else random.randint(0, 10000)
        torch.manual_seed(self.seed)
        self.p = torch.randperm(256, dtype=torch.int32)
        self.p = torch.cat((self.p, self.p))

    def fade(self, t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    def lerp(self, t, a, b):
        return a + t * (b - a)

    def grad(self, hash, x):
        h = hash & 15
        grad = 1 + (h & 7)
        if h & 8:
            grad = -grad
        return grad * x

    def noise(self, x):
        x = x % 255
        X = torch.tensor(x, dtype=torch.float32).floor().long()
        x -= X.float()
        u = self.fade(x)

        A = self.p[X % 256]
        B = self.p[(X + 1) % 256]

        return self.lerp(u, self.grad(self.p[A], x), self.grad(self.p[B], x - 1))

    def sample(self, x, scale=1.0, octaves=1, persistence=0.5, lacunarity=2.0):
        total = 0.0
        frequency = 1.0
        amplitude = 1.0
        max_value = 0.0
        for _ in range(octaves):
            total += self.noise(x * scale * frequency) * amplitude
            max_value += amplitude
            amplitude *= persistence
            frequency *= lacunarity

        return (total / max_value).item()

# Helper function
def apply_noise(perlin_noise, frame_number, scale, octaves, persistence, lacunarity):
    return perlin_noise.sample(frame_number, scale=scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity)

# Movement Presets
def movement_mode_curves(perlin_noise, movement_tremor_scale, frame_number, tremor_octaves, tremor_persistence, tremor_lacunarity):
    return {
        'orbit': lambda angle_increment, amplitude: (
            amplitude * math.cos(math.radians(frame_number * angle_increment)) + apply_noise(perlin_noise, frame_number, movement_tremor_scale, tremor_octaves, tremor_persistence, tremor_lacunarity) * amplitude,
            amplitude * math.sin(math.radians(frame_number * angle_increment)) + apply_noise(perlin_noise, frame_number, movement_tremor_scale, tremor_octaves, tremor_persistence, tremor_lacunarity) * amplitude
        ),
        'side-to-side': lambda angle_increment, amplitude: (
            amplitude * math.sin(math.radians(frame_number * angle_increment)) + apply_noise(perlin_noise, frame_number, movement_tremor_scale, tremor_octaves, tremor_persistence, tremor_lacunarity) * amplitude,
            0
        ),
        'up-and-down': lambda angle_increment, amplitude: (
            0,
            amplitude * math.sin(math.radians(frame_number * angle_increment)) + apply_noise(perlin_noise, frame_number, movement_tremor_scale, tremor_octaves, tremor_persistence, tremor_lacunarity) * amplitude
        ),
        'diagonal_bottom_left': lambda angle_increment, amplitude: (
            amplitude * math.sin(math.radians(frame_number * angle_increment)) + apply_noise(perlin_noise, frame_number, movement_tremor_scale, tremor_octaves, tremor_persistence, tremor_lacunarity) * amplitude,
            amplitude * math.sin(math.radians(frame_number * angle_increment)) + apply_noise(perlin_noise, frame_number, movement_tremor_scale, tremor_octaves, tremor_persistence, tremor_lacunarity) * amplitude
        ),
        'diagonal_top_right': lambda angle_increment, amplitude: (
            -amplitude * math.sin(math.radians(frame_number * angle_increment)) + apply_noise(perlin_noise, frame_number, movement_tremor_scale, tremor_octaves, tremor_persistence, tremor_lacunarity) * amplitude,
            -amplitude * math.sin(math.radians(frame_number * angle_increment)) + apply_noise(perlin_noise, frame_number, movement_tremor_scale, tremor_octaves, tremor_persistence, tremor_lacunarity) * amplitude
        )
    }

# Image Transform movement modes
movement_modes = ["orbit", "side-to-side", "up-and-down", "diagonal_bottom_left", "diagonal_top_right"]

# Edge Handling Modes
edge_modes = {
    'clamp': lambda idx, max_val: np.clip(idx, 0, max_val - 1),
    'mirror': lambda idx, max_val: np.abs(2*(idx // max_val) * (idx % max_val) + idx % max_val),
    'wrap': lambda idx, max_val: np.mod(idx, max_val),
    'smear': lambda idx, max_val: np.where(idx < 0, 0, np.where(idx >= max_val, max_val - 1, idx))
}

# Edge Detection
def edge_fx(indices, max_value, mode, mask=None, repetition=False):
    original_indices = np.copy(indices)
    
    if mode in edge_modes:
        indices = edge_modes[mode](indices, max_value)

    if repetition:
        indices = np.mod(indices, max_value)
        
    if mask is not None:
        mask[np.where(indices != original_indices)] = 255

    return indices

def seam_fx(texture, mask, edge_mode):
    rows, cols, _ = texture.shape
    new_texture = np.copy(texture)
    
    for row in range(rows):
        for col in range(cols):
            if mask[row, col] == 128:
                neighbor_indices = [(i, j) for i in range(row-1, row+2) for j in range(col-1, col+2)
                                    if 0 <= i < rows and 0 <= j < cols]
                new_texture[row, col] = np.mean([texture[i, j] for i, j in neighbor_indices], axis=0).astype(np.uint8)
                
    return edge_fx(new_texture, 255, edge_mode)

def generate_frame(
        num_frames, 
        frame_number,
        texture_image, 
        displacement_image, 
        amplitude, 
        angle_increment=3, 
        displacement_start_frame=None,
        displacement_end_frame=None,
        movement_mode=None, 
        movement_tremor_scale=0.01,
        easing_function=None,
        edge_mode="clamp", 
        repetition=False,
        create_mask=False,
        zoom_factor=1,
        zoom_increment=0.0,
        zoom_easing="ease-in-out",
        zoom_coordinates=(-1, -1),
        zoom_start_frame=None,
        zoom_end_frame=None,
        zoom_tremor_scale=0.01,
        tremor_octaves=1,
        tremor_persistence=0.5,
        tremor_lacunarity=2,
        seed=1492
    ):
    perlin_noise = PerlinNoise(seed=seed)

    modes = movement_mode_curves(perlin_noise, movement_tremor_scale, frame_number, tremor_octaves, tremor_persistence, tremor_lacunarity)
    current_mode = modes[movement_mode]

    if zoom_factor <= 0:
        raise ValueError("Zoom factor must be greater than 0.")

    if zoom_start_frame is None:
        zoom_start_frame = 0
    elif zoom_start_frame >= zoom_end_frame:
        zoom_start_frame = 0

    if zoom_end_frame is None:
        zoom_end_frame = num_frames
    elif zoom_end_frame <= zoom_start_frame:
        zoom_end_frame = num_frames

    if displacement_start_frame is None:
        displacement_start_frame = 0
    elif displacement_start_frame >= displacement_end_frame:
        displacement_start_frame = 0

    if displacement_end_frame is None:
        displacement_end_frame = num_frames
    elif displacement_end_frame <= displacement_start_frame:
        displacement_end_frame = num_frames
    
    if zoom_increment != 0:
        if zoom_start_frame <= frame_number <= zoom_end_frame:
            relative_frame = frame_number - zoom_start_frame
            total_zoom_frames = zoom_end_frame - zoom_start_frame
            zoom_progress = relative_frame / total_zoom_frames

            if zoom_easing != 'None':
                dynamic_zoom_factor = 1 + easing_functions[zoom_easing](zoom_progress) * zoom_increment
            else:
                dynamic_zoom_factor = 1 + zoom_progress * zoom_increment

            dynamic_zoom_factor *= 1 + perlin_noise.sample(frame_number, scale=zoom_tremor_scale, octaves=tremor_octaves, persistence=tremor_persistence, lacunarity=tremor_lacunarity) * 0.01
        else:
            dynamic_zoom_factor = 1
    else:
        dynamic_zoom_factor = zoom_factor

    if dynamic_zoom_factor != 1:
        texture_image = zoom_and_crop(np.array(texture_image)[:, :, ::-1], dynamic_zoom_factor, zoom_coordinates)
        displacement_image = zoom_and_crop(np.array(displacement_image.convert("L")), dynamic_zoom_factor, zoom_coordinates)
    else:
        texture_image = np.array(texture_image)[:, :, ::-1]
        displacement_image = np.array(displacement_image.convert("L"))

    try:
        texture_rows, texture_cols, _ = texture_image.shape
        if displacement_image.shape[:2] != (texture_rows, texture_cols):
            displacement_image = cv2.resize(displacement_image, (texture_cols, texture_rows))

        if displacement_start_frame <= frame_number <= displacement_end_frame:
            if movement_mode:
                x_displacement, y_displacement = current_mode(angle_increment, amplitude)
                if easing_function:
                    relative_frame = frame_number - displacement_start_frame
                    total_displacement_frames = displacement_end_frame - displacement_start_frame
                    ease_value = easing_functions[easing_function](relative_frame / total_displacement_frames)
                    x_displacement *= ease_value
                    y_displacement *= ease_value
            else:
                angle = frame_number * angle_increment
                x_displacement = amplitude * math.cos(math.radians(angle))
                y_displacement = amplitude * math.sin(math.radians(angle))
        else:
            x_displacement, y_displacement = 0, 0

        texture_rows, texture_cols, _ = texture_image.shape
        y_idx, x_idx = np.indices((texture_rows, texture_cols))

        occupancy = np.zeros((texture_rows, texture_cols), dtype=np.uint8)
        edge_mask = np.zeros((texture_rows, texture_cols), dtype=np.uint8)

        x_idx = x_idx.astype(np.float32) + (x_displacement * displacement_image / 255.0).astype(np.float32)
        y_idx = y_idx.astype(np.float32) + (y_displacement * displacement_image / 255.0).astype(np.float32)

        x_idx = edge_fx(x_idx, texture_cols, edge_mode, mask=edge_mask, repetition=repetition)
        y_idx = edge_fx(y_idx, texture_rows, edge_mode, mask=edge_mask, repetition=repetition)

        new_y_idx = np.clip(y_idx.astype(int), 0, texture_rows - 1)
        new_x_idx = np.clip(x_idx.astype(int), 0, texture_cols - 1)

        occupancy[new_y_idx, new_x_idx] = 1

        #left_behind_mask = np.where(occupancy == 0, 128, 0).astype(np.uint8)
        #combined_mask = cv2.bitwise_or(edge_mask, left_behind_mask)

        transformed_texture = texture_image[new_y_idx, new_x_idx]

        otexture_zoomed = texture_image
        if dynamic_zoom_factor != 1:
            transformed_texture = zoom_and_crop(transformed_texture, dynamic_zoom_factor, zoom_coordinates)
            otexture_zoomed = zoom_and_crop(texture_image, dynamic_zoom_factor, zoom_coordinates)

        inverse_displacement_mask = np.where(displacement_image == 0, 1, 0).astype(np.uint8)

        original_non_focal = cv2.bitwise_and(otexture_zoomed, otexture_zoomed, mask=inverse_displacement_mask)
        transformed_non_focal = cv2.bitwise_and(transformed_texture, transformed_texture, mask=inverse_displacement_mask)

        original_non_focal_gray = cv2.cvtColor(original_non_focal, cv2.COLOR_BGR2GRAY)
        transformed_non_focal_gray = cv2.cvtColor(transformed_non_focal, cv2.COLOR_BGR2GRAY)

        diff_mask = cv2.absdiff(original_non_focal_gray, transformed_non_focal_gray)

        diff_mask_blurred = cv2.GaussianBlur(diff_mask, (0, 0), 3.0)
        diff_mask_clamped = np.interp(diff_mask_blurred, [30, 35], [0, 255]).astype(np.uint8)

        transformed_texture_pil = Image.fromarray(cv2.cvtColor(transformed_texture, cv2.COLOR_BGR2RGB))
        mask_image_pil = None
        if create_mask:
            mask_image_pil = Image.fromarray(diff_mask_clamped)

        return transformed_texture_pil, mask_image_pil
            
    except Exception as e:
        traceback.print_exc()
        return None, None

def zoom_and_crop(image, zoom_factor, coordinates=(-1, -1)):
    if zoom_factor == 1:
        return image

    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape

    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    x, y = coordinates

    if x == -1:
        x = width // 2
    if y == -1:
        y = height // 2

    x = int(x * zoom_factor)
    y = int(y * zoom_factor)

    y1 = max(y - height // 2, 0)
    y2 = min(y1 + height, new_height)
    x1 = max(x - width // 2, 0)
    x2 = min(x1 + width, new_width)

    if y2 == new_height:
        y1 = new_height - height
    if x2 == new_width:
        x1 = new_width - width

    cropped_image = resized_image[y1:y2, x1:x2]
    
    return cropped_image

class OrganicPerlinCameraScheduler:
    def __init__(
                self, 
                frame_count, 
                zoom_speed=0.1,
                pan_speed=0.1, 
                pan_directions=[90], 
                direction_change_frames=[0], 
                tremor_params={'scale': 0.1, 'octaves': 1, 'persistence': 0.5, 'lacunarity': 2.0},
                direction_curve='linear', 
                start_x=0, 
                start_y=0
            ):
        self.frame_count = frame_count
        self.zoom_speed = zoom_speed
        self.pan_speed = pan_speed
        self.pan_directions = [np.radians(direction) for direction in pan_directions]
        self.direction_change_frames = direction_change_frames
        self.tremor_params = tremor_params
        self.perlin_noise = PerlinNoise()
        self.global_tremor = [self.perlin_noise.sample(x, **self.tremor_params) for x in range(frame_count)]
        self.direction_curve = direction_curve
        self.start_x = start_x
        self.start_y = start_y 

    def interpolate_direction(self, current_frame):
        is_moving = False
        direction = self.pan_directions[0]
        for i, change_frame in enumerate(self.direction_change_frames):
            if current_frame < change_frame:
                direction = self.pan_directions[0]
                break
            elif i == len(self.direction_change_frames) - 1 or current_frame < self.direction_change_frames[i + 1]:
                is_moving = True
                start_direction = self.pan_directions[i]
                end_direction = self.pan_directions[i + 1] if i + 1 < len(self.pan_directions) else start_direction
                segment_start = change_frame
                segment_end = self.direction_change_frames[i + 1] if i + 1 < len(self.direction_change_frames) else self.frame_count
                progress = (current_frame - segment_start) / (segment_end - segment_start)
                direction = self.apply_curve(start_direction, end_direction, progress)
                break

        return direction, is_moving

    def apply_curve(self, start, end, progress):

        """
        easing_functions = {
            'linear': lambda t: t,
            'ease-in': lambda t: t ** 2,
            'ease-out': lambda t: t * (2 - t),
            'ease-in-out': lambda t: 3 * t ** 2 - 2 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2,
            'bounce-in': bounce_in,
            'bounce-out': bounce_out,
            'bounce-in-out': bounce_in_out,
            'sinusoidal-in': sinusoidal_in,
            'sinusoidal-out': sinusoidal_out,
            'sinusoidal-in-out': sinusoidal_in_out,
        }
        """

        if self.direction_curve in easing_functions:
            progress = easing_functions[self.direction_curve](progress)

        return start + (end - start) * progress

    def get_pan_offset(self, frame, direction, speed_factor=1.0, is_moving=True):
        if not is_moving:
            return 0, 0
        dx = np.cos(direction) * self.pan_speed * speed_factor * frame
        dy = np.sin(direction) * self.pan_speed * speed_factor * frame
        return dx, dy

    def get_zoom(self, mode, frame, speed_factor=1.0):
        effective_speed = self.zoom_speed * speed_factor
        if mode == 'zoom-in':
            return 1 + effective_speed * frame
        elif mode == 'zoom-out':
            return max(1 - effective_speed * frame, 0.1)
        elif mode == 'zoom-in-out':
            mid_point = self.frame_count / 2
            if frame <= mid_point:
                return 1 + effective_speed * frame
            else:
                return max(1 + effective_speed * (self.frame_count - frame), 0.1)
        return 1

    def apply_tremor(self, value, frame, is_global=False):
        tremor_scale = 0.01 if is_global else 0.1
        return value + self.global_tremor[frame] * tremor_scale

    def animate(self, mode, layer_offsets="1"):
        layer_offsets = [float(offset) for offset in layer_offsets.split(',')]
        layers_data = []

        for offset in layer_offsets:
            layer_data = []
            for frame in range(self.frame_count):
                current_direction, is_moving = self.interpolate_direction(frame)
                zoom = self.get_zoom(mode, frame, speed_factor=offset)
                dx, dy = self.get_pan_offset(frame, current_direction, speed_factor=offset, is_moving=is_moving)

                zoom = self.apply_tremor(zoom, frame, is_global=True)
                dx = self.apply_tremor(dx, frame, is_global=True)
                dy = self.apply_tremor(dy, frame, is_global=True)

                layer_data.append((zoom, dx, dy))
            layers_data.append(layer_data)

        return layers_data

# Zoom Presets
zoom_presets = {
    "Zoom In": (1, 1.3),
    "Zoom Out": (1, 0.7),
    "Zoom In/Out": (1.3, 0.7),
    "Zoom Out/In": (0.7, 1.3),
}

# Horizontal Pan Presets
horizontal_pan_presets = {
    "Pan Left → Right": (-64, 64),
    "Pan Right → Left": (64, -64),
    "Pan Left → Center": (-64, 0),
    "Pan Right → Center": (64, 0),
    "Pan Center → Right": (0, 64),
    "Pan Center → Left": (0, -64),
}


# Vertical Pan Presets
vertical_pan_presets = {
    "Pan Up → Down": (64, -64),
    "Pan Down → Up": (-64, 64),
    "Pan Up → Center": (64, 0),
    "Pan Down → Center": (-64, 0),
    "Pan Center → Up": (0, 64),
    "Pan Center → Down": (0, -64),
}

class ImageRotationTransition:
    def __init__(self, img_list_a, img_list_b, max_frames, mode='right'):
        self.img_list_a = img_list_a
        self.img_list_b = img_list_b
        self.max_frames = max_frames
        self.mode = mode
        self.zoom_start_frame = max_frames // 10
        self.zoom_end_frame = max_frames - max_frames // 10
        self.zoomed_in = False

    def process_frame(self, current_frame, max_scale):
        total_rotation_angle = 360
        angle = (total_rotation_angle * current_frame / self.max_frames) % total_rotation_angle
        angle = angle if self.mode == 'right' else -angle
        scale_factor = self.ease_scale(current_frame, max_scale)
        img_a = self.img_list_a[current_frame]
        img_b = self.img_list_b[current_frame]

        img_a_rotated = self.rotate_image(img_a, angle, scale_factor)
        img_b_rotated = self.rotate_image(img_b, angle, scale_factor)
        blend_factor = current_frame / self.max_frames
        img_rotated = Image.blend(img_a_rotated, img_b_rotated, blend_factor)
        return img_rotated

    def ease_scale(self, current_frame, max_scale):
        half_frames = self.max_frames / 2
        progress = current_frame / half_frames

        if current_frame <= self.zoom_start_frame:
            scale = 1 + (self.calculate_max_scale(self.img_list_a[0], max_scale) - 1) * math.sin((current_frame / self.zoom_start_frame) * (math.pi / 2))
        elif current_frame >= self.zoom_end_frame:
            scale = 1 + (self.calculate_max_scale(self.img_list_a[0], max_scale) - 1) * math.sin(((self.max_frames - current_frame) / (half_frames / 2)) * (math.pi / 2))
        else:
            scale = self.calculate_max_scale(self.img_list_a[0], max_scale)

        return scale

    def calculate_max_scale(self, img, max_scale):
        original_width, original_height = img.size
        diagonal_length = math.sqrt(original_width ** 2 + original_height ** 2)
        max_dimension = max(original_width, original_height)
        return diagonal_length / max_dimension * max_scale

    def rotate_image(self, img, angle, scale_factor):
        original_width, original_height = img.size
        scaled_width = int(original_width * scale_factor)
        scaled_height = int(original_height * scale_factor)
        img_scaled = img.resize((scaled_width, scaled_height), Image.LANCZOS)

        diagonal_length = math.sqrt(scaled_width ** 2 + scaled_height ** 2)
        canvas_size = int(diagonal_length)
        canvas = Image.new('RGBA', (canvas_size, canvas_size), (0, 0, 0, 0))

        paste_x = (canvas_size - scaled_width) // 2
        paste_y = (canvas_size - scaled_height) // 2
        canvas.paste(img_scaled, (paste_x, paste_y))

        canvas_rotated = canvas.rotate(angle, resample=Image.BILINEAR, expand=True)

        new_width, new_height = canvas_rotated.size
        new_center_x, new_center_y = new_width // 2, new_height // 2
        box = (
            new_center_x - original_width // 2,
            new_center_y - original_height // 2,
            new_center_x + original_width // 2,
            new_center_y + original_height // 2
        )
        img_rotated = canvas_rotated.crop(box)
        return img_rotated

class ImageRotationTransformer:
    def __init__(self, max_frames, mode='right'):
        self.max_frames = max_frames
        self.mode = mode

    def rotate_image(self, img, current_frame):
        angle = (360 * current_frame / self.max_frames) % 360
        angle = angle if self.mode == 'right' else -angle
        return img.rotate(angle, resample=Image.BILINEAR, expand=True)

class ImageBatchTransition:
    def __init__(self, list_a, list_b, frames_per_transition, blur_radius=[0], mode='swipe_right', max_zoom_scale=[2]):
        self.list_a = list_a
        self.list_b = list_b
        self.frames = frames_per_transition
        self.blur_radius = blur_radius
        self.mode = mode
        self.max_zoom_scale = max_zoom_scale

    def create_transition(self):
        transition_images = self.list_a[:-self.frames]

        for frame in range(self.frames):
            if "rotate" in self.mode:
                zoom_idx = (frame if frame < len(self.max_zoom_scale) else -1)
                rotate_transition = ImageRotationTransition(self.list_a[-self.frames:], self.list_b[:self.frames], self.frames, self.mode)
                frame_img = rotate_transition.process_frame(frame, self.max_zoom_scale[zoom_idx])
                transition_images.append(frame_img)
                continue

            new_img = Image.new('RGB', (self.list_a[0].width, self.list_a[0].height))
            factor = (frame + 1) / self.frames
            max_dimension = max(new_img.width, new_img.height)
            diagonal = math.sqrt(new_img.width**2 + new_img.height**2)
            radius = int(diagonal * factor)

            src_img_index_a = len(self.list_a) - self.frames + frame
            src_img_index_b = frame

            img_a = self.list_a[src_img_index_a]
            img_b = self.list_b[src_img_index_b]

            mask = Image.new("L", (new_img.width, new_img.height), 0)
            draw = ImageDraw.Draw(mask)

            slide_width = int(factor * new_img.width)
            slide_height = int(factor * new_img.height)

            if "diagonal" in self.mode:
                diagonal_length = math.sqrt(new_img.width ** 2.2 + new_img.height ** 2.2)
                for y in range(new_img.height):
                    for x in range(new_img.width):
                        if self.mode == 'diagonal_tl_br':
                            if (x + y) / diagonal_length >= factor:
                                mask.putpixel((x, y), 255)
                        elif self.mode == 'diagonal_tr_bl':
                            if (new_img.width - x + y) / diagonal_length >= factor:
                                mask.putpixel((x, y), 255)
                        elif self.mode == 'diagonal_bl_tr':
                            if (x + new_img.width - y) / diagonal_length >= factor:
                                mask.putpixel((x, y), 255)
                        elif self.mode == 'diagonal_br_tl':
                            if (new_img.width - x + new_img.height - y) / diagonal_length >= factor:
                                mask.putpixel((x, y), 255)
            elif self.mode == 'circle_expand':
                draw.ellipse((new_img.width//2 - radius, new_img.height//2 - radius, new_img.width//2 + radius, new_img.height//2 + radius), fill=255)
            elif self.mode == 'circle_contract':
                current_radius = int(diagonal * (1 - factor) / 2)
                draw.ellipse((new_img.width//2 - current_radius, new_img.height//2 - current_radius, new_img.width//2 + current_radius, new_img.height//2 + current_radius), fill=255)
                mask = ImageChops.invert(mask)
                mask = ImageChops.invert(mask) 
            elif self.mode == 'swipe_right':
                mask.paste(255, (0, 0, slide_width, new_img.height))
            elif self.mode == 'swipe_left':
                mask.paste(255, (new_img.width - slide_width, 0, new_img.width, new_img.height))
            elif self.mode == 'swipe_up':
                mask.paste(255, (0, new_img.height - slide_height, new_img.width, new_img.height))
            elif self.mode == 'swipe_down':
                mask.paste(255, (0, 0, new_img.width, slide_height))

            blur_idx = (frame if frame < len(self.blur_radius) else -1)
            if self.blur_radius[blur_idx] > 0:
                mask = mask.filter(ImageFilter.GaussianBlur(self.blur_radius[blur_idx]))

            if "swipe" in self.mode or "circle" in self.mode:
                if "contract" in self.mode:
                    new_img.paste(img_b, mask=ImageChops.invert(mask))
                    new_img.paste(img_a, mask=mask)
                else:
                    new_img.paste(img_b, mask=mask)
                    new_img.paste(img_a, mask=ImageChops.invert(mask))
            else:
                new_img.paste(img_a, mask=mask)
                new_img.paste(img_b, mask=ImageChops.invert(mask))

            transition_images.append(new_img)

        transition_images.extend(self.list_b[self.frames:])
        return transition_images
