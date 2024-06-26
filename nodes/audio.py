import hashlib
import io
import os
import json
import matplotlib.pyplot as plt
import scipy
import subprocess
import tempfile
import time
import uuid
import numpy as np
import torch

from pydub import AudioSegment
from pydub import effects as AudioEffects
from scipy.interpolate import interp1d

import librosa

from diffusers import AudioLDM2Pipeline

import folder_paths
from comfy.utils import ProgressBar

from .. import MENU_NAME, SUB_MENU_NAME, logger
from ..modules.easing import easing_functions
from ..modules.utils import ffmpeg_path

INPUT = folder_paths.get_input_directory()


# AUDIO UTILITIES


def get_buffer(audio):
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    return buffer.getvalue()


def numpy2wav(audio_data, samplerate=16000):
    buffer = io.BytesIO()
    scipy.io.wavfile.write(buffer, samplerate, audio_data)
    wav_bytes = buffer.getvalue()
    return wav_bytes


# AUDIO NODES


class SaltLoadAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {}),
                "start_seconds": ("FLOAT", {"min": 0.0, "default": 0.0}),
                "manual_bpm": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0}),
                "frame_rate": ("INT", {"default": 8, "min": 1, "max": 244}),
            },
            "optional": {
                "duration_seconds": ("FLOAT", {"min": 0.0, "default": 0.0, "optional": True}),
            },
        }

    RETURN_TYPES = ("AUDIO", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("audio", "bpm", "frame_rate", "frame_count")
    FUNCTION = "load_audio"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio"

    def load_audio(self, file_path, start_seconds, duration_seconds=0.0, manual_bpm=0.0, frame_rate=24.0):
        INPUT = folder_paths.get_input_directory()
        file_path = os.path.join(INPUT, file_path)

        # Load the audio segment (by start/duration)
        audio = self.get_audio(file_path, start_seconds, duration_seconds)
        audio_segment = AudioSegment.from_file(io.BytesIO(audio), format="wav")
        audio_segment = audio_segment.set_frame_rate(frame_rate)
        duration_ms = int(duration_seconds * 1000) if duration_seconds else len(audio_segment) - int(start_seconds * 1000)

        bpm = self.analyze_bpm(audio, manual_bpm)
        frame_count = int((duration_ms / 1000.0) * frame_rate)

        return (audio, bpm, frame_rate, frame_count)

    def get_audio(self, file, start_time=0, duration=0):
        TEMP = folder_paths.get_temp_directory()
        os.makedirs(TEMP, exist_ok=True)
        
        temp_file_path = None
        try:
            # Create a temporary file in the specified directory
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir=TEMP) as temp_file:
                temp_file_path = temp_file.name
            
            args = [ffmpeg_path, "-y", "-v", "error", "-i", file, "-acodec", "pcm_s16le", "-ar", "44100"]
            if start_time > 0:
                args += ["-ss", str(start_time)]
            if duration > 0:
                args += ["-t", str(duration)]
            args += [temp_file_path]

            subprocess.run(args, check=True)

            with open(temp_file_path, "rb") as f:
                audio_data = f.read()

            return audio_data
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def analyze_bpm(self, audio_bytes, manual_bpm=0.0):
        with io.BytesIO(audio_bytes) as audio_file:
            y, sr = librosa.load(audio_file, sr=None)

        if manual_bpm <= 0:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            if isinstance(tempo, np.ndarray):
                bpm = float(tempo[0]) if tempo.size > 0 else 0.0
            else:
                bpm = tempo
        else:
            bpm = manual_bpm

        return round(bpm, ndigits=2)

    @staticmethod
    def calculate_file_hash(filename: str):
        try:
            h = hashlib.sha256()
            h.update(filename.encode())
            h.update(str(os.path.getmtime(filename)).encode())
            return h.hexdigest()
        except Exception as e:
            logger.error(e)
            return float("NaN")

    @classmethod
    def IS_CHANGED(cls, file_path, *args, **kwargs):
        INPUT = folder_paths.get_input_directory()
        file_path = os.path.join(INPUT, file_path)
        hash = cls.calculate_file_hash(file_path)
        return hash

    
class SaltSaveAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "audio_sfx"}),
                "format": (["wav", "mp3", "flac"], ),
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_audio"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio"

    def save_audio(self, audio, filename_prefix="audio_sfx", format="wav"):
        OUTPUT = folder_paths.get_output_directory()
        index = 0

        file_extension = format.lower()
        if format not in ['wav', 'mp3', 'flac']:
            logger.error(f"Unsupported format: {format}. Defaulting to WAV.")
            file_extension = "wav"
            format = "wav"

        while True:
            filename = f"{filename_prefix}_%04d.{file_extension}" % index
            full_path = os.path.realpath(os.path.join(OUTPUT, filename))
            if not os.path.exists(full_path):
                break
            index += 1

        audio_segment = AudioSegment.from_file(io.BytesIO(audio), format="wav")
        audio_segment.export(full_path, format=format)

        logger.info(f"Audio saved to {filename} in {format.upper()} format")
        return ()


class SaltAudioFramesyncSchedule:
    @classmethod
    def INPUT_TYPES(cls):
        easing_fns = list(easing_functions.keys())
        easing_fns.insert(0, "None")
        return {
            "required": {
                "audio": ("AUDIO",),
                "amp_control": ("FLOAT", {"min": 0.1, "max": 1024.0, "default": 1.0, "step": 0.01}),
                "amp_offset": ("FLOAT", {"min": 0.0, "max": 1023.0, "default": 0.0, "step": 0.01}),
                "frame_rate": ("INT", {"min": 1, "max": 244, "default": 8}),
                "start_frame": ("INT", {"min": 0, "default": 0}),
                "end_frame": ("INT", {"min": -1}),
                "curves_mode": (easing_fns,)
            }
        }

    RETURN_TYPES = ("LIST", "INT", "INT")
    RETURN_NAMES = ("average_sum", "frame_count", "frame_rate")

    FUNCTION = "schedule"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Scheduling"

    def dbfs_floor_ceiling(self, audio_segment):
        min_dbfs = 0
        max_dbfs = -float('inf')
        for chunk in audio_segment[::1000]:
            if chunk.dBFS > max_dbfs:
                max_dbfs = chunk.dBFS
            if chunk.dBFS < min_dbfs and chunk.dBFS != -float('inf'):
                min_dbfs = chunk.dBFS
        return min_dbfs, max_dbfs

    def dbfs2loudness(self, dbfs, amp_control, amp_offset, dbfs_min, dbfs_max):
        if dbfs == -float('inf'):
            return amp_offset
        else:
            normalized_loudness = (dbfs - dbfs_min) / (dbfs_max - dbfs_min)
            controlled_loudness = normalized_loudness * amp_control
            adjusted_loudness = controlled_loudness + amp_offset
            return max(amp_offset, min(adjusted_loudness, amp_control + amp_offset))

    def interpolate_easing(self, values, easing_function):
        if len(values) < 3 or easing_function == "None":
            return values
        interpolated_values = [values[0]]
        for i in range(1, len(values) - 1):
            prev_val, curr_val, next_val = values[i - 1], values[i], values[i + 1]
            diff_prev = curr_val - prev_val
            diff_next = next_val - curr_val
            direction = 1 if diff_next > diff_prev else -1
            norm_diff = abs(diff_next) / (abs(diff_prev) + abs(diff_next) if abs(diff_prev) + abs(diff_next) != 0 else 1)
            eased_diff = easing_function(norm_diff) * direction
            interpolated_value = curr_val + eased_diff * (abs(diff_next) / 2)
            interpolated_values.append(interpolated_value)
        interpolated_values.append(values[-1])
        return interpolated_values

    def schedule(self, audio, amp_control, amp_offset, frame_rate, start_frame, end_frame, curves_mode):
        audio_segment = AudioSegment.from_file(io.BytesIO(audio), format="wav")
        
        frame_duration_ms = int(1000 / frame_rate)
        start_ms = start_frame * frame_duration_ms
        total_length_ms = len(audio_segment)
        total_frames = total_length_ms // frame_duration_ms
        end_ms = total_length_ms if end_frame <= 0 else min(end_frame * frame_duration_ms, total_length_ms)

        audio_segment = audio_segment[start_ms:end_ms]
        dbfs_min, dbfs_max = self.dbfs_floor_ceiling(audio_segment)

        output = {'average': {'sum': []}}

        max_frames = (end_ms - start_ms) // frame_duration_ms
        for frame_start_ms in range(0, (max_frames * frame_duration_ms), frame_duration_ms):
            frame_end_ms = frame_start_ms + frame_duration_ms
            frame_segment = audio_segment[frame_start_ms:frame_end_ms]

            overall_loudness = self.dbfs2loudness(frame_segment.dBFS, amp_control, amp_offset, dbfs_min, dbfs_max)
            output['average']['sum'].append(overall_loudness)

        if curves_mode != "None":
            output['average']['sum'] = self.interpolate_easing(output['average']['sum'], easing_functions[curves_mode])

        output['average']['sum'] = [round(value, 2) for value in output['average']['sum']]

        return (
            output['average']['sum'],
            max_frames,
            frame_rate
        )
    
    
class SaltAudio2VHS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ), 
            },
        }

    RETURN_TYPES = ("VHS_AUDIO",)
    RETURN_NAMES = ("vhs_audio",)

    FUNCTION = "convert"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Util"

    def convert(self, audio):
        return (lambda : audio,)


class SaltChangeAudioVolume:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "volume_decibals": ("FLOAT", {"min": -60, "max": 60, "default": 0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "change_volume"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Process"

    def change_volume(self, audio_data, volume_decibals):
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="wav")
        modified_audio_segment = audio_segment + volume_decibals
        return (get_buffer(modified_audio_segment), )


class SaltAudioFade:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "fade_type": (["in", "out"],),
                "fade_duration": ("FLOAT", {"min": 0.01}),
            },
            "optional": {
                "fade_start": ("FLOAT", {"default": 0, "min": 0}),
            },
        }

    RETURN_TYPES = ("AUDIO", )
    RETURN_NAMES = ("audio", )
    FUNCTION = "apply_fade"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Process"


    def apply_fade(self, audio, fade_type, fade_duration, fade_start=0):
        audio_segment = AudioSegment.from_file(io.BytesIO(audio), format="wav")

        start_ms = int(fade_start * 1000)
        duration_ms = int(fade_duration * 1000)
        end_fade_ms = start_ms + duration_ms

        before_fade = audio_segment[:start_ms]
        during_fade = audio_segment[start_ms:end_fade_ms]
        after_fade = audio_segment[end_fade_ms:]

        if fade_type == "in":
            faded_part = during_fade.fade_in(duration_ms)
        else:
            faded_part = during_fade.fade_out(duration_ms)

        output = before_fade + faded_part + after_fade

        return (get_buffer(output), )


class SaltAudioFrequencyBoost:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "frequency": ("INT", {"min": 20, "max": 20000, "default": 1000}), 
                "bandwidth": ("FLOAT", {"default": 2.0}),
                "gain_dB": ("FLOAT", {"min": -60, "max": 60, "default": 0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "boost_frequency"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Effect"

    def boost_frequency(self, audio, frequency, bandwidth, gain_dB):
        TEMP = folder_paths.get_temp_directory()
        os.makedirs(TEMP, exist_ok=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=TEMP) as temp_input:
            temp_input.write(audio)
            temp_input_path = temp_input.name

        temp_output_path = tempfile.mktemp(suffix='.wav', dir=TEMP)
        
        eq_filter = f"equalizer=f={frequency}:width_type=o:width={bandwidth}:g={gain_dB}"
        command = [ffmpeg_path, "-y", "-i", temp_input_path, "-af", eq_filter, temp_output_path]

        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with open(temp_output_path, "rb") as temp_output:
                modified_audio_data = temp_output.read()
                
            os.unlink(temp_input_path)
            os.unlink(temp_output_path)
                
            return (modified_audio_data,)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to apply frequency boost with FFmpeg: {e}")
            if os.path.exists(temp_input_path):
                os.unlink(temp_input_path)
            if os.path.exists(temp_output_path):
                os.unlink(temp_output_path)
            return (audio,)
        
        
class SaltAudioFrequencyCutoff:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "filter_type": (["lowpass", "highpass"], ),
                "cutoff_frequency": ("INT", {"min": 20, "max": 20000, "default": 1000}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "apply_cutoff"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Effect"

    def apply_cutoff(self, audio, filter_type, cutoff_frequency):
        TEMP = folder_paths.get_temp_directory()
        os.makedirs(TEMP, exist_ok=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=TEMP) as temp_input:
            temp_input.write(audio)
            temp_input_path = temp_input.name

        temp_output_path = tempfile.mktemp(suffix='.wav', dir=TEMP)
        
        filter_command = f"{filter_type}=f={cutoff_frequency}"
        command = [ffmpeg_path, '-y', "-i", temp_input_path, "-af", filter_command, temp_output_path]

        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            with open(temp_output_path, "rb") as temp_output:
                modified_audio_data = temp_output.read()
                
            os.unlink(temp_input_path)
            os.unlink(temp_output_path)
                
            return (modified_audio_data,)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to apply frequency cutoff with FFmpeg: {e}")
            if os.path.exists(temp_input_path):
                os.unlink(temp_input_path)
            if os.path.exists(temp_output_path):
                os.unlink(temp_output_path)
            return (audio,)
        

class SaltAudioVisualizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "frame_rate": ("INT", {"default": 8, "min": 1, "max": 244}),
            },
            "optional": {
                "start_frame": ("INT", {"min": 0, "default": 0}),
                "end_frame": ("INT", {"min": 0, "default": -1}),
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True

    FUNCTION = "visualize_audio"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Util"

    def visualize_audio(self, audio, frame_rate, start_frame=0, end_frame=-1):
        TEMP = folder_paths.get_temp_directory()
        os.makedirs(TEMP, exist_ok=True)

        audio_segment = AudioSegment.from_file(io.BytesIO(audio), format="wav", dir=TEMP)
        
        frame_duration_ms = 1000 / frame_rate
        start_ms = start_frame * frame_duration_ms
        end_ms = end_frame * frame_duration_ms if end_frame != -1 else len(audio_segment)
        
        relevant_audio_segment = audio_segment[start_ms:end_ms]

        samples = np.array(relevant_audio_segment.get_array_of_samples())
        if relevant_audio_segment.channels == 2:
            samples = samples.reshape((-1, 2))
            samples = samples.sum(axis=1) / 2

        max_val = max(abs(samples.min()), samples.max())
        normalized_samples = samples / max_val

        total_frames = len(normalized_samples) / (audio_segment.frame_rate / frame_rate)
        frame_numbers = np.linspace(start_frame, start_frame + total_frames, num=len(normalized_samples), endpoint=False)

        plt.figure(figsize=(10, 4))
        plt.plot(frame_numbers, normalized_samples, linewidth=0.5)
        plt.title("Audio Visualization")
        plt.ylim(-1, 1) 
        plt.xlabel("Frame")
        plt.ylabel("Amplitude")
        
        filename = str(uuid.uuid4())+"_visualization.png"
        file_path = os.path.join(TEMP, filename)
        plt.savefig(file_path)
        plt.close()

        ui_output = {
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

        return ui_output
    
    @staticmethod
    def gen_hash(input_dict):
        sorted_json = json.dumps(input_dict, sort_keys=True)
        hash_obj = hashlib.sha256()
        hash_obj.update(sorted_json.encode('utf-8'))
        return hash_obj.hexdigest()
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return cls.gen_hash(kwargs)


class SaltAudioStereoSplitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
            },
        }

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("left_channel_mono", "right_channel_mono")
    FUNCTION = "split_stereo_to_mono"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Process"

    def split_stereo_to_mono(self, audio):
        stereo_audio_segment = AudioSegment.from_file(io.BytesIO(audio), format="wav")

        if stereo_audio_segment.channels != 2:
            raise ValueError("Input audio must be two channel stereo.")

        left_channel_mono = stereo_audio_segment.split_to_mono()[0]
        right_channel_mono = stereo_audio_segment.split_to_mono()[1]

        left_audio = get_buffer(left_channel_mono)
        right_audio = get_buffer(right_channel_mono)

        return (left_audio, right_audio)



class SaltAudioMixer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_a": ("AUDIO", {}),
                "audio_b": ("AUDIO", {}),
                "mix_time_seconds": ("FLOAT", {"min": 0.0, "default": 0.0}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("mixed_audio",)
    FUNCTION = "mix_audios"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Process"

    def mix_audios(self, audio_a, audio_b, mix_time_seconds):
        audio_segment_1 = AudioSegment.from_file(io.BytesIO(audio_a), format="wav")
        audio_segment_2 = AudioSegment.from_file(io.BytesIO(audio_b), format="wav")

        position_ms = int(mix_time_seconds * 1000)
        audio_segment = audio_segment_1.overlay(audio_segment_2, position=position_ms)

        return (get_buffer(audio_segment), )


class SaltAudioStitcher:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_a": ("AUDIO", {}),
                "audio_b": ("AUDIO", {}),
            },
            "optional": {
                "silent_transition_seconds": ("FLOAT", {"min": 0.0, "default": 0.0}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("stitched_audio",)
    FUNCTION = "stitch_audios"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Process"

    def stitch_audios(self, audio_a, audio_b, silent_transition_seconds=0.0):
        audio_segment_1 = AudioSegment.from_file(io.BytesIO(audio_a), format="wav")
        audio_segment_2 = AudioSegment.from_file(io.BytesIO(audio_b), format="wav")

        if silent_transition_seconds > 0:
            silence_segment = AudioSegment.silent(duration=int(silent_transition_seconds * 1000))
            stitched = audio_segment_1 + silence_segment + audio_segment_2
        else:
            stitched = audio_segment_1 + audio_segment_2

        return (get_buffer(stitched), )


class SaltAudioLDM2LoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["cvssp/audioldm2", "cvssp/audioldm2-large", "cvssp/audioldm2-music"], ),
            },
            "optional": {
                "device": (["cuda", "cpu"], ),
            },
        }
    
    RETURN_TYPES = ("AUDIOLDM_MODEL", )
    RETURN_NAMES = ("audioldm2_model", )

    FUNCTION = "load_model"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/AudioLDM2"

    def load_model(self, model, device="cuda"):
        models = folder_paths.models_dir
        audioldm2_models = os.path.join(models, "AudioLDM2")
        return (AudioLDM2Pipeline.from_pretrained(model, cache_dir=audioldm2_models, torch_dtype=torch.float16).to(device), )
    

class SaltAudioLDM2Sampler:  
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audioldm2_model": ("AUDIOLDM_MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 200, "min": 1, "max": 1000}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "max": 12.0, "min": 1.0}),
                "audio_length_seconds": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "num_waveforms": ("INT", {"default": 3, "min": 1}),
                "positive_prompt": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "negative_prompt": ("STRING", {"multiline": True, "dynamicPrompts": False})
            },
            "optional": {

            },
        }

    RETURN_TYPES = ("AUDIO", )
    RETURN_NAMES = ("audio", )

    FUNCTION = "sample"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/AudioLDM2"

    def sample(self, audioldm2_model, **kwargs):
        generator = torch.Generator("cuda").manual_seed(kwargs.get("seed", 0))
        steps = kwargs.get("steps", 200)

        def update_comfy_pbar(step, timestep, latents, **kwargs):
            comfy_pbar.update(1)

        comfy_pbar = ProgressBar(steps)

        audio = audioldm2_model(
            kwargs.get("positive_prompt", "The sound of a hammer hitting a wooden surface."),
            negative_prompt=kwargs.get("negative_prompt", "Low quality."),
            num_inference_steps=steps,
            guidance_scale=kwargs.get("guidance_scale", 3.5),
            audio_length_in_s=kwargs.get("audio_length_seconds", 10.0),
            num_waveforms_per_prompt=kwargs.get("num_waveforms", 3),
            generator=generator,
            output_type="np",
            callback=update_comfy_pbar
        ).audios

        wave_bytes = numpy2wav(audio[0])
        return (wave_bytes, )
    

class SaltAudioSimpleReverb:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "reverb_level": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.01}),
                "decay": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "apply_reverb"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Effect"

    def apply_reverb(self, audio, reverb_level, decay):
        original = AudioSegment.from_file(io.BytesIO(audio), format="wav")
        reverb_duration = int(len(original) * reverb_level)
        output = original

        for overlay_delay in range(50, reverb_duration, 50):
            decayed_overlay = original - (decay * overlay_delay)
            output = output.overlay(decayed_overlay, position=overlay_delay)

        return (get_buffer(output),)



class SaltAudioSimpleEcho:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "times": ("INT", {"default": 3, "min": 1, "max": 10}),
                "delay_ms": ("INT", {"default": 100, "min": 100, "max": 2000}),
                "decay_factor": ("FLOAT", {"default": 0.4, "min": 0.1, "max": 0.9, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "apply_echo"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Effect"

    def apply_echo(self, audio, times, delay_ms, decay_factor):
        original = AudioSegment.from_file(io.BytesIO(audio), format="wav")
        silence = AudioSegment.silent(duration=delay_ms)
        echo = original
        fade_duration = int(delay_ms * 0.1)
        speed_increase_factor = 1.01
        speed_increase_step = 0.01

        for i in range(1, times):
            decayed = original - (decay_factor * 10 * i)
            playback_speed = speed_increase_factor + (speed_increase_step * i)
            decayed = decayed.speedup(playback_speed=playback_speed)
            for j in range(1, 5):
                decayed = decayed.overlay(silence + decayed - (5 * j), position=50 * j)
            decayed_with_fades = decayed.fade_in(fade_duration).fade_out(fade_duration)
            echo = echo.overlay(silence + decayed_with_fades, position=delay_ms * i)

        return (get_buffer(echo),)


class SaltAudioNormalize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "normalize_audio"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Process"

    def normalize_audio(self, audio):
        audio_segment = AudioSegment.from_file(io.BytesIO(audio), format="wav")
        normalized_segment = AudioEffects.normalize(audio_segment)
        return (get_buffer(normalized_segment), )
    

class SaltAudioBandpassFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "low_cutoff_frequency": ("INT", {"min": 20, "max": 20000, "default": 300}),
                "high_cutoff_frequency": ("INT", {"min": 20, "max": 20000, "default": 3000}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "apply_bandpass_filter"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Effect"

    def apply_bandpass_filter(self, audio, low_cutoff_frequency, high_cutoff_frequency):
        audio_segment = AudioSegment.from_file(io.BytesIO(audio), format="wav")
        filtered_audio = audio_segment.low_pass_filter(high_cutoff_frequency).high_pass_filter(low_cutoff_frequency)
        return (get_buffer(filtered_audio), )
    

class SaltAudioCompressor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "threshold_dB": ("FLOAT", {"default": -20.0, "min": -60.0, "max": 0.0}),
                "ratio": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 20.0}),
                "attack_ms": ("INT", {"default": 50, "min": 0, "max": 1000}),
                "release_ms": ("INT", {"default": 200, "min": 0, "max": 3000}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "compress_audio"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Process"

    def compress_audio(self, audio, threshold_dB, ratio, attack_ms, release_ms):
        TEMP = folder_paths.get_temp_directory()
        os.makedirs(TEMP, exist_ok=True)

        audio_segment = AudioSegment.from_file(io.BytesIO(audio), format="wav")
        temp_input_path = tempfile.mktemp(suffix='.wav', dir=TEMP)
        temp_output_path = tempfile.mktemp(suffix='.wav', dir=TEMP)

        audio_segment.export(temp_input_path, format="wav")
        points = f"-80/-80|-60/-60|{threshold_dB}/{threshold_dB + (20 / ratio)}|20/20"
        
        command = [
            'ffmpeg', '-y', '-i', temp_input_path,
            '-filter_complex',
            f'compand=attacks={attack_ms / 1000.0}:decays={release_ms / 1000.0}:points={points}',
            temp_output_path
        ]

        subprocess.run(command, check=True)
        
        with open(temp_output_path, 'rb') as f:
            compressed_audio = f.read()

        os.remove(temp_input_path)
        os.remove(temp_output_path)

        return (compressed_audio,)


class SaltAdvancedAudioCompressor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "threshold_dB": ("FLOAT", {"default": -20.0, "min": -60.0, "max": 0.0}),
                "ratio": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0}),
                "attack_ms": ("INT", {"default": 5, "min": 1, "max": 100}),
                "release_ms": ("INT", {"default": 50, "min": 10, "max": 1000}),
                "makeup_gain": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 24.0}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "compress_detailed_audio"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Process"

    def compress_detailed_audio(self, audio, threshold_dB, ratio, attack_ms, release_ms, makeup_gain):
        TEMP = folder_paths.get_temp_directory()
        os.makedirs(TEMP, exist_ok=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=TEMP) as temp_input:
            temp_input.write(audio)
            temp_input_path = temp_input.name

        temp_output_path = tempfile.mktemp(suffix='.wav', dir=TEMP)
        attack_sec = max(attack_ms / 1000.0, 0.01)
        release_sec = max(release_ms / 1000.0, 0.01)
        command = [
            'ffmpeg', '-y', '-i', temp_input_path,
            '-af', f'acompressor=threshold={threshold_dB}dB:ratio={ratio}:attack={attack_sec}:release={release_sec}:makeup={makeup_gain}dB',
            temp_output_path
        ]

        subprocess.run(command, check=True)
        
        with open(temp_output_path, 'rb') as temp_output:
            compressed_audio = temp_output.read()

        os.unlink(temp_input_path)
        os.unlink(temp_output_path)

        return (compressed_audio,)


class SaltAudioDeesser:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "intensity": ("FLOAT", {"default": 0.125, "min": 0.0, "max": 1.0}),
                "amount": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0}),
                "frequency_keep": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "apply_deesser"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Effect"

    def apply_deesser(cls, audio, intensity, amount, frequency_keep):
        TEMP = folder_paths.get_temp_directory()
        os.makedirs(TEMP, exist_ok=True)

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=TEMP) as temp_input, \
                 tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=TEMP) as temp_output:

                temp_input.write(audio)
                temp_input.flush()

                command = [
                    'ffmpeg', '-y', '-i', temp_input.name,
                    '-af', f'deesser=i={intensity}:m={amount}:f={frequency_keep}:s=o',
                    temp_output.name
                ]

                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                temp_output.flush()

            with open(temp_output.name, 'rb') as f:
                processed_audio = f.read()

        finally:
            # Retry mechanism for deletion
            def safe_delete(file_path):
                for attempt in range(3):
                    try:
                        os.unlink(file_path)
                        break
                    except PermissionError:
                        time.sleep(0.1)

            safe_delete(temp_input.name)
            safe_delete(temp_output.name)

        return (processed_audio, )



class SaltAudioNoiseReductionSpectralSubtraction:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "noise_floor": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "reduce_noise_spectral"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Effect"

    def reduce_noise_spectral(cls, audio, noise_floor):
        TEMP = folder_paths.get_temp_directory()
        os.makedirs(TEMP, exist_ok=True)

        temp_input_path = tempfile.mktemp(suffix='.wav', dir=TEMP)
        temp_output_path = tempfile.mktemp(suffix='.wav', dir=TEMP)

        with open(temp_input_path, 'wb') as f:
            f.write(audio)

        command = [
            ffmpeg_path, '-y', '-i', temp_input_path,
            '-af', f'afftdn=nr={noise_floor * 100}',
            temp_output_path
        ]

        subprocess.run(command, check=True)

        with open(temp_output_path, 'rb') as f:
            noise_reduced_audio = f.read()

        os.remove(temp_input_path)
        os.remove(temp_output_path)

        return (noise_reduced_audio,)


class SaltAudioPitchShift:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "semitones": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "shift_pitch"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Effect"

    def shift_pitch(cls, audio, semitones):
        TEMP = tempfile.gettempdir()
        os.makedirs(TEMP, exist_ok=True)

        temp_input_path = tempfile.mktemp(suffix='.wav', dir=TEMP)
        temp_output_path = tempfile.mktemp(suffix='.wav', dir=TEMP)

        with open(temp_input_path, 'wb') as f:
            f.write(audio)

        command = [
            'ffmpeg', '-y', '-i', temp_input_path,
            '-af', f'asetrate=44100*2^({semitones}/12),aresample=44100',
            temp_output_path
        ]

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error during pitch shifting: {e}")
            os.remove(temp_input_path)
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
            raise

        with open(temp_output_path, 'rb') as f:
            pitch_shifted_audio = f.read()

        os.remove(temp_input_path)
        os.remove(temp_output_path)

        return (pitch_shifted_audio,)
    

class SaltAudioPitchShiftScheduled:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "schedule": ("LIST", {"element_type": "FLOAT"}),
            },
            "optional": {
                "interpolate": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "shift_pitch_advanced"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Effect"

    @staticmethod
    def shift_pitch_advanced(audio_bytes, schedule, interpolate=False):
        TEMP = tempfile.gettempdir()
        os.makedirs(TEMP, exist_ok=True)

        temp_input_path = tempfile.mktemp(suffix='.wav', dir=TEMP)
        temp_output_path = tempfile.mktemp(suffix='.wav', dir=TEMP)

        with open(temp_input_path, 'wb') as f:
            f.write(audio_bytes)

        audio = AudioSegment.from_file(temp_input_path)
        frame_rate = audio.frame_rate
        total_frames = len(audio.get_array_of_samples())

        # Schedule processing: interpolate or repeat to match audio length
        if interpolate:
            x = np.linspace(0, total_frames, num=len(schedule), endpoint=True)
            f = interp1d(x, schedule, kind='linear', fill_value="extrapolate")
            pitch_schedule = f(np.arange(total_frames))
        else:
            pitch_schedule = np.tile(schedule, int(np.ceil(total_frames / len(schedule))))[:total_frames]

        # Process audio in chunks and apply pitch shift
        processed_audio = AudioSegment.empty()
        chunk_duration_ms = 100  # Duration of each chunk in milliseconds
        for i in range(0, len(audio), chunk_duration_ms):
            chunk = audio[i:i+chunk_duration_ms]
            semitones = pitch_schedule[int(i / chunk_duration_ms * frame_rate)]
            processed_chunk = chunk._spawn(chunk.raw_data, overrides={
                "frame_rate": int(chunk.frame_rate * 2**(semitones/12.0))
            }).set_frame_rate(frame_rate)
            processed_audio += processed_chunk

        # Export processed audio
        processed_audio.export(temp_output_path, format="wav")

        with open(temp_output_path, 'rb') as f:
            pitch_shifted_audio = f.read()

        os.remove(temp_input_path)
        os.remove(temp_output_path)

        return (pitch_shifted_audio,)


class SaltAudioPitchShiftV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "schedule": ("LIST", {"element_type": "FLOAT"}),
            },
            "optional": {
                "interpolate": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "shift_pitch_advanced"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Effect"

    @staticmethod
    def shift_pitch_advanced(audio_bytes, schedule, interpolate=False):
        TEMP = tempfile.gettempdir()
        os.makedirs(TEMP, exist_ok=True)

        temp_input_path = tempfile.mktemp(suffix='.wav', dir=TEMP)
        temp_output_path = tempfile.mktemp(suffix='.wav', dir=TEMP)

        with open(temp_input_path, 'wb') as f:
            f.write(audio_bytes)

        audio = AudioSegment.from_file(temp_input_path)
        frame_rate = audio.frame_rate
        total_frames = len(audio.get_array_of_samples())

        # Schedule processing: interpolate or repeat to match audio length
        if interpolate:
            x = np.linspace(0, total_frames, num=len(schedule), endpoint=True)
            f = interp1d(x, schedule, kind='linear', fill_value="extrapolate")
            pitch_schedule = f(np.arange(total_frames))
        else:
            pitch_schedule = np.tile(schedule, int(np.ceil(total_frames / len(schedule))))[:total_frames]

        # Process audio in chunks and apply pitch shift
        processed_audio = AudioSegment.empty()
        chunk_duration_ms = 100  # Duration of each chunk in milliseconds
        for i in range(0, len(audio), chunk_duration_ms):
            chunk = audio[i:i+chunk_duration_ms]
            semitones = pitch_schedule[int(i / chunk_duration_ms * frame_rate)]
            processed_chunk = chunk._spawn(chunk.raw_data, overrides={
                "frame_rate": int(chunk.frame_rate * 2**(semitones/12.0))
            }).set_frame_rate(frame_rate)
            processed_audio += processed_chunk

        # Export processed audio
        processed_audio.export(temp_output_path, format="wav")

        with open(temp_output_path, 'rb') as f:
            pitch_shifted_audio = f.read()

        os.remove(temp_input_path)
        os.remove(temp_output_path)

        return (pitch_shifted_audio,)


    

class SaltAudioTrim:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "start_time_seconds": ("FLOAT", {"min": 0.0, "default": 0.0, "step": 0.01}),
                "end_time_seconds": ("FLOAT", {"min": 0.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "trim_audio"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Process"

    def trim_audio(cls, audio, start_time_seconds, end_time_seconds):
        audio_segment = AudioSegment.from_file(io.BytesIO(audio), format="wav")
        trimmed_audio_segment = audio_segment[start_time_seconds * 1000:end_time_seconds * 1000]
        return (get_buffer(trimmed_audio_segment),)


class SaltAudioRepeat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "repeat_times": ("INT", {"min": 1, "default": 2}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "loop_audio"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Process"

    def loop_audio(cls, audio, repeat_times):
        audio_segment = AudioSegment.from_file(io.BytesIO(audio), format="wav")
        looped_audio_segment = audio_segment * repeat_times
        return (get_buffer(looped_audio_segment),)


class SaltAudioPlaybackRate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "speed_factor": ("FLOAT", {"min": 0.5, "max": 12.0, "default": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "adjust_speed"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Process"

    def adjust_speed(cls, audio, speed_factor):
        audio_segment = AudioSegment.from_file(io.BytesIO(audio), format="wav")
        playback_rate = int(audio_segment.frame_rate * speed_factor)
        adjusted_audio_segment = audio_segment.set_frame_rate(playback_rate)
        return (get_buffer(adjusted_audio_segment),)


class SaltAudioInversion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "invert_audio"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Process"

    def invert_audio(cls, audio):
        audio_segment = AudioSegment.from_file(io.BytesIO(audio), format="wav")
        inverted_samples = np.array(audio_segment.get_array_of_samples()) * -1
        inverted_audio_segment = audio_segment._spawn(inverted_samples.tobytes())
        return (get_buffer(inverted_audio_segment),)


class SaltAudioBassBoost:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "cutoff_freq": ("INT", {"default": 150, "min": 20, "max": 300, "step": 1}),
                "boost_dB": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 24.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "boost_bass"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Effect"

    def boost_bass(self, audio, cutoff_freq, boost_dB):
        original = AudioSegment.from_file(io.BytesIO(audio), format="wav")
        
        low_pass_gain = original.low_pass_filter(cutoff_freq).apply_gain(boost_dB)
        if len(low_pass_gain) > len(original):
            low_pass_gain = low_pass_gain[:len(original)]

        boosted = original.overlay(low_pass_gain, position=0)
        
        return (get_buffer(boosted), )


class SaltAudioTrebleBoost:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "cutoff_freq": ("INT", {"default": 150, "min": 20, "max": 300, "step": 1}),
                "boost_dB": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 24.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "treble_bass"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Effect"

    def treble_bass(self, audio, cutoff_freq, boost_dB):
        original = AudioSegment.from_file(io.BytesIO(audio), format="wav")
        
        high_pass_gain = original.high_pass_filter(cutoff_freq).apply_gain(boost_dB)
        if len(high_pass_gain) > len(original):
            high_pass_gain = high_pass_gain[:len(original)]

        boosted = original.overlay(high_pass_gain, position=0)
        
        return (get_buffer(boosted), )


class SaltAudioStereoMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_a": ("AUDIO", {}),
                "audio_b": ("AUDIO", {}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "merge_stereo"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Effect"

    def merge_stereo(self, audio_a, audio_b):
        segment_a = AudioSegment.from_file(io.BytesIO(audio_a), format="wav")
        segment_b = AudioSegment.from_file(io.BytesIO(audio_b), format="wav")
        
        if segment_a.channels > 1:
            segment_a = segment_a.set_channels(1)
        if segment_b.channels > 1:
            segment_b = segment_b.set_channels(1)
        
        min_length = min(len(segment_a), len(segment_b))
        segment_a = segment_a[:min_length]
        segment_b = segment_b[:min_length]
        stereo_audio = AudioSegment.from_mono_audiosegments(segment_a, segment_b)
        
        stereo_audio_bytes = io.BytesIO()
        stereo_audio.export(stereo_audio_bytes, format="wav")
        stereo_audio_bytes.seek(0)
        
        return stereo_audio_bytes.read(),



NODE_DISPLAY_NAME_MAPPINGS = {
    "SaltLoadAudio": "Load Audio",
    "SaltSaveAudio": "Save Audio",

    "SaltAudioFramesyncSchedule": "Schedule Audio Framesync",
    "SaltAudio2VHS": "Audio to VHS Audio",
    "SaltChangeAudioVolume": "Audio Volume",
    "SaltAudioFade": "Audio Fade",
    "SaltAudioFrequencyBoost": "Audio Frequency Boost",
    "SaltAudioFrequencyCutoff": "Audio Frequency Cutoff",
    "SaltAudioVisualizer": "Audio Visualizer",
    "SaltAudioStereoSplitter": "Audio Stereo Splitter",
    "SaltAudioMixer": "Audio Mixer",
    "SaltAudioStitcher": "Audio Stitcher",
    "SaltAudioSimpleReverb": "Audio Simple Reverb",
    "SaltAudioSimpleEcho": "Audio Simple Echo",
    "SaltAudioNormalize": "Audio Normalize",
    "SaltAudioBandpassFilter": "Audio Bandpass Filter",
    "SaltAudioCompressor": "Audio Compressor",
    "SaltAdvancedAudioCompressor": "Audio Compressor Advanced",
    "SaltAudioDeesser": "Audio De-esser",
    "SaltAudioNoiseReductionSpectralSubtraction": "Audio Noise Reduction (Spectral Subtraction)",
    "SaltAudioPitchShift": "Audio Pitch Shift",
    "SaltAudioPitchShiftScheduled": "Audio Scheduled Pitch Shift",
    "SaltAudioTrim": "Audio Trim",
    "SaltAudioRepeat": "Audio Repeat",
    "SaltAudioPlaybackRate": "Audio Playback Rate",
    "SaltAudioInversion": "Audio Reverse",
    "SaltAudioBassBoost": "Audio Bass Boost",
    "SaltAudioTrebleBoost": "Audio Treble Boost",
    "SaltAudioStereoMerge": "Audio Stereo Merge",

    "SaltAudioLDM2LoadModel": "AudioLDM2 Model Loader",
    "SaltAudioLDM2Sampler": "AudioLDM2 Sampler",
}

NODE_CLASS_MAPPINGS = {
    key: globals()[key] for key in NODE_DISPLAY_NAME_MAPPINGS.keys()
}