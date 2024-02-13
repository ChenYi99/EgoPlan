"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import copy

import torch
from src.video_llama.video_llama.common.registry import registry
import numpy as np
from src.video_llama.video_llama.processors import transforms_video
from src.video_llama.video_llama.processors.base_processor import BaseProcessor
from src.video_llama.video_llama.processors.randaugment import VideoRandomAugment
from src.video_llama.video_llama.processors import functional_video as F
from omegaconf import OmegaConf
from torchvision import transforms
import random as rnd
import cv2
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import os


MAX_INT = registry.get("MAX_INT")
N_ACTIONS = 4

def load_video(sample, n_frms=MAX_INT, n_actions=N_ACTIONS, sampling="uniform"):
    # frame_idx starts with 1 in epic-kitchens dataset

    task_progress_metadata = sample["task_progress_metadata"]
    current_observation_frame_idx = sample["current_observation_frame"]
    video_rgb_frame_dir = sample["video_rgb_frame_dir"]

    most_recent_actions_metadata = list(
        filter(lambda item: item['stop_frame'] <= current_observation_frame_idx,
               task_progress_metadata))[-n_actions:]

    clips = []
    for i, action_metadata in enumerate(most_recent_actions_metadata):
        start_frame_idx, stop_frame_idx = action_metadata["start_frame"], action_metadata["stop_frame"]
        if i+1 < len(most_recent_actions_metadata):
            next_start_frame_idx = most_recent_actions_metadata[i+1]["start_frame"]
            stop_frame_idx = min(stop_frame_idx, next_start_frame_idx)
        else:
            stop_frame_idx = min(stop_frame_idx, current_observation_frame_idx)

        vlen = stop_frame_idx - start_frame_idx + 1

        try:
            assert vlen >= n_frms
        except Exception as e:
            # print(f"Failed to extract key frames: vlen(={vlen}) < n_frms !!!\n"
            #       f"sample_id: {sample['sample_id']}, video_id: {sample['video_id']}\n"
            #       f"start_frame_idx: {start_frame_idx}, stop_frame_idx: {stop_frame_idx}\n"
            #       f"current_observation_frame_idx: {current_observation_frame_idx}\n"
            #       f"action_metadata: {action_metadata}\n"
            #       f"most_recent_actions_metadata: {most_recent_actions_metadata}")
            raise e

        start, end = 0, vlen
        if sampling == "uniform":
            indices = np.arange(start, end, vlen / n_frms).astype(int).tolist()
        elif sampling == "headtail":
            indices_h = sorted(rnd.sample(range(vlen // 2), n_frms // 2))
            indices_t = sorted(rnd.sample(range(vlen // 2, vlen), n_frms // 2))
            indices = indices_h + indices_t
        else:
            raise NotImplementedError

        images_group = []
        for offset in indices:
            frame_idx = start_frame_idx + offset
            frame_path = os.path.join(video_rgb_frame_dir, f"frame_{str(frame_idx).zfill(10)}.jpg")
            if os.path.exists(frame_path):
                frame = Image.open(frame_path).convert('RGB')
                images_group.append(frame)
            else:
                print(f"image_path doesn't exist!! {frame_path}")
                raise FileNotFoundError

        if len(images_group) < n_frms:
            images_group = images_group + [images_group[-1]]*(n_frms-len(images_group))
        images_group = images_group[:n_frms]
        clip = np.stack(images_group)  # T, H, W, C
        clip = torch.tensor(clip).float().permute(3, 0, 1, 2)  # C, T, H, W
        clips.append(clip)

    image_path = os.path.join(video_rgb_frame_dir, f"frame_{str(current_observation_frame_idx).zfill(10)}.jpg")
    image = Image.open(image_path).convert('RGB')  # H, W, C

    if len(clips) > 0:
        clip_mask = [1] * len(clips) + [0] * (n_actions-len(clips))
        clip_mask = torch.tensor(clip_mask)
        clips = clips + [clips[-1].clone()] * (n_actions-len(clips))
    else:
        padding_clip = torch.stack([torch.tensor(np.array(image).copy())] * n_frms).float() # T, H, W, C
        padding_clip = padding_clip.permute(3, 0, 1, 2)  # C, T, H, W
        clips = [padding_clip] * n_actions # N, C, T, H, W
        clip_mask = [0] * n_actions
        clip_mask = torch.tensor(clip_mask)

    clip_mask = clip_mask.bool()

    return image, clips, clip_mask

class EgoplanVideoBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None, n_frms=MAX_INT, n_actions=N_ACTIONS, sample_strategy="headtail"):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize_video = transforms_video.NormalizeVideo(mean, std)
        self.normalize_image = transforms.Normalize(mean, std)

        self.n_frms = n_frms
        self.n_actions = n_actions
        self.sample_strategy = sample_strategy

    def __call__(self, sample):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: video clip after transforms. Size is (C, T, size, size).
        """

        image, clips, clip_mask = load_video(
            sample=sample,
            n_frms=self.n_frms,
            n_actions=self.n_actions,
            sampling=self.sample_strategy
        )

        image = self.transform_image(image)
        # if clips is not None:
        transformed_clips = [self.transform_video(clip) for clip in clips]
        clips = torch.stack(transformed_clips) # (N, C, T, size, size)

        return image, clips, clip_mask


class ToUint8(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.to(torch.uint8)

    def __repr__(self):
        return self.__class__.__name__


class ToTHWC(object):
    """
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (C, T, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, H, W, C)
    """

    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.permute(1, 2, 3, 0)

    def __repr__(self):
        return self.__class__.__name__


class ResizeVideo(object):
    def __init__(self, target_size, interpolation_mode="bilinear"):
        self.target_size = target_size
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: central cropping of video clip. Size is
            (C, T, crop_size, crop_size)
        """
        return F.resize(clip, self.target_size, self.interpolation_mode)

    def __repr__(self):
        return self.__class__.__name__ + "(resize_size={0})".format(self.target_size)


@registry.register_processor("egoplan_video_train")
class EgoplanVideoTrainProcessor(EgoplanVideoBaseProcessor):
    def __init__(
        self,
        image_size=384,
        mean=None,
        std=None,
        min_scale=0.5,
        max_scale=1.0,
        n_frms=MAX_INT,
        n_actions=N_ACTIONS
    ):
        super().__init__(mean=mean, std=std, n_frms=n_frms, n_actions=n_actions, sample_strategy="headtail")

        self.image_size = image_size

        self.transform_video = transforms.Compose(
            [
                # Video size is (C, T, H, W)
                # ResizeVideo(image_size, interpolation_mode="bicubic"),
                transforms_video.RandomResizedCropVideo(
                    image_size,
                    scale=(min_scale, max_scale),
                    interpolation_mode="bicubic",
                ),
                ToTHWC(),  # C, T, H, W -> T, H, W, C
                ToUint8(),
                transforms_video.ToTensorVideo(),  # T, H, W, C -> C, T, H, W
                self.normalize_video,
            ]
        )

        self.transform_image = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(min_scale, max_scale),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                self.normalize_image,
            ]
        )

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 256)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        n_frms = cfg.get("n_frms", MAX_INT)
        n_actions = cfg.get("n_actions", N_ACTIONS)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
            n_frms=n_frms,
            n_actions=n_actions
        )


@registry.register_processor("egoplan_video_eval")
class EgoplanVideoEvalProcessor(EgoplanVideoBaseProcessor):
    def __init__(self, image_size=256, mean=None, std=None, n_frms=MAX_INT, n_actions=N_ACTIONS):
        super().__init__(mean=mean, std=std, n_frms=n_frms, n_actions=n_actions, sample_strategy="uniform")

        self.image_size = image_size

        # Input video size is (C, T, H, W)
        self.transform_video = transforms.Compose(
            [
                ResizeVideo(
                    (image_size, image_size), interpolation_mode="bicubic"
                ),
                ToUint8(),  # C, T, H, W
                ToTHWC(),  # T, H, W, C
                transforms_video.ToTensorVideo(),  # C, T, H, W
                self.normalize_video,  # C, T, H, W
            ]
        )

        self.transform_image = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize_image,
            ]
        )

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 256)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        n_frms = cfg.get("n_frms", MAX_INT)
        n_actions = cfg.get("n_actions", N_ACTIONS)

        return cls(image_size=image_size, mean=mean, std=std, n_frms=n_frms, n_actions=n_actions)
