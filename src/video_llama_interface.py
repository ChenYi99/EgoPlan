from src.video_llama.video_llama.common.config import Config
from src.video_llama.video_llama.common.dist_utils import get_rank
from src.video_llama.video_llama.common.registry import registry
from src.video_llama.video_llama.conversation.conversation_video import StoppingCriteriaSub, default_conversation, conv_llava_llama_2
from src.video_llama.video_llama.datasets.datasets.video_instruct_dataset import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    convert_source_vicuna_format,
    preprocess_multimodal,
    preprocess,
    preprocess_for_llama_v2
)
from transformers import LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
import torch
from functools import partial
import string
import copy
import time
import numpy as np
import cv2
from PIL import Image
import os

NUM_VIDEO_QUERY_TOKEN = 32

class Arguments:
    def __init__(self,
                 cfg_path,
                 model_type):
        self.cfg_path = cfg_path
        self.model_type = model_type
        self.options = None


def ego4d_video_process(sample, vis_processor, n_frms=8):
    video_path = sample["video_path"]
    video = cv2.VideoCapture(video_path)

    current_observation_frame_idx = sample["current_observation_frame"]
    video.set(cv2.CAP_PROP_POS_FRAMES, current_observation_frame_idx)
    ret, image = video.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image).convert("RGB")

    task_progress_metadata = sample["task_progress_metadata"]
    if len(task_progress_metadata) > 0:
        start_frame_idx = task_progress_metadata[0]["start_frame"]
        end_frame_idx = current_observation_frame_idx
        frame_indices = np.arange(start_frame_idx, end_frame_idx,
                                  (end_frame_idx - start_frame_idx) / (n_frms-1)).astype(int).tolist()

        clip = []
        for i in frame_indices:
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame).convert("RGB")
                clip.append(frame)
        clip = clip[:n_frms-1]
        clip.append(image)
    else:
        clip = [image] * n_frms

    clip = np.stack(clip)  # T, H, W, C
    clip = torch.tensor(clip).float().permute(3, 0, 1, 2)  # C, T, H, W
    clip = vis_processor.transform_video(clip)
    return clip

def epic_kitchens_video_process(sample, vis_processor, n_frms=8):
    current_observation_frame_idx = sample["current_observation_frame"]
    video_rgb_frame_dir = sample["video_rgb_frame_dir"]

    image_path = os.path.join(video_rgb_frame_dir, f"frame_{str(current_observation_frame_idx).zfill(10)}.jpg")
    image = Image.open(image_path).convert('RGB')  # H, W, C

    task_progress_metadata = sample["task_progress_metadata"]
    if len(task_progress_metadata) > 0:
        start_frame_idx = task_progress_metadata[0]["start_frame"]
        end_frame_idx = current_observation_frame_idx
        frame_indices = np.arange(start_frame_idx, end_frame_idx,
                                  (end_frame_idx - start_frame_idx) / (n_frms-1)).astype(int).tolist()

        clip = []
        for i in frame_indices:
            frame_path = os.path.join(video_rgb_frame_dir, f"frame_{str(i).zfill(10)}.jpg")
            if os.path.exists(frame_path):
                frame = Image.open(frame_path).convert('RGB')
                clip.append(frame)
            else:
                print(f"image_path doesn't exist!! {frame_path}")
                raise FileNotFoundError
        clip = clip[:n_frms - 1]
        clip.append(image)
    else:
        clip = [image] * n_frms

    clip = np.stack(clip)  # T, H, W, C
    clip = torch.tensor(clip).float().permute(3, 0, 1, 2)  # C, T, H, W
    clip = vis_processor.transform_video(clip)
    return clip

@torch.no_grad()
def predict_choice(model, vis_processor, model_type, sample, return_loss=True, subset_name="EpicKitchens"):
    llama_tokenizer = model.llama_tokenizer

    if subset_name == "EpicKitchens":
        clip = epic_kitchens_video_process(sample, vis_processor)
    elif subset_name == "Ego4D":
        clip = ego4d_video_process(sample, vis_processor)
    else:
        print('not support')
        raise 'not support'

    question = sample["question"]

    candidates = []
    for choice_idx in ["A", "B", "C", "D"]:
        candidates.append(sample[f"choice_{choice_idx.lower()}"])

    sources = []
    for candidate in candidates:
        conversation_list = [{'q': question,
                              'a': candidate}]
        sources.extend(preprocess_multimodal(
            copy.deepcopy(conversation_list),
            None,
            cur_token_len=NUM_VIDEO_QUERY_TOKEN,
            msg=''))

    new_sources = convert_source_vicuna_format(sources)
    # print(new_sources)
    if model_type == 'vicuna':
        data_dict = preprocess(
            new_sources,
            llama_tokenizer)
    elif model_type == 'llama_v2':
        data_dict = preprocess_for_llama_v2(
            new_sources,
            llama_tokenizer)
    else:
        print('not support')
        raise 'not support'

    data_dict["images"] = torch.repeat_interleave(clip.unsqueeze(0), len(candidates), dim=0)
    data_dict['attention_mask'] = data_dict['input_ids'].ne(llama_tokenizer.pad_token_id)
    for k, v in data_dict.items():
        data_dict[k] = v.cuda()

    data_dict['conv_type'] = 'multi'

    all_losses = model(samples=data_dict, reduction='none')["loss"]
    predicted_choice_idx = torch.argmin(all_losses, dim=-1).item()
    predicted_choice = candidates[predicted_choice_idx]

    if return_loss:
        choice2loss = {}
        for choice, loss in zip(candidates, all_losses.cpu().tolist()):
            choice2loss[choice] = loss
        choice2loss = dict(sorted(choice2loss.items(), key=lambda item: item[1]))
        return predicted_choice, choice2loss
    else:
        return predicted_choice



def build(config_path = "src/video_llama/eval_configs/video_llama_eval_only_vl.yaml"):
    args = Arguments(cfg_path=config_path,
                     model_type="llama_v2")
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.device_8bit = 0
    model_cls = registry.get_model_class(model_config.arch)

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model = model_cls.from_config(model_config).to(device)
    model.eval()

    vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.eval
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)


    return partial(predict_choice, model=model, vis_processor=vis_processor,
                   model_type=args.model_type, return_loss=True)

