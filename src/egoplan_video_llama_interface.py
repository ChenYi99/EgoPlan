from src.video_llama.video_llama.common.config import Config
from src.video_llama.video_llama.common.dist_utils import get_rank
from src.video_llama.video_llama.common.registry import registry
from src.video_llama.video_llama.conversation.conversation_video import StoppingCriteriaSub, default_conversation, conv_llava_llama_2
from src.video_llama.video_llama.datasets.datasets.egoplan_video_instruct_dataset import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_AUDIO_PATCH_TOKEN,
    NUM_VIDEO_QUERY_TOKEN,
    N_ACTIONS,
    convert_source_vicuna_format,
    preprocess_multimodal,
    preprocess,
    preprocess_for_llama_v2,
    create_question_for_egoplan_task_planning
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

class Arguments:
    def __init__(self,
                 cfg_path,
                 model_type):
        self.cfg_path = cfg_path
        self.model_type = model_type
        self.options = None


def ego4d_video_image_process(sample, vis_processor, n_frms=8, n_actions=4):
    video_path = sample["video_path"]
    video = cv2.VideoCapture(video_path)

    task_progress_metadata = sample["task_progress_metadata"]
    clips = []
    for action_metadata in task_progress_metadata[-n_actions:]:
        clip = []
        start_frame_idx = action_metadata["start_frame"]
        end_frame_idx = action_metadata["stop_frame"]
        step_size = (end_frame_idx-start_frame_idx) // n_frms
        for i in range(start_frame_idx, end_frame_idx, step_size):
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame).convert("RGB")
                clip.append(frame)
        clip = clip[:n_frms]
        assert len(clip) == n_frms
        clip = np.stack(clip)  # T, H, W, C
        clip = torch.tensor(clip).float().permute(3, 0, 1, 2)  # C, T, H, W
        clips.append(clip)

    current_observation_frame_idx = sample["current_observation_frame"]
    video.set(cv2.CAP_PROP_POS_FRAMES, current_observation_frame_idx)
    ret, image = video.read()
    image = Image.fromarray(image).convert("RGB")

    video.release()
    cv2.destroyAllWindows()

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

    image = vis_processor.transform_image(image)
    transformed_clips = [vis_processor.transform_video(clip) for clip in clips]
    clips = torch.stack(transformed_clips)  # (N, C, T, size, size)

    clip_mask = clip_mask.bool()
    return image, clips, clip_mask

@torch.no_grad()
def predict_choice(model, vis_processor, model_type, sample, return_loss=True, subset_name="EpicKitchens"):
    llama_tokenizer = model.llama_tokenizer

    if subset_name == "EpicKitchens":
        image, clips, clip_mask = vis_processor(sample)
    elif subset_name == "Ego4D":
        image, clips, clip_mask = ego4d_video_image_process(sample, vis_processor)
    else:
        print('not support')
        raise 'not support'

    task_goal = sample["task_goal"]
    question = create_question_for_egoplan_task_planning(task_goal, mode='eval')
    sample["question"] = question

    candidates = []
    for choice_idx in ["A", "B", "C", "D"]:
        candidates.append(sample[f"choice_{choice_idx.lower()}"])

    sources = []
    for candidate in candidates:
        conversation_list = [{'q': question,
                              'a': candidate}]
        sources.extend(preprocess_multimodal(
            conversation_list=copy.deepcopy(conversation_list),
            image_token_len=NUM_VIDEO_QUERY_TOKEN,
            n_actions=N_ACTIONS,
            msg=''))

    new_sources = convert_source_vicuna_format(sources)

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


    data_dict['attention_mask'] = data_dict['input_ids'].ne(llama_tokenizer.pad_token_id)
    data_dict['image'] = torch.repeat_interleave(image.unsqueeze(0), len(candidates), dim=0)
    data_dict['clips'] = torch.repeat_interleave(clips.unsqueeze(0), len(candidates), dim=0)
    data_dict['clip_mask'] = torch.repeat_interleave(clip_mask.unsqueeze(0), len(candidates), dim=0)
    for k, v in data_dict.items():
        data_dict[k] = v.cuda()

    data_dict['conv_type'] = 'egoplan'

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



def build():
    config_path = "src/video_llama/eval_configs/egoplan_video_llama_eval_only_vl.yaml"
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

