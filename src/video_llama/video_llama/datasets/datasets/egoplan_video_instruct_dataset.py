import sys
import os
from src.video_llama.video_llama.datasets.datasets.base_dataset import BaseDataset
from src.video_llama.video_llama.datasets.datasets.caption_datasets import CaptionDataset
import pandas as pd
import decord
from decord import VideoReader
import random
import torch
from torch.utils.data.dataloader import default_collate
from PIL import Image
from typing import Dict, Optional, Sequence
import transformers
import pathlib
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
import copy
from src.video_llama.video_llama.processors import transforms_video, EgoplanVideoTrainProcessor

from torchvision import transforms
from src.video_llama.video_llama.processors.video_processor import ToTHWC,ToUint8,load_video
from src.video_llama.video_llama.conversation.conversation_video import Conversation,SeparatorStyle
import string
import random
from itertools import chain

DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'
DEFAULT_AUDIO_PATCH_TOKEN = '<AudioHere>'
NUM_VIDEO_QUERY_TOKEN = 32
N_ACTIONS = 4

video_conversation = Conversation(
    system="",
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)
llama_v2_video_conversation = Conversation(
    system=" ",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)
IGNORE_INDEX = -100

class Egoplan_Video_Instruct_Dataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root,
                 ann_root, num_video_query_token=32,
                 tokenizer_name = '/mnt/workspace/ckpt/vicuna-13b/', data_type = 'video',
                 model_type='vicuna', n_actions=4, answer_type='egoplan_qa'):
        """
        vis_root (string): Root directory of Llava images (e.g. webvid_eval/video/)
        ann_root (string): Root directory of video (e.g. webvid_eval/annotations/)
        split (string): val or test
        """
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        data_path = pathlib.Path(ann_root)
        with data_path.open(encoding='utf-8') as f:
            self.annotation = json.load(f)

        # self.annotation = self.annotation[:100]
        # print(self.annotation[:5])
        # print(f"annotation_len: {len(self.annotation)}")
        self.num_video_query_token = num_video_query_token
        self.vis_root = vis_root
        self.resize_size = 224
        self.num_frm = 8
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.tokenizer.add_tokens([DEFAULT_AUDIO_PATCH_TOKEN], special_tokens=True)

        self.IMAGE_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[DEFAULT_IMAGE_PATCH_TOKEN]
        self.AUDIO_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[DEFAULT_AUDIO_PATCH_TOKEN]

        self.data_type = data_type
        self.model_type = model_type
        self.n_actions = n_actions
        self.answer_type = answer_type

        # print(f"model_type: {self.model_type}")
        # print(f"n_actions: {self.n_actions}")
        # print(f"answer_type: {self.answer_type}")

    def _get_video_rgb_frame_dir(self, sample):
        video_id = sample["video_id"]
        participant_id = video_id.split("_")[0]
        video_rgb_frame_dir = os.path.join(self.vis_root, participant_id, "rgb_frames", video_id)
        return video_rgb_frame_dir

    def __getitem__(self, index):
        # print(index)
        num_retries = 10  # skip error videos
        for _ in range(num_retries):
            try:
                sample = self.annotation[index]
                video_rgb_frame_dir = self._get_video_rgb_frame_dir(sample)
                sample["video_rgb_frame_dir"] = video_rgb_frame_dir

                # print(sample)

                if self.answer_type == "egoplan_qa":
                    task_goal = sample['task_goal']
                    question = create_question_for_egoplan_task_planning(task_goal=task_goal,
                                                                          mode='train')
                    answer = sample['answer']
                else:
                    candidate_questions = ["What exact actions were depicted in the video? Please list them in order, describing each action with a brief verb-noun phrase.",
                                           "Can you enumerate the actions in the video, describing each with a short verb-noun combination?",
                                           "Could you break down the individual actions from the video? Use succinct verb-noun pairs to describe each one in sequence.",
                                           "Can you dissect the video's content into distinct actions? Please use verb-noun pairs to outline them in sequence.",
                                           "Can you detail the specific actions shown in the video? List them sequentially using a short verb-noun description."]
                    question = random.choice(candidate_questions)

                    current_observation_frame_idx = sample["current_observation_frame"]
                    task_progress_metadata = sample["task_progress_metadata"]
                    most_recent_actions_metadata = list(
                        filter(lambda item: item['stop_frame'] <= current_observation_frame_idx,
                               task_progress_metadata))[-self.n_actions:]
                    # most_recent_actions_metadata = task_progress_metadata[-self.n_actions:]
                    observed_actions = []
                    for action_metadata in most_recent_actions_metadata:
                        observed_actions.append(action_metadata["narration_text"])
                    if len(observed_actions) > 0:
                        answer = f"{', '.join(observed_actions)}."
                    else:
                        answer = "No meaningful action occurred in the video."

                # print("q: "+question+"\n")
                # print("a: "+answer+"\n\n")

                conversation_list = [{'q': question,
                                      'a': answer}]

                image, clips, clip_mask = self.vis_processor(sample)

                sources = preprocess_multimodal(conversation_list=copy.deepcopy(conversation_list),
                                                image_token_len=self.num_video_query_token,
                                                n_actions=self.n_actions,
                                                msg='')
                new_sources = convert_source_vicuna_format(sources)

                if self.model_type =='vicuna':
                    data_dict = preprocess(
                        new_sources,
                        self.tokenizer)
                elif self.model_type =='llama_v2':
                    data_dict = preprocess_for_llama_v2(
                        new_sources,
                        self.tokenizer)
                else:
                    print('not support')
                    raise('not support')
                data_dict = dict(input_ids=data_dict["input_ids"][0],
                                labels=data_dict["labels"][0])
                # image exist in the data
                data_dict['image'] = image
                data_dict['clips'] = clips
                data_dict['clip_mask'] = clip_mask
            except Exception as e:
                print(e)
                print(f"Failed to load examples with video: {sample['video_id']}. "
                      f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            break
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")

        return {
            'image': data_dict['image'],
            'clips': data_dict['clips'],
            'clip_mask': data_dict['clip_mask'],
            "text_input": data_dict["input_ids"],
            "labels": data_dict["labels"],
            "type":'video',
        }

    def __len__(self):
        return len(self.annotation)

    def collater(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("text_input", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        for k in ['image', 'clips', 'clip_mask']:
            values = [instance[k] for instance in instances]
            batch[k] = torch.stack(values)
            # print(f"{k}: {batch[k].shape}")

        batch['conv_type'] = 'egoplan'
        return batch


class Egoplan_Video_Contrastive_Dataset(Egoplan_Video_Instruct_Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_llm_input_ids_and_labels(self, question, answer):
        conversation_list = [{'q': question,
                              'a': answer}]
        sources = preprocess_multimodal(conversation_list=copy.deepcopy(conversation_list),
                                        image_token_len=self.num_video_query_token,
                                        n_actions=self.n_actions,
                                        msg='')
        new_sources = convert_source_vicuna_format(sources)

        if self.model_type == 'vicuna':
            data_dict = preprocess(
                new_sources,
                self.tokenizer)
        elif self.model_type == 'llama_v2':
            data_dict = preprocess_for_llama_v2(
                new_sources,
                self.tokenizer)
        else:
            print('not support')
            raise ('not support')
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                         labels=data_dict["labels"][0])
        return data_dict

    def __getitem__(self, index):
        # print(index)
        num_retries = 10  # skip error videos
        for _ in range(num_retries):
            try:
                sample = self.annotation[index]
                video_rgb_frame_dir = self._get_video_rgb_frame_dir(sample)
                sample["video_rgb_frame_dir"] = video_rgb_frame_dir

                # print(sample)

                task_goal = sample['task_goal']
                question = create_question_for_egoplan_task_planning(task_goal=task_goal,
                                                                      mode='train')
                answer = sample['answer']
                image, clips, clip_mask = self.vis_processor(sample)
                data_dict = self.get_llm_input_ids_and_labels(question, answer)

                # sample negative_answer
                negative_answers = sample["negative_answers"]
                negative_answer = random.choice(negative_answers)
                negative_data_dict = self.get_llm_input_ids_and_labels(question, negative_answer)
                # print(f"question: {question}\n"
                #       f"positive_answer: {answer}\n"
                #       f"negative_answer: {negative_answer}")

                # image exist in the data
                data_dict['image'] = image
                data_dict['clips'] = clips
                data_dict['clip_mask'] = clip_mask
                data_dict['negative_input_ids'] = negative_data_dict["input_ids"]
                data_dict['negative_labels'] = negative_data_dict["labels"]


            except Exception as e:
                print(e)
                print(f"Failed to load examples with video: {sample['video_id']}. "
                      f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            break
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            'image': data_dict['image'],
            'clips': data_dict['clips'],
            'clip_mask': data_dict['clip_mask'],
            "text_input": data_dict["input_ids"],
            "labels": data_dict["labels"],
            "negative_text_input": data_dict["negative_input_ids"],
            "negative_labels": data_dict["negative_labels"],
            "type":'video',
        }

    def collater(self, instances):
        positive_input_ids, positive_labels = tuple([instance[key] for instance in instances]
                                  for key in ("text_input", "labels"))
        negative_input_ids, negative_labels = tuple([instance[key] for instance in instances]
                                                    for key in ("negative_text_input", "negative_labels"))
        batch_size = len(positive_input_ids)
        all_input_ids = torch.nn.utils.rnn.pad_sequence(
            positive_input_ids + negative_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        all_labels = torch.nn.utils.rnn.pad_sequence(positive_labels + negative_labels,
                                                     batch_first=True,
                                                     padding_value=IGNORE_INDEX)

        input_ids = all_input_ids[:batch_size]
        labels = all_labels[:batch_size]
        negative_input_ids = all_input_ids[batch_size:]
        negative_labels = all_labels[batch_size:]

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            negative_input_ids=negative_input_ids,
            negative_labels=negative_labels,
            negative_attention_mask=negative_input_ids.ne(self.tokenizer.pad_token_id),
        )

        for k in ['image', 'clips', 'clip_mask']:
            values = [instance[k] for instance in instances]
            batch[k] = torch.stack(values)

        batch['conv_type'] = 'egoplan_contrastive'
        return batch


def convert_source_vicuna_format(sources):
    new_sources = []
    for source in sources:
        new_source = []
        for i, sentence in enumerate(source):
            role_0_msg = sentence['q']
            role_1_msg = sentence['a']
            new_source.append({
                'from':'human',
                'value': role_0_msg,
            })
            new_source.append({
                'from':'gpt',
                'value': role_1_msg,
            })
        new_sources.append(new_source)
    return new_sources

def preprocess_multimodal(
    conversation_list: Sequence[dict],
    image_token_len: int,
    n_actions: int,
    msg=''
) -> Dict:
    conversation_list[0]["q"] = "<Video>" + DEFAULT_AUDIO_PATCH_TOKEN * image_token_len * n_actions + "</Video> " \
                                + "<Image>" + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + "</Image> " \
                                + msg + conversation_list[0]["q"]
    return [conversation_list]

def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "###"
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = video_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = video_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation
        
def _tokenize_fn(strings: Sequence[str],
                tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=512,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{video_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source],
                                      tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

def preprocess_for_llama_v2(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # add end signal and concatenate together
    conversations = []
    conv = copy.deepcopy(llama_v2_video_conversation.copy())
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    for source in sources:
        # <s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n
        header = f"<s>[INST] <<SYS>>\n{conv.system}\n</SYS>>\n\n"

        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2]
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # print(conversations)
    input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=512,
            truncation=True,
        ).input_ids
    targets = copy.deepcopy(input_ids)


    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        # total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2 # 为什么减去2,speical token 的数目

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)

def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


prompt_list = [
    [
        "I am tasked with {}. "
        "The task's progress is demonstrated in the provided video. "
        "My current field of view is shown in the provided image. "
        "What should be my next action? "
        "Please output the most reasonable action you think, expressed in a short phrase.",

        "As I am responsible for {}, "
        "the video illustrates the task's progression, "
        "and the image represents my current viewpoint. "
        "What would be the most sensible action to take next? Please offer a concise suggestion.",

        "I am in charge of {}, "
        "and the video reveals the task's advancement, "
        "along with an image of my current view. "
        "What is the most rational next move? Please propose a short and reasonable action.",

        "As I am in the process of {}, "
        "with my progress visible in the video "
        "and my viewpoint displayed in the image, "
        "what do you suggest as my next move? Kindly offer a concise suggestion.",
    ],

    [
        "My current task is to {}. "
        "The task's progress is demonstrated in the provided video. "
        "My current field of view is shown in the provided image. "
        "What should be my next action? "
        "Please output the most reasonable action you think, expressed in a short phrase.",

        "Given my responsibility to {}, "
        "the video shows the progress, "
        "and the image displays my current view. "
        "What is the most logical next step? Please provide a brief response.",

        "My assignment is to {}, "
        "with the task's progress evident in the video "
        "and my current perspective shown in the image. "
        "What should I do next? Please provide a reasonable and succinct recommendation.",

        "I have been assigned to {}, "
        "and the video demonstrates my progress, "
        "while the image presents my current visual field. "
        "What is the most appropriate next action? Please share a brief and practical suggestion.",
    ],
]


def create_question_for_egoplan_task_planning(task_goal, mode='train'):
    task_goal = task_goal.strip(string.punctuation + " ").lower()
    if "goal" in task_goal:
        task_goal = task_goal.split("to", 1)[1].strip()
    words = task_goal.split()

    if mode == 'train':
        if words[0].endswith("ing"):
            question_pattern = random.choice(prompt_list[0])
        else:
            question_pattern = random.choice(prompt_list[1])
    else:
        if words[0].endswith("ing"):
            question_pattern = "I am tasked with {}. " \
                               "The task's progress is demonstrated in the provided video. " \
                               "My current field of view is shown in the provided image. " \
                               "What should be my next action? " \
                               "Please output the most reasonable action you think, expressed in a short phrase."
        else:
            question_pattern = "My current task is to {}. " \
                               "The task's progress is demonstrated in the provided video. " \
                               "My current field of view is shown in the provided image. " \
                               "What should be my next action? " \
                               "Please output the most reasonable action you think, expressed in a short phrase."

    question = question_pattern.format(task_goal)
    return question

if __name__ == '__main__':
    create_question_for_egoplan_task_planning(task_goal="making coffee")
    create_question_for_egoplan_task_planning(task_goal="prepare breakfast")
