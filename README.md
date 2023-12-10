# EgoPlan-Bench: Benchmarking Egocentric Embodied Planning with Multimodal Large Language Models

<a href='https://chenyi99.github.io/ego_plan/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a ><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> 
<a href='https://drive.google.com/drive/folders/1OUnQzG79kxhJdaquBKLv1rrKz36TTkP6?usp=sharing'><img src='https://img.shields.io/badge/Dataset-EgoPlan-blue'></a> 
<a href='https://huggingface.co/ChenYi99/EgoPlan-Video-LLaMA-2-7B'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoint-blue'></a> 

## Brief Introduction

<div align="center">
<p align="center">
  <img src="./figures/intro.png" width="100%" height="100%" />
</p>
</div>

Multimodal Large Language Models (MLLMs), building upon the powerful Large Language Models (LLMs)
with exceptional reasoning and generalization capability,
have opened up new avenues for embodied task planning.
MLLMs excel in their ability to integrate diverse environmental inputs, such as real-time task progress, visual observations, and open-form language instructions, which are
crucial for executable task planning. 

In this work, we introduce a benchmark with human annotations, **EgoPlan-Bench**, to quantitatively investigate the potential of MLLMs
as embodied task planners in real-world scenarios. Our
benchmark is distinguished by realistic tasks derived from
real-world videos, a diverse set of actions involving interactions with hundreds of different objects, and complex visual observations from varied environments.
We further construct an instruction-tuning
dataset **EgoPlan-IT** from videos of human-object interactions, to facilitate the learning of high-level task planning
in intricate real-world situations. 

This repository describes the usage of our evaluation data EgoPlan-Bench and instruction-tuning data EgoPlan-IT, and provides the corresponding codes for evaluating and fine-tuning MLLMs on our benchmark. 
Welcome to evaluate your models and explore methods to enhance the models' EgoPlan capabilities on our benchmark!

## Usage

### 1. Installation
Clone the repo and install dependent packages:

  ```bash
  git clone https://github.com/ChenYi99/EgoPlan.git
  cd EgoPlan
  pip install -r requirements.txt
  ```

### 2. EgoPlan Datasets
#### Egocentric Videos
Download the RGB frames of [Epic-Kitchens-100](https://github.com/epic-kitchens/epic-kitchens-download-scripts). The folder structure of the dataset is shown below:
```
EPIC-KITCHENS
└── P01
    └── rgb_frames
        └── P01_01
            ├── frame_0000000001.jpg
            └── ...
```

Download the videos of [Ego4D](https://ego4d-data.org/#download). The folder structure of the dataset is shown below:
```
Ego4D
└──v1_288p
    ├── 000786a7-3f9d-4fe6-bfb3-045b368f7d44.mp4
    └── ...
```

#### EgoPlan-Bench
Download our evaluation dataset [EgoPlan-Bench](https://drive.google.com/drive/folders/1hn5vgfz0fMNlSm6p7C-LpuoMdZrOffLB?usp=sharing). 
There are two evaluation subsets, [EgoPlan_Bench_EpicKitchens.json](https://drive.google.com/file/d/1YYJeZVhwqV2QRIh_8w_qnksEL7arYEZu/view?usp=sharing) and [EgoPlan_Bench_Ego4D.json](https://drive.google.com/file/d/1NJlcqi4Xd1GEVXnm--_73YRyVRCexZ-X/view?usp=sharing). 
Put these two JSON files under the directory [data/](data).
Below is an example of a single data sample in the evaluation dataset:
```
{
    "sample_id": 115,
    "video_id": "P01_13",
    "task_goal": "store cereal",
    "question": "Considering the progress shown in the video and my current observation in the last frame, what action should I take next in order to store cereal?",
    "choice_a": "put cereal box into cupboard",
    "choice_b": "take cereal bag",
    "choice_c": "open cupboard",
    "choice_d": "put cereal bag into cereal box",
    "golden_choice_idx": "A",
    "answer": "put cereal box into cupboard",
    "current_observation_frame": 760,
    "task_progress_metadata": [
        {
            "narration_text": "take cereal bag",
            "start_frame": 36,
            "stop_frame": 105
        },
        {
            "narration_text": "fold cereal bag",
            "start_frame": 111,
            "stop_frame": 253
        },
        {
            "narration_text": "put cereal bag into cereal box",
            "start_frame": 274,
            "stop_frame": 456
        },
        {
            "narration_text": "close cereal box",
            "start_frame": 457,
            "stop_frame": 606
        },
        {
            "narration_text": "open cupboard",
            "start_frame": 689,
            "stop_frame": 760
        }
    ],  
}
```

#### EgoPlan-IT
Download our instruction-tuning dataset [EgoPlan-IT](https://drive.google.com/drive/folders/1y-zkGcIofRfZyOaznbflWmb9qrKv09ws?usp=sharing).
Put the JSON file [EgoPlan-IT.json](https://drive.google.com/file/d/1dhV4xIkoWCfXnBOh_ez4b0inDeN_PDBH/view?usp=sharing) under the directory [data/](data).
Below is an example of a single data sample in the instruction-tuning dataset:
```
{
    "sample_id": 39,
    "video_id": "P07_113",
    "task_goal": "Cut and peel the onion",
    "question": "Considering the progress shown in the video and my current observation in the last frame, what action should I take next in order to cut and peel the onion?",
    "answer": "grab onion",
    "current_observation_frame": 9308,
    "task_progress_metadata": [
        {
            "narration_text": "open drawer",
            "start_frame": 9162,
            "stop_frame": 9203
        },
        {
            "narration_text": "grab knife",
            "start_frame": 9214,
            "stop_frame": 9273
        },
        {
            "narration_text": "close drawer",
            "start_frame": 9272,
            "stop_frame": 9303
        }
    ],
    "negative_answers": [
        "open drawer",
        "grab knife",
        "close drawer",
        "slice onion",
        "remove peel from onion",
        "peel onion"
    ]
}
```

### 3. Model Weights
We use [Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA) as an example for evaluation and instruction-tuning.

#### Prepare the pretrained model checkpoints
- The checkpoint of the vanilla Video-LLaMA can be downloaded from
[Video-LLaMA-2-7B-Finetuned](https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned).
- The checkpoint of the Video-LLaMA that has been further tuned on our EgoPlan-IT can be downloaded from
[EgoPlan-Video-LLaMA-2-7B](https://huggingface.co/ChenYi99/EgoPlan-Video-LLaMA-2-7B).

#### Prepare the pretrained LLM weights
Video-LLaMA is based on Llama2 Chat 7B. The corresponding LLM weights can be downloaded from [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).

#### Prepare weights for initializing the Visual Encoder and Q-Former (optional)
If the server cannot access the Internet, the following weights should be downloaded in advance:<br>
- VIT ([eva_vit_g.pth](https://link.zhihu.com/?target=https%3A//storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pt))<br>
- Q-Former ([blip2_pretrained_flant5xxl.pth](https://link.zhihu.com/?target=https%3A//storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth))<br>
- Bert ([bert-base-uncased](https://link.zhihu.com/?target=https%3A//huggingface.co/bert-base-uncased))

### 4. Evaluation on EgoPlan-Bench
#### Evaluating the Vanilla Video-LLaMA

Config the paths for model weights in [video_llama_eval_only_vl.yaml](src/video_llama/eval_configs/video_llama_eval_only_vl.yaml).<br>
Set the paths for the `project root`, `Epic-Kitchens-100 RGB frames` and `Ego4D videos` in [eval_video_llama.sh](scripts/eval_video_llama.sh).<br>
Then, run the script on 1xV100 (32G) GPU:
```bash
bash scripts/eval_video_llama.sh
```

#### Evaluating the Video-LLaMA Tuned on EgoPlan-IT
Config the paths for model weights in [egoplan_video_llama_eval_only_vl.yaml](src/video_llama/eval_configs/egoplan_video_llama_eval_only_vl.yaml).<br>
Set the paths for the `project root`, `Epic-Kitchens-100 RGB frames` and `Ego4D videos` in [eval_egoplan_video_llama.sh](scripts/eval_egoplan_video_llama.sh).<br>
Then, run the script on 1xV100 (32G) GPU:
```bash
bash scripts/eval_egoplan_video_llama.sh
```

### 5. Fine-tuning on EgoPlan-IT
For increasing instruction diversity, in addition to EgoPlan-IT, we also include an additional collection of 164K instruction data following [Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA):

* 3K image-based instructions from MiniGPT-4 [[link](https://github.com/Vision-CAIR/MiniGPT-4/blob/main/dataset/README_2_STAGE.md)]. 
* 150K image-based instructions from LLaVA [[link](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/raw/main/llava_instruct_150k.json)]. The images can be downloaded from [here](http://images.cocodataset.org/zips/train2014.zip).
* 11K video-based instructions from VideoChat [[link](https://github.com/OpenGVLab/InternVideo/tree/main/Data/instruction_data)]. The videos can be downloaded following the instructions from the official Github repo of [Webvid](https://github.com/m-bain/webvid).



Config the paths for model weights and datasets in [visionbranch_stage3_finetune_on_EgoPlan_IT.yaml](src/video_llama/train_configs/visionbranch_stage3_finetune_on_EgoPlan_IT.yaml).<br>
Set the path for the `project root` in [finetune_egoplan_video_llama.sh](scripts/finetune_egoplan_video_llama.sh).<br>
Then, run the script on 8xV100 (32G) GPUs:
```bash
bash scripts/finetune_egoplan_video_llama.sh
```

## Acknowledgement
This repo benefits from [Epic-Kitchens](https://epic-kitchens.github.io/2023), [Ego4D](https://ego4d-data.org/), 
[Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA), 
[LLaMA](https://github.com/facebookresearch/llama),
[MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), 
[LLaVA](https://github.com/haotian-liu/LLaVA), 
[VideoChat](https://github.com/OpenGVLab/Ask-Anything). Thanks for their wonderful works!
