import os
import os.path as osp
import sys
import json
import argparse
from PIL import Image
import torch
import pdb
from src import build
from tqdm import tqdm
import torch
import numpy as np
import random
import string

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='egoplan_video_llama')
    parser.add_argument('--epic_kitchens_rgb_frame_dir', type=str)
    parser.add_argument('--ego4d_video_dir', type=str)
    args = parser.parse_args()

    print(f'evaluating.. {args.model}')
    predict_choice = build(args.model)

    with open(f"accuracy_for_{args.model}.txt", "w") as fo:
        total_correct_num = 0
        total_question_num = 0
        for subset_name in ["EpicKitchens", "Ego4D"]:
            subset_path = os.path.join("data", f"EgoPlan_Bench_{subset_name}.json")
            with open(subset_path) as fi:
                samples = json.load(fi)

            correct_num = 0
            question_num = 0
            for i, sample in tqdm(enumerate(samples), desc="Processing questions", total=len(samples)):
                # if i >= 5:
                #     break

                print("\n" + "-" * 50 + f" {subset_name}-sample-{sample['sample_id']} " + "-" * 50)

                if subset_name == "EpicKitchens":
                    video_id = sample["video_id"]
                    participant_id = video_id.split("_")[0]
                    video_rgb_frame_dir = os.path.join(args.epic_kitchens_rgb_frame_dir,
                                                       participant_id, "rgb_frames", video_id)
                    sample["video_rgb_frame_dir"] = video_rgb_frame_dir
                else:
                    video_id = sample["video_id"]
                    video_path = os.path.join(args.ego4d_video_dir, f"{video_id}.mp4")
                    sample["video_path"] = video_path

                predicted_choice, choice2loss = predict_choice(sample=sample, return_loss=True, subset_name=subset_name)
                print("***** question *****")
                print(sample["question"])

                print("***** golden choice *****")
                print(sample["answer"])

                print("***** predicted choice *****")
                print(predicted_choice)

                print("***** predicted choice2loss *****")
                print(choice2loss)

                if sample["answer"] == predicted_choice:
                    correct_num += 1
                question_num += 1

            fo.write("{} accuracy of {} questions: {:.4f}\n".format(subset_name, question_num, correct_num / question_num))
            total_correct_num += correct_num
            total_question_num += question_num
        fo.write("Overall accuracy of {} questions: {:.4f}\n".format(total_question_num, total_correct_num / total_question_num))