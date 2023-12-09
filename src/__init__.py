from .egoplan_video_llama_interface import build as build_egoplan_video_llama
from .video_llama_interface import build as build_video_llama

def build(model_name):
    if model_name == 'video_llama':
        return build_video_llama()
    elif model_name == 'egoplan_video_llama':
        return build_egoplan_video_llama()

    print(f"model {model_name} not exist")
    exit(0)