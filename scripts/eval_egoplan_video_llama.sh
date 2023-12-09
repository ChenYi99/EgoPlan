MODEL_NAME="egoplan_video_llama"

PROJECT_ROOT="path/EgoPlan"
export PYTHONPATH=${PYTHONPATH}:${PROJECT_ROOT}

cd ${PROJECT_ROOT}
nohup python3 -u eval_multiple_choice.py \
--model ${MODEL_NAME} \
--epic_kitchens_rgb_frame_dir path/EPIC-KITCHENS/ \
--ego4d_video_dir path/Ego4D/v1_288p/ \
> eval_multiple_choice_${MODEL_NAME}.log 2>&1 &

