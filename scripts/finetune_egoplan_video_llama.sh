PROJECT_ROOT="path/EgoPlan"
export PYTHONPATH=${PYTHONPATH}:${PROJECT_ROOT}

cd ${PROJECT_ROOT}/src/video_llama
python3 -u -m torch.distributed.run \
--nproc_per_node=8 train.py \
--cfg-path ./train_configs/visionbranch_stage3_finetune_on_EgoPlan_IT.yaml