DEBUG=False

task_name=${1}
head_camera_type=${2}
expert_data_num=${3}
checkpoint_num=${4}
seed=${5}
gpu_id=${6}

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

python ./script/eval_policy.py "$task_name" "$head_camera_type" "$expert_data_num" "$checkpoint_num" "$seed" "$policy_name"