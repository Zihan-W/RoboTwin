<h1 align="center">
	A Simple Residual Policy based on RoboTwin
</h1>

Original Work: 

RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins: [Webpage](https://robotwin-benchmark.github.io/early-version) | [PDF](https://arxiv.org/pdf/2409.02920) | [arXiv](https://arxiv.org/abs/2409.02920)<br>

# 🛠️ Installation
> Please note that you need to strictly follow the steps: **Modify `mplib` Library Code** and **Download Assert**.

See [INSTALLATION.md](./INSTALLATION.md) for installation instructions. It takes about 20 minutes for installation.

# 🧑🏻‍💻 Usage 
## 1. Task Running and Data Collection
Running the following command will first search for a random seed for the target collection quantity, and then replay the seed to collect data.
```
bash run_task.sh ${task_name_rl} ${gpu_id}
# As example: bash run_task.sh put_apple_cabinet_rl 0
# As example: bash run_task.sh dual_bottles_pick_easy_rl 0
```
This new env `put_apple_cabinet_rl` will record all data used for RL task.

If you want to modified your reward fuction, go to 'envs/' and modified `compute_reward` function in `put_apple_cabinet_rl.py` or `dual_bottles_pick_easy_rl.py`.

## 2. Task Config
> We strongly recommend you to see [Config Tutorial](./CONFIG_TUTORIAL.md) for more details.

Data collection configurations are located in the `config` folder, corresponding to each task. 

The most important setting is `head_camera_type` (default is `D435`), which directly affects the visual observation collected. This setting indicates the type of camera for the head camera, and it is aligned with the real machine. You can see its configuration in `task_config/_camera_config.yml`.

## 3. Train a base policy using BC
First, pre-process the dataset for BC
```
python script/pkl2zarr_bc.py ${task_name_rl} ${head_camera_type} ${expert_data_num}
# As example: python script/pkl2zarr_bc.py put_apple_cabinet_rl D435 40, which indicates preprocessing of 40 put_apple_cabinet task trajectory data using D435 camera.
# As example: python script/pkl2zarr_bc.py dual_bottles_pick_easy_rl D435 40
```

Then, move to `policy/Diffusion-Policy`, and run the following code to train BC:
```
bash bc.sh ${task_name} ${head_camera_type} ${expert_data_num} ${seed} ${gpu_id}
# As example: bash bc.sh put_apple_cabinet D435 40 0 0
# As example: bash bc.sh dual_bottles_pick_easy D435 40 0 0
```

[optional] If you want to evaluate your BC policy, move to root of the workspace, run the following code to evaluate BC for a specific task for 30 times:
```
python script/eval_policy.py ${task_name} ${head_camera_type} ${expert_data_num} ${checkpoint_num} ${seed} ${policy_name}
# As example: python script/eval_policy.py put_apple_cabinet_bc D435 40 200 0 bc
# if you want to change evalute times, modified the parameter test_num in script/eval_policy.py 
```

## 4. Train the residual policy using BCQ 
First, pre-process the dataset for RL
```
python script/pkl2zarr_rl.py ${task_name_rl} ${head_camera_type} ${expert_data_num}
# As example: python script/pkl2zarr_rl.py put_apple_cabinet_rl D435 40, which indicates preprocessing of 40 put_apple_cabinet task trajectory data using D435 camera.
# As example: python script/pkl2zarr_rl.py dual_bottles_pick_easy_rl D435 40
```

Then, move to `policy/Diffusion-Policy`, and run the following code to train BC+BCQ:
```
bash run_bc_bcq.sh ${task_name_bc_bcq} ${head_camera_type} ${expert_data_num} ${seed} ${gpu_id}
# As example: bash run_bc_bcq.sh put_apple_cabinet_bc_bcq D435 40 0 0
# As example: bash run_bc_bcq.sh dual_bottles_pick_easy_bc_bcq D435 40 0 0
```

[optional] If you want to evaluate your BC+BCQ policy, move to root of the workspace, run the following code to evaluate BC+BCQ for a specific task for 30 times:
```
python script/eval_policy.py ${task_name_bc_bcq} ${head_camera_type} ${expert_data_num} ${checkpoint_num} ${seed} ${policy_name}
# As example: python script/eval_policy.py put_apple_cabinet_bc_bcq D435 40 200 0 bc_bcq
# As example: python script/eval_policy.py dual_bottles_pick_easy_bc_bcq D435 40 200 0 bc_bcq
# if you want to change evalute times, modified the parameter test_num in script/eval_policy.py 
```

## 5. [optional] Train the RL policy using BCQ 
First, pre-process the dataset for RL
You can easily rename the dataset from `put_apple_cabinet_bc_bcq_D435_40.zarr` to `put_apple_cabinet_rl_D435_40.zarr` 
And then, move to `policy/Diffusion-Policy`, and run the following code to train BCQ:
```
bash run_bcq.sh ${task_name_rl} ${head_camera_type} ${expert_data_num} ${seed} ${gpu_id}
# As example: bash run_bcq.sh put_apple_cabinet_rl D435 40 0 0
# As example: bash run_bcq.sh dual_bottles_pick_easy_rl D435 40 0 0
```

[optional] If you want to evaluate your BCQ policy, move to root of the workspace, run the following code to evaluate BC for a specific task for 30 times:
```
python script/eval_policy.py ${task_name_rl} ${head_camera_type} ${expert_data_num} ${checkpoint_num} ${seed} ${policy_name}
# As example: python script/eval_policy.py put_apple_cabinet_rl D435 40 200 0 bcq
# As example: python script/eval_policy.py dual_bottles_pick_easy_rl D435 40 200 0 bcq
# if you want to change evalute times, modified the parameter test_num in script/eval_policy.py 
```

# 👍 Citation
This work is based on RoboTwin, if you find our work useful, please consider citing:

RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins (**early version**), accepted to <i style="color: red; display: inline;"><b>ECCV Workshop 2024 (Best Paper)</b></i>
```
@article{mu2024robotwin,
  title={RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins (early version)},
  author={Mu, Yao and Chen, Tianxing and Peng, Shijia and Chen, Zanxin and Gao, Zeyu and Zou, Yude and Lin, Lunkai and Xie, Zhiqiang and Luo, Ping},
  journal={arXiv preprint arXiv:2409.02920},
  year={2024}
}
```

# 🏷️ License
This repository is released under the MIT license. See [LICENSE](./LICENSE) for additional details.

