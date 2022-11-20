# How to run

### IDCT based on ICCT codebase and uses stable-baselines3 ppo
----------------------------------------------------------
`python ICCT/icct/runfiles/train_idct.py --env_name cartpole  --visualization_output lunar_lander_tree.png --policy_type ddt   --seed 0   --num_leaves 4   --lr 5e-1   --ddt_lr 1e-1   --buffer_size 1000000   --batch_size 256   --gamma 0.99   --learning_starts 10000   --eval_freq 5000   --min_reward 225   --training_steps 500000   --log_interval 4   --save_path log/cartpole   --use_individual_alpha  --submodels --hard_node   --gpu   --argmax_tau 1.0`

### IDCT based on ProLoNets codebase and uses ppo implementation from ProLoNets
---------------------------------------------------------------------------
`python ProLoNets/runfiles/gym_runner.py -a idct -e 2000 -env cart -rand`

### Get env that runs IDCT trainer for overcooked
-------------------------------------------------
conda create --name overcooked_trainer
conda activate overcooked_trainer
conda install pip
pip install -e .
git clone https://github.com/Stanford-ILIAD/PantheonRL.git
cd PantheonRL
pip install -e .
git submodule update --init --recursive
pip install -e overcookedgym/human_aware_rl/overcooked_ai
pip install stable_baselines3==1.2.0
