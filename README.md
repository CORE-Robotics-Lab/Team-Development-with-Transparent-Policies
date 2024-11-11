# Designs for Enabling Collaboration in
Human-Machine Teaming via Interactive and
Explainable Systems

This is the codebase for 
"Designs for Enabling Collaboration in
Human-Machine Teaming via Interactive and
Explainable Systems," which is published in NeurIPS 2024 (poster).
The presentation video of this work can be found here [FILL]. 

Authors: [Rohan Paleja](rohanpaleja.com), Michael Munje, Kimberlee Chang, Reed Jensen, and Matthew Gombolay

### Training Agent Models
As mentioned in the paper, we leverage the PantheonRL codebase to train agents and change the agent representation to be an Interpretable
Discrete Control Tree instead of the default Neural Network. The IDCT model can be found in ipm/models/idct.py and the domains 
can be found in overcooked_ai/src/overcooked_ai_py/data/layouts/two_rooms_narrow.layout and overcooked_ai/src/overcooked_ai_py/data/layouts/forced_coordination.layout

### IDCT based on ProLoNets codebase and uses ppo implementation from ProLoNets
---------------------------------------------------------------------------
`python ProLoNets/runfiles/gym_runner.py -a idct -e 2000 -env cart -rand`

### Get env that runs IDCT trainer for overcooked
-------------------------------------------------
```
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
pip install pygad
```
