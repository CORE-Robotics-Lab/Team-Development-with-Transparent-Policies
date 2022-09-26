set -e

conda create -n ipm_testing python=3.7 anaconda
pip install tensorflow-gpu==1.13.1
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
pip install stable-baselines3==1.1.0a11
pip install requests pygame==2.1.2 tqdm pillow requests gym matplotlib seaborn ipython box2d==2.3.10 scikit-learn
conda install cudatoolkit=10.0.130
conda install cudnn
pip install protobuf==3.20.0
pip install memory_profiler GitPython sacred==0.7.4 pymongo numpy
pip install -e human_aware_rl/overcooked_ai/.
pip install -e human_aware_rl/.
pip uninstall numpy
pip install numpy==1.18.0
sudo apt install libopenmpi-dev
pip install -e human_aware_rl/stable-baselines
pip install -e human_aware_rl/baselines
pip install gym==0.17.2