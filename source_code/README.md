#  ICML 8299 Souce Code

This repo contains the official implementation for GCQS.

## 1. Dependencies
Create conda environment.
```
conda create -n gcqs python=3.7.4
conda activate gcqs
```
Install PyTorch
```
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
```
Download [mujoco200](https://www.roboti.us/download.html). Then install pip requirements:
```
pip install -r requirements.txt
```

## 2. Code structure
The code structure is listed in below. Note that we provide 

implementation in PyTorch as well for the convenience of future research, though
they are not used in our paper.
```
Multi-goal Reinforcement Learning with Subgoals Generated from Relabeling
 └─run_gcqs.sh     (the script to run with a specific critic architecture)
 └─main.py    (the main file to run all code)
 └─robotics_plot.py    (plotting utils to make figures in the paper)
 └─src
    └─model.py (include different critic architectures, and the actor architecture)
    └─agent
       └─base.py  (base class for goal-conditioned agent)
       └─her.py   (DDPG+HER agent)
       └─ddpg.py  (DDPG agent)
       └─actionablemodel.py  (AM agent)
       └─mher.py  (MHER agent)
       └─gcsl.py  (GCSL agent)
       └─wgcsl.py (WGCSL agent)
       └─gofar.py (GoFar agent)
       └─dwsl.py  (DWSL agent)
       └─smore.py  (SMORE agent)
       └─gcqs.py (GCHR agent)
 ```

## 2. To reproduce results in the paper
```
./run_gchr.sh
```

## 3. Logs and checkpoints of trained models
You should first train to generate result files for each algorithm, which are located in the results directory and in pt format. 
Then you can use robotics_plot.py to reproduce 
the main figures about the results on the 8 robot tasks.

