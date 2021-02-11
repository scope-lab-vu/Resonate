# ReSonAte: A Runtime Risk Assessment Framework for  Autonomous Systems

ReSonAte uses the information gathered by hazard analysis and assurance cases to build [Bow-Tie Diagrams](https://www.cgerisk.com/knowledgebase/The_bowtie_method) to model hazard propagation paths and capture their relationships with the state of the system and environment. These Bow-tie diagrams are used to synthesize graphical models that are then used at runtime along with the information gathered from prior incidents about the possible environmental hazards and the hypothesis from failure diagnosers and system runtime monitors to estimate the hazard rates at runtime. These hazard rates are then used to determine the likelihood of unsafe system-level consequences captured in the bow-tie diagram. 

This repo has the steps to run the ReSonAte framework in the CARLA simulator. We demonstrate the utility of the ReSonAte framework for NHTSA-inspired pre-crash scenarios that are a part of the CARLA Autonomous Driving Challange  https://carlachallenge.org/challenge/nhtsa/

<p align="center">
  <img src="https://github.com/Shreyasramakrishna90/AV-Runtime-Risk/blob/main/videos/readme2.gif" />
</p>

Resonate estimated collision rate as the Autonomous Vehicle navigated through a nominal CARLA scene with weather(cloud = 0.0, precipitation = 0.0, deposits = 0.0). The scene gets adverse with high brightness. The [B-VAE assurance monitor](https://ieeexplore-ieee-org.proxy.library.vanderbilt.edu/stamp/stamp.jsp?arnumber=9283847) detects the increase in brightness and its martingale increases. The Blur detectors and Occlusion detector (on left) remain low throughout. 

Additional videos of other scenes can be found in the [Video Folder](https://github.com/Shreyasramakrishna90/AV-Runtime-Risk/blob/main/videos/)

## Installing the CARLA Autonomous Driving setup

The ReSonAte framework implementation in CARLA is built on top of the CARLA AD example from [Learning By Cheating](https://github.com/bradyz/2020_CARLA_challenge). To run the CARLA AD setup with varied weather patterns and evaluate the AV's collision rate, clone this repo.

```bash
git clone https://github.com/Shreyasramakrishna90/Resonate-Dynamic-Risk
```
All python packages used are specified in `carla_project/requirements.txt`. This code uses CARLA 0.9.9.

You will also need to install CARLA 0.9.9, along with the additional maps.
See [link](https://github.com/carla-simulator/carla/releases/tag/0.9.9) for more instructions.

```
To run this setup first create a virtual environment with python 3.7
conda create -n py37 python=3.7
conda activate py37
cd ${CARLA_ROOT/Carla-AD}  # Change ${CARLA_ROOT} for your CARLA root folder
pip3 install -r PythonAPI/carla/requirements.txt
```

##### Download pre-trained Learning Enabled Component
The preitrained LEC is got from [Learning By Cheating](https://github.com/bradyz/2020_CARLA_challenge). Download epoch=24.ckpt file from [Wandb.ai](https://wandb.ai/bradyz/2020_carla_challenge_lbc/runs/command_coefficient=0.01_sample_by=even_stage2/files?workspace=user-)

Navigate to the following location to download the pre-trained LEC weights. 

bradyz--> Projects --> 2020_carla_challenge_lbc --> Runs --> command_coefficient=0.01_sample_by=even_stage2 --> Files

Alternately, the weights can be downloaded from [here]()

Save the epoch=24.ckpt file to resonate-carla/carla_project folder. 


#### Download the pre-trained B-VAE assurance monitor weights
The B-VAE assurance monitor is designed to detect adversities in the environment such as excessive brightness. 

Download the weights from https://vanderbilt365-my.sharepoint.com/:u:/g/personal/shreyas_ramakrishna_vanderbilt_edu/EY5JCqsI65JEtvwMelR6OZwBPfho7FNtBOG5pDWAMXh1ng?e=7hR7pa

Unzip and save the weights to resonate-carla/leaderboard/team_code/detector_code


# Running the Carla setup with ReSonAte Collision rate estimation

```bash
./CarlaUE4.sh -quality-level=Epic -world-port=2000 -resx=800 -resy=600 -opengl
```

```bash
export CARLA_ROOT=/home/scope/Carla/CARLA_0.9.9      # change to where you installed CARLA
export PORT=2000                                            # change to port that CARLA is running on
export ROUTES=leaderboard/data/routes/route_19.xml          # change to any route generated
export TEAM_AGENT=image_agent.py                            # no need to change
export TEAM_CONFIG=model.ckpt                               # change path to checkpoint
export HAS_DISPLAY=1                                        # set to 0 if you don't want a debug window

./run_agent.sh
```

# Acknowledgement 

This work was supported by the DARPA Assured Autonomy project and Air Force Research Laboratory. The views presented in this paper are those of the authors and do not reflect the opinion or endorsement of DARPA or ARFL.

# License

This repo is released under the MIT License (please refer to the LICENSE file for details). PythonAPI for the CARLA setup was borrowed from the [CARLA Leaderboard challenge repo](https://leaderboard.carla.org/), which is under MIT license. Also, several python scripts including the LEC driving agent was borrowed from dotchen's [Learning by cheating repo](https://github.com/dotchen/LearningByCheating) which is released under MIT license.


