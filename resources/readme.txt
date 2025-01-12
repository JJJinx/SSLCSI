1.run command "docker build -f ./docker/Dockerfile --rm -t mmselfsup:torch1.11.0-cuda11.3-cudnn8 ."
3.run command "docker run -v $PWD:/workspace --gpus all -it mmselfsupcsi /bin/bash"
4.run "cd mmselfsup"
5.run "pip install -v -e ."

build from scratch on anaconda env:
1.pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
2.pip install -U openmim
3.mim install mmcv-full
4.cd mmselfsup
5.pip install -v -e .
other dependent please install according to the requirements.txt
