1.download the Dockerfile.txt and requirements.txt and code
2.run command "docker build -t mmselfsupcsi ." on the terminal under the same path of Dockerfile
3.run command "docker run -v $PWD:/workspace --gpus all -it mmselfsupcsi /bin/bash"
4.run "cd mmselfsup"
5.run "pip install -v -e ."

attention:
if "mim install mmcv-full" does not work, try "pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
"

in linux remember add the path to environment variable 
    export PYTHONPATH=/path/to/lib:/path/to/another/lib


build from scratch on anaconda env:
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
TODO