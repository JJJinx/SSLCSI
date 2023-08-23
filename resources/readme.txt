1.download the Dockerfile.txt and requirements.txt and code
2.run command "docker build -t mmselfsupcsi ." on the terminal under the same path of Dockerfile
3.run command "docker run -v $PWD:/workspace -it mmselfsupcsi /bin/bash"
4.run "cd mmselfsup"
5.run "pip install -v -e ."