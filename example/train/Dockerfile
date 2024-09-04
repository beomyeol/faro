FROM beomyeol/ray-pytorch:2.1.0_1.13.0

RUN sudo apt update && sudo apt install -y build-essential
RUN HOROVOD_WITH_GLOO=1 $ANACONDA_BIN/pip install horovod[ray]==0.26.1 --no-cache-dir
RUN sudo apt remove -y build-essential && sudo apt autoremove -y && sudo apt clean
