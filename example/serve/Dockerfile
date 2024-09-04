FROM rayproject/ray:2.0.0

ENV ANACONDA_BIN=/home/ray/anaconda3/bin

RUN $ANACONDA_BIN/conda install tqdm && $ANACONDA_BIN/conda install pytorch torchvision torchaudio cpuonly -c pytorch && $ANACONDA_BIN/conda clean -y --all
