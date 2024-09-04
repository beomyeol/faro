FROM beomyeol/custom-ray-pytorch:dd846e9
RUN conda install python -y
RUN pip install kubernetes \
    darts==0.22.0 \
    scikit-learn==1.0.2 \
    scikit-optimize==0.9.0 \
    pandas==1.3.5 \
    pytorch-lightning==1.9.4
RUN sudo apt-get install -y tmux
ADD models /home/ray/models
ADD src /home/ray/src
ARG SERVE_DIR=/home/ray/anaconda3/lib/python3.7/site-packages/ray/serve
COPY ray/serve/ ${SERVE_DIR}/
CMD trap : TERM INT; sleep infinity & wait;
