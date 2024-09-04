FROM beomyeol/ray-pytorch:2.0.0_1.12.1

ARG SERVE_DIR=/home/ray/anaconda3/lib/python3.7/site-packages/ray/serve
ARG AUTOSCALER_DIR=/home/ray/anaconda3/lib/python3.7/site-packages/ray/autoscaler
ARG PRIVATE_DIR=/home/ray/anaconda3/lib/python3.7/site-packages/ray/_private

COPY ray/serve/ ${SERVE_DIR}/
COPY ray/autoscaler/ ${AUTOSCALER_DIR}/
COPY ray/_private/ ${PRIVATE_DIR}/
