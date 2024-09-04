#!/usr/bin/env bash
# pass exit code in using pipe
set -o pipefail

function exit_if_fail() {
  if [ $? -ne 0 ]; then
    exit $?
  fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && pwd)"
DEPLOY_SCRIPT_PATH="$SCRIPT_DIR/../deploy.py"

MIN_QPS=5
MAX_QPS=50
WORKDIR="$SCRIPT_DIR/../../../misc/infaas/qps_${MIN_QPS}_${MAX_QPS}"
CONFIG="singleray"
USE_AUTOSCALE=0
TARGET_QPS=4

# Generate config
CLUSTER_IP=`kubectl get services -n k8s-ray | grep serve-cluster1 | awk '{print $3}'`
CONFIG_PATH="$WORKDIR/${CONFIG}_config.json"
CONTENT="{\n
\t\"http://${CLUSTER_IP}:8000/classifier\": 0.8,\n
\t\"http://${CLUSTER_IP}:8000/classifier1\": 0.05,\n
\t\"http://${CLUSTER_IP}:8000/classifier2\": 0.05,\n
\t\"http://${CLUSTER_IP}:8000/classifier3\": 0.05,\n
\t\"http://${CLUSTER_IP}:8000/classifier4\": 0.05\n
}"
echo -e $CONTENT > $CONFIG_PATH

# Deploy replicas
HEAD=`kubectl get pods -n k8s-ray | grep cluster1-ray-head | awk '{print $1}'`
NUM_REPLICAS_LIST=(11 1 1 1 1)
for i in ${!NUM_REPLICAS_LIST[@]};
do
  NUM_REPLICAS=${NUM_REPLICAS_LIST[$i]}
  if [ $i -eq 0 ]; then
    MIN_NUM_REPLICAS=2
  else
    MIN_NUM_REPLICAS=1
  fi
  if [ $USE_AUTOSCALE -eq 1 ]; then
    NUM_REPLICAS=${NUM_REPLICAS_LIST[0]}
    CMD="/home/ray/anaconda3/bin/python /home/ray/deploy.py"
    CMD+=" --autoscale"
    CMD+=" --autoscale-target-num=$TARGET_QPS"
    CMD+=" --autoscale-min-replicas=$MIN_NUM_REPLICAS"
    CMD+=" --autoscale-max-replicas=$NUM_REPLICAS"
    CMD+=" --serve-dedicated-cpu"
  else
    CMD="/home/ray/anaconda3/bin/python /home/ray/deploy.py"
    CMD+=" --num-replicas=$NUM_REPLICAS"
    CMD+=" --serve-dedicated-cpu"
  fi
  if [ $i -ne 0 ]; then
    CMD+=" --name=classifier$i"
  fi
  echo "[$HEAD] $CMD"
  kubectl cp -n k8s-ray $DEPLOY_SCRIPT_PATH $HEAD:/home/ray
  exit_if_fail
  kubectl exec -n k8s-ray --tty pod/$HEAD -- $CMD
  exit_if_fail
done

# Run assigner
ASSIGNER_PATH="${SCRIPT_DIR}/../trace_replayer/assigner.py"
OUT_PATH="$WORKDIR/${CONFIG}_04_06.json"
if [ -f $OUT_PATH ]; then
    python ${ASSIGNER_PATH} ${OUT_PATH} --config $CONFIG_PATH --out ${OUT_PATH} --reassign
else
    python ${ASSIGNER_PATH} ${WORKDIR}/twitter_04_06_norm.txt --config $CONFIG_PATH --out ${OUT_PATH}
fi
exit_if_fail

# Copy input
REPLAYER_POD="replayer"
REPLAYER_DIR="$SCRIPT_DIR/../trace_replayer"
kubectl cp -n k8s-ray ${OUT_PATH} ${REPLAYER_POD}:/go/input.json
exit_if_fail