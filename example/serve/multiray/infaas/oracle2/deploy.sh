#!/usr/bin/env bash
# pass exit code in using pipe
set -o pipefail

function exit_if_fail() {
  if [ $? -ne 0 ]; then
    exit $?
  fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && pwd)"
DEPLOY_SCRIPT_PATH="$SCRIPT_DIR/../../deploy.py"

MIN_QPS=5
MAX_QPS=50
WORKDIR="$SCRIPT_DIR/../../../../misc/infaas/qps_${MIN_QPS}_${MAX_QPS}"
CONFIG="multiray_oracle2"
USE_AUTOSCALE=0
TARGET_QPS=4

# Generate config
CLUSTER_IP_LIST=`kubectl get services -n k8s-ray | awk 'NR>1 {print $1, $3}'`
CONFIG_PATH="$WORKDIR/${CONFIG}_config.json"
CONTENT="{"
FIRST=0
while read -r CLUSTER_IP
do
    array=($CLUSTER_IP)
    if [ $FIRST -eq 0 ]; then
        FIRST=1
        PREFIX="\n"
    else
        PREFIX=",\n"
    fi
    if [[ ${array[0]} == *"cluster1"* ]]; then
        CONTENT+="$PREFIX\t\"http://${array[1]}:8000/classifier\": 0.8"
    else
        CONTENT+="$PREFIX\t\"http://${array[1]}:8000/classifier\": 0.05"
    fi
done <<<$CLUSTER_IP_LIST
CONTENT+="\n}"
echo -e $CONTENT > $CONFIG_PATH

# Deploy replicas
HEADS=`kubectl get pods -n k8s-ray | grep ray-head | awk '{print $1}'`
for HEAD in $HEADS;
do
    MAX_NUM_REPLICAS=11
    if [[ $HEAD == *"cluster1"* ]]; then
        MIN_NUM_REPLICAS=2
        NUM_REPLICAS=${MAX_NUM_REPLICAS}
        EXTRA_FLAGS="--serve-dedicated-cpu"
    else
        MIN_NUM_REPLICAS=1
        NUM_REPLICAS=1
        EXTRA_FLAGS=""
    fi
    if [ $USE_AUTOSCALE -eq 1 ]; then
        CMD="/home/ray/anaconda3/bin/python /home/ray/deploy.py"
        CMD+=" --autoscale"
        CMD+=" --autoscale-target-num=$TARGET_QPS"
        CMD+=" --autoscale-min-replicas=$MIN_NUM_REPLICAS"
        CMD+=" --autoscale-max-replicas=$MAX_NUM_REPLICAS"
        CMD+=" $EXTRA_FLAGS"
    else
        CMD="/home/ray/anaconda3/bin/python /home/ray/deploy.py"
        CMD+=" --num-replicas=$NUM_REPLICAS"
        CMD+=" $EXTRA_FLAGS"
    fi
    echo "[$HEAD] $CMD"
    kubectl cp -n k8s-ray $DEPLOY_SCRIPT_PATH $HEAD:/home/ray
    exit_if_fail
    kubectl exec -n k8s-ray --tty pod/$HEAD -- $CMD
    exit_if_fail
done

# Run assigner
ASSIGNER_PATH="${SCRIPT_DIR}/../../trace_replayer/assigner.py"
OUT_PATH="$WORKDIR/${CONFIG}_04_06.json"
if [ -f $OUT_PATH ]; then
    python ${ASSIGNER_PATH} ${OUT_PATH} --config $CONFIG_PATH --out ${OUT_PATH} --reassign
else
    python ${ASSIGNER_PATH} ${WORKDIR}/twitter_04_06_norm.txt --config $CONFIG_PATH --out ${OUT_PATH}
fi
exit_if_fail

# Copy input
REPLAYER_POD="replayer"
REPLAYER_DIR="$SCRIPT_DIR/../../trace_replayer"
kubectl cp -n k8s-ray ${OUT_PATH} ${REPLAYER_POD}:/go/input.json
exit_if_fail