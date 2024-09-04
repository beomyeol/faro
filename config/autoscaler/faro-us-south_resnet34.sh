#/bin/bash

PROCESSING_TIME=180
TARGET_LATENCY=`expr 4 \* ${PROCESSING_TIME}`

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

sed -i -z -r 's/target_metric: [0-9]+\n/target_metric: '${TARGET_LATENCY}'\n/g' $SCRIPT_DIR/autoscale_config.yaml
sed -i -z -r 's/target_metric: [0-9]+\n/target_metric: '${TARGET_LATENCY}'\n/g' $SCRIPT_DIR/pred_autoscale_config.yaml
find $SCRIPT_DIR -name '*.yaml' | xargs sed -i -z -r 's/processing_time: [0-9\.]+\n/processing_time: '${PROCESSING_TIME}'\n/g'
