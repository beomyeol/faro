# A House United Within Itself: SLO-Awareness for On-Premises Containerized ML Inference Clusters via Faro (Eurosys 25)

## Simulation

We provide [Dockerfile](Dockerfile_sim) and prebuilt Docker image: `beomyeol/faro-operator-1600:sim`.

Please see [src/simulation/README.md](src/simulation/README.md) to see how to run simulation.

## Cluster experiments

Tested on the IBM Cloud VPC Kubernetes Cluster (two cx2-32x64 VM instances or 32 cx2-4x8 VM instances).

Currently, all components including Ray clusters should run under the namespace `k8s-ray`.

### Prerequisites
- kubectl (https://kubernetes.io/docs/reference/kubectl/)
- kustomize (https://kustomize.io/)
- docker (https://www.docker.com/)

### Prepare: build and push a controller docker image, and download `kustomize`.
```sh
make build && make push && make kustomize
```

### Setup: create namespace, install crd, etc.
```sh
make install
kubectl create -f example/ray/cluster_crd.yaml
```

### Deploy a Ray controller
```sh
kubectl create -f example/ray/operator.yaml
```

### Deploy trace replayer
```sh
kubectl create -f example/serve/trace_replayer/replayer.yaml
```

### Build autoscaler docker image and push (skip this if the image is already built)
```sh
make build
make push
```

### Deploy autoscaler
```sh
make deploy
```

### Set a quota for namespace to limit resources
```sh
kubectl apply -f experiments/top9_twitter_1_1600_avgproc_min_int5m_reduced_6hr_augmented/quota/30_workers.yaml
kubectl apply -f experiments/top9_twitter_1_1600_avgproc_min_int5m_reduced_6hr_augmented/quota/32_workers.yaml
kubectl apply -f experiments/top9_twitter_1_1600_avgproc_min_int5m_reduced_6hr_augmented/quota/36_workers.yaml
kubectl apply -f experiments/top9_twitter_1_1600_avgproc_min_int5m_reduced_6hr_augmented/quota/40_workers.yaml
kubectl apply -f experiments/top9_twitter_1_1600_avgproc_min_int5m_reduced_6hr_augmented/quota/44_workers.yaml
```

### Deploy Ray clusters
```sh
kubectl kustomize experiments/top9_twitter_1_1600_avgproc_min_int5m_reduced_6hr_augmented/k8s | kubectl create -f -
```

### Deploy inference jobs for each cluster and copy input
```sh
python scripts/deploy_jobs.py experiments/top9_twitter_1_1600_avgproc_min_int5m_reduced_6hr_augmented/input.json
```

### Copy config and launch terminal for autoscaler
```sh
python scripts/deploy_autoscaler_config.py config/autoscaler/aiad.yaml
python scripts/deploy_autoscaler_config.py config/autoscaler/oneshot.yaml
python scripts/deploy_autoscaler_config.py config/autoscaler/mark.yaml
python scripts/deploy_autoscaler_config.py config/autoscaler/32_workers/faro_sum.yaml
python scripts/deploy_autoscaler_config.py config/autoscaler/36_workers/faro_sum.yaml
python scripts/deploy_autoscaler_config.py config/autoscaler/40_workers/faro_sum.yaml
kubectl exec -n k8s-ray deployment/faro-operator -it -- /bin/bash
```

### Launch autoscaler (inside `deployment/faro-operator`)
```sh
python src/autoscaler.py config.yaml |& tee run.log
```

### Launch replayer
```sh
kubectl exec -n k8s-ray pod/replayer -it -- /bin/bash
sleep 230 && ./loadgen -i input.json -img image.jpg -max_idle_conn 100 -interval_type poisson -seed 42 -unit_time 60 --max_trials 2
```

### Get logs and parse them
```bash
python scripts/get_serve_logs.py results/faro-us-south/mixed/top9_twitter_1_1600_avgproc_min_int5m_reduced_6hr_augmented/32_cpus/aiad --with-autoscaler [--with-worker]
python scripts/parse_serve_logs.py results/faro-us-south/mixed/top9_twitter_1_1600_avgproc_min_int5m_reduced_6hr_augmented/32_cpus/aiad
```

### Delete Ray clusters
```sh
kubectl kustomize experiments/top9_twitter_1_1600_avgproc_min_int5m_reduced_6hr_augmented/k8s | kubectl delete -f -
```

Repeat from [changing quota](#set-a-quota-for-namespace-to-limit-resources) to [parsing logs](#get-logs-and-parse-them) while changing policies (`AIAD`, `Faro-Sum`, etc.)

### Generate stats from the parsed logs
```sh
python -m scripts.simulation.run_suite --max-workers=8 $(pwd)/results/faro-us-south/mixed/top9_twitter_1_1600_avgproc_min_int5m_reduced_6hr_augmented/32_cpus --stats --unit_time=60 --utility=latency
```

This will create `latency_stats.pkl` that is used for [generating plots](#generate-plots)

### Undeploy all components
```sh
make undeploy
kubectl kustomize experiments/top9_twitter_1_1600_avgproc_min_int5m_reduced_6hr_augmented/k8s | kubectl delete -f -
kubectl delete -f example/serve/trace_replayer/replayer.yaml
kubectl delete -f example/ray/operator.yaml
```

## Generate plots

See [scripts/plots/README.md](scripts/plots/README.md)


## Miscellaneous

### Setup IBM Cloud Registry

[Reference](https://cloud.ibm.com/registry/start)
```sh
ibmcloud plugin install container-registry -r 'IBM Cloud'
ibmcloud cr region-set us-south
ibmcloud cr login
```

### Push images to IBM Cloud CR
```sh
docker push us.icr.io/faro/faro-operator-1600:234b913`
```

### Copy secret for k8s-ray namespaces and patch default account to use image pull secrets
```sh
kubectl get secret all-icr-io -n default -o yaml | sed 's/default/k8s-ray/g' | kubectl create -n k8s-ray -f - 
kubectl patch -n k8s-ray serviceaccount/default -p '{"imagePullSecrets":[{"name": "all-icr-io"}]}'
```

### Cilantro

Use a Cilantro fork that supports ResNet34 and uses the same traces that Faro uses: https://github.com/beomyeol/cilantro and https://github.com/beomyeol/cilantro-workloads

Create docker images for each by running
```sh
docker build -t beomyeol/cilantro:latest . && docker push beomyeol/cilantro:latest
docker build -t beomyeol/cray-workloads:latest . && docker push beomyeol/cray-workloads:latest
```

For evaluation, we used `beomyeol/cilantro:ef28039` and `beomyeol/cray-workloads:2a1d9e1`.

#### Run cilantro and get logs
```sh
./starters/launch_cilantro_driver_kind.sh ~/.kube/config utilwelflearn
```
Wait for 6 hours to finish experiments. Then fetch results by using the following command.
```sh
./starters/fetch_results.sh
```
This will create `workdirs_eks`. Provide this for generating the cilantro timeline plot. See [scripts/plots/README.md](scripts/plots/README.md).


## License

University of Illinois/NCSA Open Source License