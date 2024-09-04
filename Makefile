NAMESPACE = k8s-ray
IMG ?= beomyeol/faro-operator
JOB_IMG ?= beomyeol/k8s-ray-job:latest

SHELL = /usr/bin/env bash -o pipefail
.SHELLFLAGS = -ec
KUSTOMIZE = $(shell pwd)/bin/kustomize

.PHONY: kustomize
kustomize:
ifeq (,$(wildcard $(KUSTOMIZE)))
	mkdir -p bin
	cd bin && curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh"  | bash
endif

build:
	docker build -t ${IMG} . --no-cache

push:
	docker push ${IMG}

install: kustomize
	$(KUSTOMIZE) build config/crd | kubectl apply -f -

uninstall:
	$(KUSTOMIZE) build config/crd | kubectl delete -f -

deploy: install
	$(KUSTOMIZE) build config/deployment | kubectl apply -f -

undeploy:
	$(KUSTOMIZE) build config/deployment | kubectl delete -f -
# kubectl delete ValidatingWebhookConfiguration auto.kopf.dev

job-docker-build: ## Build docker image with the example job.
	docker build -t ${JOB_IMG} example/job

job-docker-push: job-docker-build ## Push docker image with the example job.
	docker push ${JOB_IMG}