NAME ?= vector-embedder
TAG ?= latest
CONTAINER ?= $(NAME):$(TAG)
REGISTRY ?= localhost
UPLOADREGISTRY ?= quay.io/validatedpatterns

##@ Pattern Must Gather Tasks

.PHONY: help
help: ## This help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^(\s|[a-zA-Z_0-9-])+:.*?##/ { printf "  \033[36m%-35s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

.PHONY: build
build: podman-build

.PHONY: podman-build
podman-build: ## build container
	podman build -t ${REGISTRY}/${CONTAINER} .

.PHONY: upload
upload: ## push container
	podman tag ${REGISTRY}/${CONTAINER} ${UPLOADREGISTRY}/${CONTAINER}
	podman push ${UPLOADREGISTRY}/${CONTAINER}

.PHONY: super-linter
super-linter: ## Runs super linter locally
	rm -rf .mypy_cache
	podman run -e RUN_LOCAL=true -e USE_FIND_ALGORITHM=true	\
					$(DISABLE_LINTERS) \
					-v $(PWD):/tmp/lint:rw,z \
					-w /tmp/lint \
					ghcr.io/super-linter/super-linter:slim-v8
