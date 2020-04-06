DOCKER_NAME=deepdish

.PHONY: clean docker clean-docker all

all: docker

clean:
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -delete
	find . -name '*~' -delete

docker:
	docker build -t $(DOCKER_NAME) \
		-f Dockerfile \
		--build-arg USER_ID=`id -u` \
		--build-arg GROUP_ID=`id -g` \
		.

clean-docker:
	docker rmi $(DOCKER_NAME)
