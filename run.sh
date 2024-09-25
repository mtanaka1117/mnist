docker image build -t mnist:test .
docker container run --rm --gpus "device=0" \
-v $(pwd):/mnist \
-e LOCAL_UID=$(id -u $USER) -e LOCAL_GID=$(id -g $USER) \
-it mnist:test bash
