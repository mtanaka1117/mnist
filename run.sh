docker container run --rm --gpus "device=0" -v /home/srv-admin/mnist:/mnist -e LOCAL_UID=$(id -u $USER) -e LOCAL_GID=$(id -g $USER) -it mnist:test bash
