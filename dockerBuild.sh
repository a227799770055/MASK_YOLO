
DOCKER_CONTAINER_NAME="${USER}_stylegan2"
DOCKER_IMAGE="stylegan2-ada"

echo "Image: $DOCKER_IMAGE"
echo "Volume Path (LOCAL): $PWD"
echo "Container Name: $DOCKER_CONTAINER_NAME"

docker run --gpus all  --name ${DOCKER_CONTAINER_NAME} -u $(id -u ${USER}):$(id -g ${USER}) -it -d --pid=host --ipc=host -v $PWD:/workspace $DOCKER_IMAGE bash
docker restart $DOCKER_CONTAINER_NAME

echo "Create account in container..."

echo "useradd -u $(id -u ${USER}) -U -m -s /bin/bash me" >> $PWD/useradd.sh
docker exec -it -u root $DOCKER_CONTAINER_NAME bash -c "useradd -u $(id -u ${USER}) -U -m -s /bin/bash me"
rm $PWD/useradd.sh

docker exec -it $DOCKER_CONTAINER_NAME bash -c "echo \"export PATH=\"\$PATH:/home/me/.local/bin\"\" >> /home/me/.bashrc"

echo "Create account in finished."
