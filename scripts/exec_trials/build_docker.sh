# !/bin/bash
set -e

HOSTNAME=$(id -gn)
HOST_ID=$(id -u)
GROUP_ID=$(id -g)
GROUP_NAME=$(id -gn)



echo "HOSTNAME=${HOSTNAME}"
echo "HOST_ID=${HOST_ID}"
echo "GROUP_ID=${GROUP_ID}"
echo "GROUP_NAME=${GROUP_NAME}"
# Clone it out here first so we 100% have a specific version
docker buildx build --platform linux/amd64 --build-arg="HOST_USER_NAME=${HOSTNAME}" \
    --build-arg="HOST_USER_ID=${HOST_ID}" \
    --build-arg="HOST_GROUP_ID=${GROUP_ID}" \
    --build-arg="HOST_GROUP_NAME=${GROUP_NAME}" \
    --force-rm \
    -f scripts/exec_trials/Dockerfile.exec \
    -t codeorm:exec-trials .