docker run --rm -it \
  --shm-size=10.24gb \
  -v "$(pwd)":/piper \
  piper \
  bash