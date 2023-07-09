export TMPDIR="$HOME/.local/share/containers/tmp/"
podman stop news_models
podman rm news_models
podman rmi news_models
source .env_s3
podman build \
  --build-arg S3_ACCESS_KEY=$S3_ACCESS_KEY \
  --build-arg S3_SECRET_KEY=$S3_SECRET_KEY \
  ./ \
  -t news_models
podman run -d \
  --env-file .env_s3 \
  -p 10.8.0.5:8985:8989 \
  --name news_models \
  -t news_models
