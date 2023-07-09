import os
import zipfile
from tqdm import tqdm

from ranepa_s3_wrapper.wrapper import MinioS3


def get_model_from_s3(access_key: str, secret_key: str, model_key: str, model_n=0):
    """[a function to download a model from ranepa s3]

    Arguments:
        model_key {str} -- [the name of the zip-file containing the model]
        access_key {str} -- [the access key for s3]
        secret_key {str} -- [the secret key for s3]
    """
    CONFIG = {
        "host": "10.8.0.2",
        "port": "9000",
        "access_key": access_key,
        "secret_key": secret_key,
    }
    if not model_key.endswith(".zip"):
        raise Exception("a model should be in zip-format")
    if access_key not in ("api", "admin"):
        raise Exception("`api` and `admin` are the only possible keys")
    instance_s3 = MinioS3(CONFIG)

    if not instance_s3.check_bucket("models"):
        print(instance_s3.check_bucket("models"))
        raise Exception("bucket `models` doesn't exist")

    if model_key not in instance_s3.get_list("models"):
        raise Exception(f"the model {model_key} is not in the bucket `models`")
    print("downloading the model", model_key)
    filename = os.path.basename(model_key)
    with open(filename, "wb") as f:
        instance_s3.s3_client.download_fileobj("models", model_key, f)
    target_dir = "/app/" + filename.replace(".zip", "")

    print("unpacking the model", filename, target_dir)
    top_folders = []
    with zipfile.ZipFile(filename, "r") as zip_ref:
        top = list({item.split("/")[0] for item in zip_ref.namelist()})
        top = ["/app/" + t for t in top]
        top_folders += top
        zip_ref.extractall("/app/")
    os.remove(filename)
    if top_folders and len(top_folders) == 1 and os.path.exists(top_folders[0]):
        target_dir = top_folders[0]
    if os.path.exists(target_dir):
        os.rename(target_dir, f"/app/model{model_n}")


def main():
    """[download the model for building a container image]

    Raises:
        Exception: [description]
    """
    access_key = os.getenv("S3_ACCESS_KEY")
    secret_key = os.getenv("S3_SECRET_KEY")

    if not os.path.exists("/app/s3_models.txt"):
        raise Exception("/app/s3_models.txt doesn't exist")
    with open("/app/s3_models.txt") as f:
        models = f.readlines()
    models = [line.strip() for line in models]
    print(f"List of models: {models}")

    for model_n, model in enumerate(models):
        if model.startswith("#"):
            continue
        get_model_from_s3(access_key, secret_key, model, model_n)
        print(f"Model {model} is downloaded.")


if __name__ == "__main__":
    main()
