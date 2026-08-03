import modal

MONAI_DATA_DIR = "/content/monai_data"

def download_brain_tumour_data():
    """Runs at image-build time to fetch and extract the dataset into the image."""
    from monai.apps import download_and_extract
    import os

    os.makedirs(MONAI_DATA_DIR, exist_ok=True)

    resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
    md5 = "240a19d752f0d9e9101544901065d872"
    compressed_file = os.path.join(MONAI_DATA_DIR, "Task01_BrainTumour.tar")

    download_and_extract(resource, compressed_file, MONAI_DATA_DIR, md5)

image = (modal.Image.debian_slim()
         .apt_install("git", "curl", "unzip", "wget")
         .pip_install("torch", "monai", "accelerate", "tqdm", "datasets", "transformers")
         .run_commands("git clone https://github.com/Manula99/Attention_Fusion.git /root/Attention_Fusion")
         .workdir("/root/Attention_Fusion")
         .run_commands(
       f"mkdir -p {MONAI_DATA_DIR}",
       f"curl -L https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar -o {MONAI_DATA_DIR}/Task01_BrainTumour.tar",
       f"tar -xf {MONAI_DATA_DIR}/Task01_BrainTumour.tar -C {MONAI_DATA_DIR}",)
       )

app = modal.App(image=image)

@app.function(gpu="A100-80GB:3")
def run():
    from ddp_computing import distributed_trainer
    from models import U_Transformer
    import sys, os
    sys.path.insert(0, "/root/Attention_Fusion")
    os.environ['MONAI_DATA_DIRECTORY'] = MONAI_DATA_DIR

    model = U_Transformer(4, 3)
    distributed_trainer(model)