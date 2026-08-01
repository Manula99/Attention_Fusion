import modal
from ddp_computing import distributed_trainer
from models import U_Transformer
import os
os.environ['MONAI_DATA_DIRECTORY'] = '/content/monai_data'

image = modal.Image.debian_slim().pip_install("torch", "monai", "accelerate", "tqdm", "datasets", "transformers")
app = modal.App(image=image)

@app.function(gpu="A100-80GB")
def run():
    model = U_Transformer(4, 3)
    distributed_trainer(model)