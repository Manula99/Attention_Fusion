import modal

image = (modal.Image.debian_slim()
         .apt_install("git")
         .pip_install("torch", "monai", "accelerate", "tqdm", "datasets", "transformers")
         .run_commands("git clone https://github.com/Manula99/Attention_Fusion.git /root/Attention_Fusion")
         .workdir("/root/Attention_Fusion"))

app = modal.App(image=image)

@app.function(gpu="A100-80GB:3")
def run():
    from ddp_computing import distributed_trainer
    from models import U_Transformer
    import sys, os
    sys.path.insert(0, "/root/Attention_Fusion")
    os.environ['MONAI_DATA_DIRECTORY'] = '/content/monai_data'

    model = U_Transformer(4, 3)
    distributed_trainer(model)