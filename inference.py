from models import U_Transformer_VAE_DS
from train import inference
from monai.transforms import LoadImage, Compose, Activations, AsDiscrete
from utils import show_tensor, visualize_mosaic
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = U_Transformer_VAE_DS(1, 3, batch_size=10).to(device)
state_dict = torch.load('best_metric_model_vae_kl_anneal.pth', weights_only=True, map_location=device)
model.load_state_dict(state_dict)

def adjust_tumor_classes(tensor):
    result = []
    # merge label 2 and label 3 to construct TC
    result.append(torch.logical_or(tensor == 2, tensor == 3))
    # merge labels 1, 2 and 3 to construct WT
    result.append(torch.logical_or(torch.logical_or(tensor == 2, tensor == 3), tensor == 1))
    # label 2 is ET
    result.append(tensor == 2)
    return torch.stack(result, axis=0).float()

post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

if __name__ == '__main__':
    loader = LoadImage(image_only=True)
    image_data = loader("./monai_data/BRATS_001.nii.gz")
    image_data = image_data.unsqueeze(0).to(device)
    image_data = image_data.permute(0, 4, 1, 2, 3)
    image_data = image_data[:, :, 16:, 16:, :-11]

    labels = loader("./monai_data/BRATS_001_labels.nii.gz")
    labels = labels.to(device)
    labels = adjust_tumor_classes(labels)[:, 16:, 16:, :-11]
    print(labels.size())
    model.eval()
    seg = inference(image_data, model)
    print(seg.size())
    seg = post_trans(seg)
    visualize_mosaic(image_data.squeeze(0), seg[0], labels, save_path='seg_mosaic.png')
    #show_tensor(seg.squeeze(0)[0, :, :, 72])