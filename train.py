import os
import pickle
import time
import torch
import torch.nn as nn
from monai.apps import DecathlonDataset
from monai.data import DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose

from models import U_Transformer_VAE_DS
from data import train_transform, val_transform
from utils import compute_KLD, kl_weight


def save_metrics(suffix, epoch_loss_values, metric_values, 
                 metric_values_tc, metric_values_wt, 
                 metric_values_et, epoch_val_losses,
                epoch_dice_loss_values):
    metrics_dir = "/mnt/monai-data/metrics"
    metrics_data = {
        'epoch_loss_values': epoch_loss_values,
        'epoch_val_losses': epoch_val_losses,
        'epoch_dice_loss_values': epoch_dice_loss_values,
        'metric_values': metric_values,
        'metric_values_tc': metric_values_tc,
        'metric_values_wt': metric_values_wt,
        'metric_values_et': metric_values_et,
    }
    
    # Define the filename to save the pickled data
    filename = os.path.join(metrics_dir, f"training_metrics_{suffix}.pkl")
    
    # Save the dictionary to a file using pickle
    with open(filename, 'wb') as f:
        pickle.dump(metrics_data, f)
    
    print(f"Training metrics saved to {filename}")


def inference(input, model, ds=False):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(224, 224, 144) if not ds else (64, 64, 32),
            sw_batch_size=2,
            predictor=model,
            overlap=0.5,
        )

    VAL_AMP = True
    if VAL_AMP:
        with torch.autocast("cuda"):
            return _compute(input)
    else:
        return _compute(input)


def train_model(model, model_name, max_epochs=100, val_interval=1, batch_size=2, model_weight_path=None, 
    root_dir='/mnt/monai-data/data', best_metric=-1, accum_iter = 5, has_kl=False, suffix='_r1'):
    os.environ['MONAI_DATA_DIRECTORY'] = root_dir
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    if directory is not None:
        os.makedirs(directory, exist_ok=True)

    train_ds = DecathlonDataset(
        root_dir=root_dir,
        task="Task01_BrainTumour",
        transform=train_transform,
        section="training",
        download=False,
        cache_rate=0.0,
        num_workers=0,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_ds = DecathlonDataset(
        root_dir=root_dir,
        task="Task01_BrainTumour",
        transform=val_transform,
        section="validation",
        download=False,
        cache_rate=0.0,
        num_workers=0,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda:0")
    model.to_device(device)

    # Load pretrained weights if available
    
    if model_weight_path is not None and os.path.exists(model_weight_path):
        state_dict = torch.load(model_path, weights_only=True)
        model.load_state_dict(state_dict)

    loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
    ce = nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    scaler = torch.GradScaler("cuda")
    torch.backends.cudnn.benchmark = True

    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    best_metric_epoch = -1
    best_metrics_epochs_and_time = [[], [], []]
    epoch_val_losses = []
    epoch_loss_values = []
    epoch_dice_loss_values = []
    metric_values = []
    metric_values_tc = []
    metric_values_wt = []
    metric_values_et = []

    total_start = time.time()
    

    for epoch in range(max_epochs):
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        epoch_dice_loss = 0
        step = 0
        for batch_data in train_loader:
            step_start = time.time()

            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )

            with torch.autocast("cuda"):
                if has_kl:
                    outputs, level_params = model(inputs)
                    kld = 0
                    sum_inter_KLD = 0.0
                    sum_prior_KLD = 0.0
                    for i in range(3):
                        means = level_params[i]['mu']
                        log_var = level_params[i]['log_var']
                        inter_KLD, prior_KLD = compute_KLD(means, log_var)
                        sum_inter_KLD += inter_KLD
                        sum_prior_KLD += prior_KLD
                        
                    kld = 1 / 4 * sum_inter_KLD + 1 / 4 * sum_prior_KLD
                else:
                    outputs = model(inputs)
                
                dice_loss = loss_function(outputs, labels)
                ce_loss = ce(outputs, labels)
                loss = (dice_loss + ce_loss) / accum_iter

                if has_kl:
                    loss += kl_weight(epoch) * kld

                loss /= accum_iter

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * accum_iter
            epoch_dice_loss += dice_loss.item()

            if ((step + 1) % accum_iter == 0) or (step + 1 == len(train_loader)):
              scaler.step(optimizer)
              scaler.update()
              optimizer.zero_grad()

              print(
                  f"{step}/{len(train_ds) // train_loader.batch_size}"
                  f", train_loss: {(loss.item() * accum_iter):.4f}"
                  f", step time: {(time.time() - step_start):.4f}"
              )

            step += 1

        lr_scheduler.step()
        epoch_loss /= step
        epoch_dice_loss /= step
        epoch_loss_values.append(epoch_loss)
        epoch_dice_loss_values.append(epoch_dice_loss)
        
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    val_outputs_inf = inference(val_inputs, model)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs_inf)]
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    dice_metric_batch(y_pred=val_outputs, y=val_labels)
                    val_losses.append(loss_function(val_outputs_inf, val_labels).item())
                    
                epoch_val_losses.append(sum(val_losses) / len(val_losses))
                metric = dice_metric.aggregate().item()
                metric_values.append(metric)
                metric_batch = dice_metric_batch.aggregate()
                metric_tc = metric_batch[0].item()
                metric_values_tc.append(metric_tc)
                metric_wt = metric_batch[1].item()
                metric_values_wt.append(metric_wt)
                metric_et = metric_batch[2].item()
                metric_values_et.append(metric_et)
                dice_metric.reset()
                dice_metric_batch.reset()
                save_metrics(f"{model_name}_{epoch}_{suffix}", epoch_loss_values, 
                             metric_values, metric_values_tc, metric_values_wt, metric_values_et,
                                epoch_val_losses, epoch_dice_loss_values)

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    best_metrics_epochs_and_time[0].append(best_metric)
                    best_metrics_epochs_and_time[1].append(best_metric_epoch)
                    best_metrics_epochs_and_time[2].append(time.time() - total_start)
                    torch.save(
                        model.state_dict(),
                        os.path.join(model_weight_dir, f"best_metric_{model_name}.pth"),
                    )
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                    f"\nbest mean dice: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
                
        torch.save(
                    model.state_dict(),
                    os.path.join(model_weight_dir, f"latest_metric_model_{model_name}.pth"),
                )


if __name__ == "__main__":
    train_model()