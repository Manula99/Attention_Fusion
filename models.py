import torch
from torch import nn
import torch.nn.functional as F
from monai.networks.blocks import PatchEmbeddingBlock


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, last_relu=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True) if last_relu else nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels_from_down_path, skip_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='trilinear',
                                  align_corners=True)
            # The DoubleConv should take in_channels_from_down_path (x1) + skip_channels (x2) as its input
            # After upsampling, x1 still has in_channels_from_down_path
            self.conv = DoubleConv(in_channels_from_down_path + skip_channels, out_channels, (in_channels_from_down_path + skip_channels) // 2)
        else:
            # If not bilinear, transposed conv halves channels of x1.
            self.up = nn.ConvTranspose3d(
                in_channels_from_down_path,
                in_channels_from_down_path // 2,
                kernel_size=2,
                stride=2,
            )
            # Then concatenate with skip_channels.
            self.conv = DoubleConv(in_channels_from_down_path // 2 + skip_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHWD
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]

        x1 = F.pad(
            x1, [diffX // 2, diffX - diffX // 2,
                 diffY // 2, diffY - diffY // 2,
                 diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, last_relu=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels, last_relu=last_relu),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, channel, spatial_dims, patch_size, hidden):
        super(MultiHeadSelfAttention, self).__init__()
        self.pe = PatchEmbeddingBlock(channel, spatial_dims, patch_size, hidden, 1)
        self.attn = torch.nn.MultiheadAttention(channel, 1, batch_first=True)

    def forward(self, x):
        b, c, h, w, d = x.size()
        x = self.pe(x)
        attn_output = self.attn(x, x, x, need_weights=False)[0]
        x = attn_output.transpose(-1, -2).reshape(b, c, h, w, d)
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, channelY, channelS, spat_dimS, spat_dimY, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.Sconv = nn.Sequential(
            nn.MaxPool3d(2), nn.Conv3d(channelS, channelS, kernel_size=1),
            nn.BatchNorm3d(channelS), nn.LeakyReLU(inplace=True))
        self.Yconv = nn.Sequential(
            nn.Conv3d(channelY, channelS, kernel_size=1),
            nn.BatchNorm3d(channelS), nn.LeakyReLU(inplace=True))
        self.conv = nn.Sequential(
            nn.Conv3d(channelY, channelS, kernel_size=1),
            nn.BatchNorm3d(channelS), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True))
        self.Yconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(channelY, channelY, kernel_size=3, padding=1),
            nn.Conv3d(channelY, channelS, kernel_size=1),
            nn.BatchNorm3d(channelS), nn.LeakyReLU(inplace=True))

        self.Spe = PatchEmbeddingBlock(channelS, spat_dimS, (4, 4, 4), channelY, num_heads)
        self.Ype = PatchEmbeddingBlock(channelY, spat_dimY, (2, 2, 2), channelY, num_heads)
        self.attn = torch.nn.MultiheadAttention(channelY, num_heads, batch_first=True)

    def forward(self, Y, S):
        Sb, Sc, Sh, Sw, Sd = S.size()
        Yb, Yc, Yh, Yw, Yd = Y.size()
        S1 = self.Spe(S)
        Y1 = self.Ype(Y)
        attn_output = self.attn(Y1, Y1, S1, need_weights=False)[0].permute(0, 2, 1).reshape(Yb, Yc, Yh // 2, Yw // 2, Yd // 2)
        Z = self.conv(attn_output)
        Z = Z * S
        Z = torch.cat([Z, Y2], dim=1)
        return Z


class MultiHeadIntraAttention(nn.Module):
    def __init__(self, channelY, spat_dimY, num_heads,
                 patch_size=(2, 2, 2), batch_size=1, scale_factor=2,
                 post=False):
        super(MultiHeadIntraAttention, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=True)

        self.embedding1 = PatchEmbeddingBlock(channelY, spat_dimY, patch_size, channelY, num_heads)
        self.embedding2 = PatchEmbeddingBlock(channelY, spat_dimY, patch_size, channelY, num_heads)
        self.embedding3 = PatchEmbeddingBlock(channelY, spat_dimY, patch_size, channelY, num_heads)
        self.embedding4 = PatchEmbeddingBlock(channelY, spat_dimY, patch_size, channelY, num_heads)

        self.attn1 = torch.nn.MultiheadAttention(channelY, num_heads, batch_first=True)
        self.attn2 = torch.nn.MultiheadAttention(channelY, num_heads, batch_first=True)
        self.attn3 = torch.nn.MultiheadAttention(channelY, num_heads, batch_first=True)
        self.reshape = lambda x: torch.reshape(x.permute(0, 2, 1),( x.size()[0], channelY, spat_dimY[0] // patch_size[0],
                                               spat_dimY[1] // patch_size[1], spat_dimY[2] // patch_size[2]))
        self.post = post
        if self.post:
          in_dim = (spat_dimY[0] // patch_size[0]) * (spat_dimY[1] // patch_size[1]) * (spat_dimY[2] // patch_size[2])
          self.fc = nn.Linear(in_dim, channelY)
          self.post_proc = nn.Sequential(nn.LeakyReLU(inplace=True), self.fc)

    def forward(self, x1, x2, x3, x4):

        x1 = self.embedding1(x1)
        x2 = self.embedding2(x2)
        x3 = self.embedding3(x3)
        x4 = self.embedding4(x4)

        attn_output = self.attn1(x2, x1, x1, need_weights=False)[0]
        attn_output = self.attn2(x3, attn_output, attn_output, need_weights=False)[0]
        attn_output = self.attn3(x4, attn_output, attn_output, need_weights=False)[0]
        if self.post:
          attn_output = self.post_proc(attn_output)

        attn_3d = self.reshape(attn_output)
        return self.upsample(attn_3d)


class TransformerUp(nn.Module):
    def __init__(self, Ychannels, Schannels, spat_dimS, spat_dimY, num_heads):
        super(TransformerUp, self).__init__()
        self.MHCA = MultiHeadCrossAttention(Ychannels, Schannels, spat_dimS, spat_dimY, num_heads)
        self.conv = nn.Sequential(
            nn.Conv3d(Ychannels,
                      Schannels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.BatchNorm3d(Schannels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(Schannels,
                      Schannels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.BatchNorm3d(Schannels),
            nn.LeakyReLU(inplace=True))

    def forward(self, Y, S):
        x = self.MHCA(Y, S)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, last_relu=False, channels=(64, 128, 256, 512)):
        super(Encoder, self).__init__()

        self.inc = DoubleConv(in_channels, channels[0], last_relu=True)
        self.down1 = Down(channels[0], channels[1], last_relu=True)
        self.down2 = Down(channels[1], channels[2], last_relu=True)
        self.down3 = Down(channels[2], channels[3], last_relu=last_relu)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        return x1, x2, x3, x4


class Encoder_VAE(nn.Module):
    def __init__(self, in_channels, last_relu=False, channels=(64, 128, 256, 512)):
        super(Encoder_VAE, self).__init__()

        self.inc = DoubleConv(in_channels, channels[0], last_relu=True)
        self.down1 = Down(channels[0], channels[1], last_relu=True)
        self.down2 = Down(channels[1], channels[2], last_relu=True)
        self.down3 = Down(channels[2], channels[3], last_relu=last_relu)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        return x1, x2, x3, x4


class GaussianSampler(nn.Module):
    def __init__(self, name='gaussian_sampler'):
        super(GaussianSampler, self).__init__()

    def forward(self, means, logvars, list_mod, choices, is_inference):
        eps = 1e-7
        device = means[list_mod[0]].device

        # Get prior parameters
        mu_prior = torch.zeros_like(means[list_mod[0]])
        log_prior = torch.zeros_like(means[list_mod[0]])

        # Convert choices to tensor if needed
        if not isinstance(choices, torch.Tensor):
            choices = torch.tensor(choices, device=device, dtype=torch.bool)

        # Compute precision-weighted means and precisions
        T_list = []
        mu_list = []

        for mod in list_mod:
            precision = 1.0 / (torch.exp(logvars[mod]) + eps)
            precision_weighted_mean = means[mod] / (torch.exp(logvars[mod]) + eps)

            T_list.append(precision)
            mu_list.append(precision_weighted_mean)

        # Apply choices mask (select available modalities)
        T = torch.stack(T_list, dim=0)
        mu = torch.stack(mu_list, dim=0)

        # Convert choices to appropriate shape for masking
        choices_expanded = choices.view(-1, 1, 1, 1) if len(T.shape) > 1 else choices.view(-1, 1)

        T = torch.cat([T, 1.0 + log_prior.unsqueeze(0)], dim=0)
        mu = torch.cat([mu, mu_prior.unsqueeze(0)], dim=0)

        # Compute posterior mean and variance
        T_sum = torch.sum(T, dim=0)
        posterior_means = torch.sum(mu, dim=0) / T_sum
        var = 1.0 / T_sum
        posterior_logvars = torch.log(var + eps)

        if is_inference:
            return posterior_means
        else:
            # Sample from posterior using reparameterization trick
            noise_sample = torch.randn_like(posterior_means)
            output = posterior_means + torch.exp(0.5 * posterior_logvars) * noise_sample
            return output


class U_Transformer(nn.Module):
    def __init__(self, in_channels, classes, bilinear=True):
        super(U_Transformer, self).__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.bilinear = bilinear

        self.encoders = nn.ModuleList([Encoder(in_channels, channels=(8, 16, 32, 64)) for _ in range(4)])
        self.cross_attn1 = MultiHeadIntraAttention(8, (224, 224, 144), 1)
        self.cross_attn2 = MultiHeadIntraAttention(16, (112, 112, 72), 1)
        self.cross_attn3 = MultiHeadIntraAttention(32, (56, 56, 36), 1)
        self.cross_attn4 = MultiHeadIntraAttention(64, (28, 28, 18), 1)
        self.outc = OutConv(64, classes)

        self.up1 = Up(in_channels_from_down_path=64, skip_channels=32, out_channels=32)
        self.up2 = Up(in_channels_from_down_path=32, skip_channels=16, out_channels=16)
        self.up3 = Up(in_channels_from_down_path=16, skip_channels=8, out_channels=8)

    def forward(self, x):
        encodings = [self.encoders[i](x[:, i : i+1, ...]) for i in range(4)]
        # Use the first modality for now
        fusion1 = self.cross_attn1(encodings[0][0], encodings[1][0], encodings[2][0], encodings[3][0])
        fusion2 = self.cross_attn2(encodings[0][1], encodings[1][1], encodings[2][1], encodings[3][1])
        fusion3 = self.cross_attn3(encodings[0][2], encodings[1][2], encodings[2][2], encodings[3][2])

        fusion4 = self.cross_attn4(encodings[0][3], encodings[1][3], encodings[2][3], encodings[3][3])
        #x4 = self.transformer1(x4)
        #print(fusion4.size(), fusion3.size())
        x = self.up1(fusion4, fusion3)
        x = self.up2(x, fusion2)
        x = self.up3(x, fusion1)
        logits = self.outc(x)

        return logits


class U_Transformer_DS(nn.Module):
    def __init__(self, in_channels, classes, bilinear=True, batch_size=1):
        super(U_Transformer_DS, self).__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.bilinear = bilinear

        self.encoders = nn.ModuleList([Encoder(in_channels) for _ in range(4)])
        self.cross_attn1 = MultiHeadIntraAttention(64, (64, 64, 32), 1, batch_size=batch_size,
                                                   patch_size=(1,1,1), scale_factor=1)
        self.cross_attn2 = MultiHeadIntraAttention(128, (32, 32, 16), 1, batch_size=batch_size,
                                                   patch_size=(1,1,1), scale_factor=1)
        self.cross_attn3 = MultiHeadIntraAttention(256, (16, 16, 8), 1, batch_size=batch_size,
                                                   patch_size=(1,1,1), scale_factor=1)
        self.cross_attn4 = MultiHeadIntraAttention(512, (8, 8, 4), 1, batch_size=batch_size,
                                                   patch_size=(1,1,1), scale_factor=1)
        self.outc = OutConv(64, classes)
        self.batch_size = batch_size

        self.up1 = Up(in_channels_from_down_path=512, skip_channels=256, out_channels=256)
        self.up2 = Up(in_channels_from_down_path=256, skip_channels=128, out_channels=128)
        self.up3 = Up(in_channels_from_down_path=128, skip_channels=64, out_channels=64)

    def forward(self, x):
        encodings = [self.encoders[i](x[:, i : i+1, ...]) for i in range(4)]
        fusion1 = self.cross_attn1(encodings[0][0], encodings[1][0], encodings[2][0], encodings[3][0])
        fusion2 = self.cross_attn2(encodings[0][1], encodings[1][1], encodings[2][1], encodings[3][1])
        fusion3 = self.cross_attn3(encodings[0][2], encodings[1][2], encodings[2][2], encodings[3][2])

        fusion4 = self.cross_attn4(encodings[0][3], encodings[1][3], encodings[2][3], encodings[3][3])
        x = self.up1(fusion4, fusion3)
        x = self.up2(x, fusion2)
        x = self.up3(x, fusion1)
        logits = self.outc(x)

        return logits


class U_Transformer_VAE(nn.Module):
    def __init__(self, in_channels, classes, bilinear=True):
        super(U_Transformer_VAE, self).__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.bilinear = bilinear

        self.encoders = nn.ModuleList([Encoder_VAE(in_channels) for _ in range(4)])
        self.gaussians = nn.ModuleList([GaussianSampler() for _ in range(4)])

        self.cross_attn4 = MultiHeadCrossAttention(512, (28, 28, 18), 1)

        self.outc = OutConv(64, classes)

        self.up1 = Up(in_channels_from_down_path=512, skip_channels=128, out_channels=256)
        self.up2 = Up(in_channels_from_down_path=256, skip_channels=64, out_channels=128)
        self.up3 = Up(in_channels_from_down_path=128, skip_channels=32, out_channels=64)

    def forward(self, x):
        all_encodings = [self.encoders[i](x[:, i].reshape(1, 1, 224, 224, 144)) for i in range(4)]

        channels_per_level = [
            64,  # x1 (inc output)
            128, # x2 (down1 output)
            256, # x3 (down2 output)
            512  # x4 (down3 output)
        ]

        fused_encodings_per_level = []

        for level_idx in range(3):
            current_total_channels = channels_per_level[level_idx]
            half_channels = current_total_channels // 2

            level_means = []
            level_logvars = []

            for mod_idx in range(4):
                encoding_output_for_mod_level = all_encodings[mod_idx][level_idx]

                level_means.append(encoding_output_for_mod_level[:, :half_channels, ...].unsqueeze(0))
                logvar_tensor = encoding_output_for_mod_level[:, half_channels:, ...]
                level_logvars.append(torch.log(nn.ReLU(logvar_tensor) + 1e-7).unsqueeze(0))

            fused_latent_for_level = self.gaussians[level_idx](
                torch.cat(level_means, dim=0),
                torch.cat(level_logvars, dim=0),
                torch.arange(0, 4, step=1),
                torch.tensor([True, True, True, True], dtype=torch.bool),
                not self.training
            )
            fused_encodings_per_level.append(fused_latent_for_level)

        fused_x1, fused_x2, fused_x3 = fused_encodings_per_level

        final_fusion4 = self.cross_attn4(all_encodings[0][3], all_encodings[0][3],
                                         all_encodings[0][3], all_encodings[0][3])

        x = self.up1(final_fusion4, fused_x3)
        x = self.up2(x, fused_x2)
        x = self.up3(x, fused_x1)
        logits = self.outc(x)

        return logits


class U_Transformer_VAE_DS(nn.Module):
    def __init__(self, in_channels, classes, bilinear=True, batch_size=1):
        super(U_Transformer_VAE_DS, self).__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.bilinear = bilinear

        self.encoders = nn.ModuleList([Encoder_VAE(in_channels,  last_relu=i < 3, 
                                                   channels=(8, 16, 32, 64)) for i in range(4)])
        self.gaussians = nn.ModuleList([GaussianSampler() for _ in range(4)])

        self.cross_attn4 = MultiHeadIntraAttention(64, (28, 28, 18), 1, batch_size=batch_size,
                                                   patch_size=(1,1,1), scale_factor=1,
                                                   post=False)

        self.outc = OutConv(8, classes)

        self.up1 = Up(in_channels_from_down_path=64, skip_channels=16, out_channels=32)
        self.up2 = Up(in_channels_from_down_path=32, skip_channels=8, out_channels=16)
        self.up3 = Up(in_channels_from_down_path=16, skip_channels=4, out_channels=8)

        self.level_params = {}

    def _encode_and_sample_level(self, all_encodings, level_idx, channels_per_level):
        half_channels = channels_per_level[level_idx] // 2
        mus, logvars = [], []
        
        for mod_idx in range(4):
            enc = all_encodings[mod_idx][level_idx]
            enc = enc.as_tensor() if hasattr(enc, 'as_tensor') else enc
            mu = enc[:, :half_channels, ...]
            lv = torch.log(enc[:, half_channels:, ...] + 1e-7)
            lv = torch.clamp(lv, min=-10, max=10)
            mus.append(mu)
            logvars.append(lv)
        
        device = mus[0].device
        fused = self.gaussians[level_idx](
            torch.stack(mus, dim=0),
            torch.stack(logvars, dim=0),
            torch.arange(0, 4, device=device),
            torch.ones(4, dtype=torch.bool, device=device),
            not self.training
        )
        return fused, mus, logvars

    def forward(self, x):
        all_encodings = [self.encoders[i](x[:, i : i+1, ...]) for i in range(4)]

        channels_per_level = [
            8,  # x1 (inc output)
            16, # x2 (down1 output)
            32, # x3 (down2 output)
            64  # x4 (down3 output)
        ]

        fused_encodings_per_level = []

        level_params = {0 : {'mu': [], 'log_var': []},
                        1 : {'mu': [], 'log_var': []},
                        2 : {'mu': [], 'log_var': []}}

        for level_idx in range(3):
            fused, mus, logvars = self._encode_and_sample_level(
            all_encodings, level_idx, channels_per_level
            )
            fused_encodings_per_level.append(fused)
            level_params[level_idx]['mu'] = mus
            level_params[level_idx]['log_var'] = logvars

        fused_x1, fused_x2, fused_x3 = fused_encodings_per_level

        final_fusion4 = self.cross_attn4(all_encodings[0][3], all_encodings[0][3],
                                         all_encodings[0][3], all_encodings[0][3])

        x = self.up1(final_fusion4, fused_x3)
        x = self.up2(x, fused_x2)
        x = self.up3(x, fused_x1)
        logits = self.outc(x)

        return_val = (logits, level_params) if self.training else logits
        return return_val


class U_VAE_DS(nn.Module):
    def __init__(self, in_channels, classes, bilinear=True, batch_size=1):
        super(U_VAE_DS, self).__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.bilinear = bilinear

        self.encoders = nn.ModuleList([Encoder_VAE(in_channels, last_relu=True, 
                                                   channels=(8, 16, 32, 64)) for _ in range(4)])
        self.gaussians = nn.ModuleList([GaussianSampler() for _ in range(4)])

        self.outc = OutConv(8, classes)

        self.up1 = Up(in_channels_from_down_path=32, skip_channels=16, out_channels=32)
        self.up2 = Up(in_channels_from_down_path=32, skip_channels=8, out_channels=16)
        self.up3 = Up(in_channels_from_down_path=16, skip_channels=4, out_channels=8)

        self.level_params = {0 : {'mu': [], 'log_var': []},
                        1 : {'mu': [], 'log_var': []},
                        2 : {'mu': [], 'log_var': []},
                        3 : {'mu': [], 'log_var': []}}

    def _encode_and_sample_level(self, all_encodings, level_idx, channels_per_level):
        half_channels = channels_per_level[level_idx] // 2
        mus, logvars = [], []
        
        for mod_idx in range(4):
            enc = all_encodings[mod_idx][level_idx]
            # strip MetaTensor
            enc = enc.as_tensor() if hasattr(enc, 'as_tensor') else enc
            mu = enc[:, :half_channels, ...]
            lv = torch.log(enc[:, half_channels:, ...] + 1e-7)
            mus.append(mu)
            logvars.append(lv)
        
        device = mus[0].device
        fused = self.gaussians[level_idx](
            torch.stack(mus, dim=0),
            torch.stack(logvars, dim=0),
            torch.arange(0, 4, device=device),
            torch.ones(4, dtype=torch.bool, device=device),
            not self.training
        )
        return fused, mus, logvars

    def forward(self, x):

        #print("encoding")
        # all_encodings will be a list of 4 tuples, where each tuple contains (x1, x2, x3, x4) for one modality

        # Define the total channels for each encoder level output
        
        channels_per_level = [
            8,  # x1 (inc output)
            16, # x2 (down1 output)
            32, # x3 (down2 output)
            64  # x4 (down3 output)
        ]

        fused_encodings_per_level = []
        level_params = {0 : {'mu': [], 'log_var': []},
                        1 : {'mu': [], 'log_var': []},
                        2 : {'mu': [], 'log_var': []},
                        3 : {'mu': [], 'log_var': []}}

        #print("gaussian sampling per level")
        all_encodings = [checkpoint(self.encoders[i], x[:, i:i+1, ...], 
                 use_reentrant=False) for i in range(4)]

        level_params = {i: {'mu': [], 'log_var': []} for i in range(4)}
        fused_encodings_per_level = []
        
        for level_idx in range(4):
            fused, mus, logvars = checkpoint(
                self._encode_and_sample_level,
                all_encodings, level_idx, channels_per_level,
                use_reentrant=False
            )
            fused_encodings_per_level.append(fused)
            level_params[level_idx]['mu'] = mus
            level_params[level_idx]['log_var'] = logvars

        # Unpack the fused latents for each level
        fused_x1, fused_x2, fused_x3, fusion_x4 = fused_encodings_per_level
        #self.level_params = level_params

        # Adjusting the Up calls to directly use the final_fusionX after cross-attention
        x = checkpoint(self.up1, fusion_x4, fused_x3, use_reentrant=False)
        x = checkpoint(self.up2, x, fused_x2, use_reentrant=False)
        x = checkpoint(self.up3, x, fused_x1, use_reentrant=False)
        logits = self.outc(x)

        #print("training status: ", self.training)

        return_val = (logits, level_params) if self.training else logits
        return return_val