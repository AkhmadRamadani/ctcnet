"""
models.py — CTCNet Architecture
All building blocks from:
  "CTCNet: A CNN-Transformer Cooperation Network for Face Image Super-Resolution"
  Gao et al., arXiv:2204.08696
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ── Channel Attention ─────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


# ── AFDU ──────────────────────────────────────────────────────────────────────

class AFDU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        mid = channels // 2
        self.reduction  = nn.Conv2d(channels, mid, 3, padding=1)
        self.expansion  = nn.Conv2d(mid, channels, 3, padding=1)
        self.conv1x1    = nn.Conv2d(2 * channels, channels, 1)
        self.conv3x3    = nn.Conv2d(channels, channels, 3, padding=1)
        self.ca         = ChannelAttention(channels)
        self.refine     = nn.Conv2d(channels, channels, 3, padding=1)
        self.lrelu      = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        reduced  = self.lrelu(self.reduction(x))
        expanded = self.lrelu(self.expansion(reduced))
        fused    = self.lrelu(self.conv1x1(torch.cat([x, expanded], dim=1)))
        fused    = self.lrelu(self.conv3x3(fused))
        fused    = self.ca(fused)
        return self.refine(fused) + x


# ── Hourglass ─────────────────────────────────────────────────────────────────

class HourglassBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.down1  = nn.Sequential(nn.Conv2d(channels, channels, 3, stride=2, padding=1), nn.LeakyReLU(0.2, True))
        self.down2  = nn.Sequential(nn.Conv2d(channels, channels, 3, stride=2, padding=1), nn.LeakyReLU(0.2, True))
        self.middle = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.LeakyReLU(0.2, True))
        self.up2    = nn.Sequential(nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True))
        self.up1    = nn.Sequential(nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True))

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        m  = self.middle(d2)
        u2 = self.up2(m)[:, :, :d1.shape[2], :d1.shape[3]]
        u1 = self.up1(u2 + d1)[:, :, :x.shape[2], :x.shape[3]]
        return u1


# ── FSAU ──────────────────────────────────────────────────────────────────────

class FSAU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.afdu1       = AFDU(channels)
        self.hourglass   = HourglassBlock(channels)
        self.ca          = ChannelAttention(channels)
        self.spatial_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.afdu2       = AFDU(channels)

    def forward(self, x):
        feat   = self.afdu1(x)
        hg     = self.hourglass(feat)
        hg_ca  = self.ca(hg) + feat
        sa_map = torch.sigmoid(self.spatial_conv(hg_ca))
        return self.afdu2(feat * sa_map)


# ── Transformer Block (MDTA + GDFN) ──────────────────────────────────────────

class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.num_heads   = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv         = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.qkv_dw      = nn.Conv2d(channels * 3, channels * 3, 3, padding=1, groups=channels * 3, bias=False)
        self.proj        = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv_dw(self.qkv(x)).chunk(3, dim=1)
        q = rearrange(q, 'b (h d) x y -> b h d (x y)', h=self.num_heads)
        k = rearrange(k, 'b (h d) x y -> b h d (x y)', h=self.num_heads)
        v = rearrange(v, 'b (h d) x y -> b h d (x y)', h=self.num_heads)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1) * self.temperature).softmax(dim=-1)
        out  = rearrange(attn @ v, 'b h d (x y) -> b (h d) x y', x=H, y=W)
        return self.proj(out)


class GDFN(nn.Module):
    def __init__(self, channels, expansion=2.66):
        super().__init__()
        hidden       = int(channels * expansion)
        self.proj_in  = nn.Conv2d(channels, hidden * 2, 1, bias=False)
        self.dw_conv  = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=False)
        self.proj_out = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, x):
        x1, x2 = self.dw_conv(self.proj_in(x)).chunk(2, dim=1)
        return self.proj_out(x1 * F.gelu(x2))


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn  = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn   = GDFN(channels)

    def _norm(self, x, norm):
        B, C, H, W = x.shape
        return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def forward(self, x):
        x = x + self.attn(self._norm(x, self.norm1))
        x = x + self.ffn(self._norm(x, self.norm2))
        return x


# ── LGCM ──────────────────────────────────────────────────────────────────────

class LGCM(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.fsau        = FSAU(channels)
        self.transformer = TransformerBlock(channels, num_heads)

    def forward(self, x):
        return self.fsau(x) + self.transformer(x)


# ── FEU + FRM ─────────────────────────────────────────────────────────────────

class FEU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.afdu_in1   = AFDU(channels)
        self.afdu_in2   = AFDU(channels)
        self.afdu_low1  = AFDU(channels)
        self.afdu_low2  = AFDU(channels)
        self.downsample = nn.AvgPool2d(2)
        self.fuse1      = nn.Conv2d(channels * 2, channels, 1)
        self.fuse2      = nn.Conv2d(channels * 2, channels, 1)
        self.afdu_out   = AFDU(channels)
        self.calibration = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid())
        self.lrelu      = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        f_in    = self.afdu_in1(x)
        f_low   = self.afdu_low1(self.downsample(x))
        f_low2  = self.afdu_low2(f_low)
        f_low_up = F.interpolate(f_low, size=f_in.shape[2:], mode='bilinear', align_corners=False)
        f_in2   = self.lrelu(self.fuse1(torch.cat([self.afdu_in1(f_in), self.afdu_low1(f_low_up)], dim=1)))
        f_low2_up = F.interpolate(f_low2, size=f_in2.shape[2:], mode='bilinear', align_corners=False)
        f_in3   = self.lrelu(self.fuse2(torch.cat([self.afdu_in2(f_in2), self.afdu_low2(f_low2_up)], dim=1)))
        return self.afdu_out(f_in3) + x * self.calibration(x)


class FRM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fsau = FSAU(channels)
        self.feu  = FEU(channels)

    def forward(self, x):
        return self.feu(self.fsau(x))


# ── MFFU ──────────────────────────────────────────────────────────────────────

class MFFU(nn.Module):
    def __init__(self, channels_list, target_channels):
        super().__init__()
        self.fuse = nn.Conv2d(sum(channels_list) + target_channels, target_channels, 1)
        self.ca   = ChannelAttention(target_channels)

    def forward(self, enc_feats, dec_feat):
        H, W = dec_feat.shape[2], dec_feat.shape[3]
        resized = [
            F.interpolate(ef, size=(H, W), mode='bilinear', align_corners=False)
            if ef.shape[2] != H or ef.shape[3] != W else ef
            for ef in enc_feats
        ] + [dec_feat]
        return self.ca(self.fuse(torch.cat(resized, dim=1)))


# ── CTCNet ────────────────────────────────────────────────────────────────────

class CTCNet(nn.Module):
    """
    CNN-Transformer Cooperation Network for Face Super-Resolution.
    Paper: arXiv:2204.08696
    """
    def __init__(self, base_channels=32, num_frm=4, num_heads=4, scale=8, sr_head_mid_channels=None):
        super().__init__()
        C = base_channels
        self.scale = scale
        # sr_head middle channels — defaults to C but CTCGAN may use a larger value
        mid = sr_head_mid_channels if sr_head_mid_channels is not None else C

        self.shallow_conv = nn.Conv2d(3, C, 3, padding=1)

        self.enc_lgcm1  = LGCM(C, num_heads)
        self.downsample1 = self._make_downsample(C, C * 2)
        self.enc_lgcm2  = LGCM(C * 2, num_heads)
        self.downsample2 = self._make_downsample(C * 2, C * 4)
        self.enc_lgcm3  = LGCM(C * 4, num_heads)
        self.downsample3 = self._make_downsample(C * 4, C * 4)

        self.bottleneck = nn.Sequential(*[FRM(C * 4) for _ in range(num_frm)])

        self.upsample1  = self._make_upsample(C * 4, C * 4)
        self.mffu1      = MFFU([C, C * 2, C * 4], C * 4)
        self.dec_lgcm1  = LGCM(C * 4, num_heads)

        self.upsample2  = self._make_upsample(C * 4, C * 2)
        self.mffu2      = MFFU([C, C * 2, C * 4], C * 2)
        self.dec_lgcm2  = LGCM(C * 2, num_heads)

        self.upsample3  = self._make_upsample(C * 2, C)
        self.mffu3      = MFFU([C, C * 2, C * 4], C)
        self.dec_lgcm3  = LGCM(C, num_heads)

        self.sr_head = nn.Sequential(
            nn.Conv2d(C, mid, 3, padding=1),   # mid channels auto-matched to checkpoint
            nn.PixelShuffle(scale),
            nn.Conv2d(mid // (scale * scale), 3, 3, padding=1)
        )
        self.upsample_lr = nn.Upsample(scale_factor=scale, mode='bicubic', align_corners=False)

    def _make_downsample(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1)
        )

    def _make_upsample(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 6, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1)
        )

    def forward(self, x_lr):
        f0  = self.shallow_conv(x_lr)
        e1  = self.enc_lgcm1(f0)
        e1d = self.downsample1(e1)
        e2  = self.enc_lgcm2(e1d)
        e2d = self.downsample2(e2)
        e3  = self.enc_lgcm3(e2d)
        e3d = self.downsample3(e3)

        bn  = self.bottleneck(e3d)

        d1  = self.dec_lgcm1(self.mffu1([e1, e2, e3], self.upsample1(bn)))
        d2  = self.dec_lgcm2(self.mffu2([e1, e2, e3], self.upsample2(d1)))
        d3  = self.dec_lgcm3(self.mffu3([e1, e2, e3], self.upsample3(d2)))

        i_out   = self.sr_head(d3)
        i_lr_up = self.upsample_lr(x_lr)
        h, w    = i_lr_up.shape[2], i_lr_up.shape[3]

        return torch.clamp(i_lr_up + i_out[:, :, :h, :w], 0.0, 1.0)


# ── ResNetSR ──────────────────────────────────────────────────────────────────

class ResNetSR(nn.Module):
    """
    Placeholder/Adapter for the specific model architecture detected in best_model.pth.
    Keys suggest: head, res_layers_down1_pre.encoder.layer1, etc.
    """
    def __init__(self):
        super().__init__()
        # Based on keys: head.weight, res_layers_down1_pre.encoder...
        self.head = nn.Conv2d(3, 32, 3, padding=1)  # Fixed: 32 channels based on checkpoint

        # Structure for res_layers_down1_pre
        # Keys show: encoder.layer1, layer2, layer4, alise, atten
        self.res_layers_down1_pre = nn.Module()
        self.res_layers_down1_pre.encoder = nn.Module()

        # Helper to make a dummy layer that accepts weights
        def make_layer():
            return nn.Sequential(
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1)
            )

        self.res_layers_down1_pre.encoder.layer1 = make_layer()
        self.res_layers_down1_pre.encoder.layer2 = make_layer()
        self.res_layers_down1_pre.encoder.layer3 = make_layer() # Guessing layer3 exists
        self.res_layers_down1_pre.encoder.layer4 = make_layer()

        self.res_layers_down1_pre.encoder.alise = nn.Conv2d(32, 32, 1) # Guess
        self.res_layers_down1_pre.encoder.atten = nn.Sequential(nn.Conv2d(32,32,1)) # Guess

    def forward(self, x):
        # Pass-through for now as we don't have the real forward logic
        return F.interpolate(x, scale_factor=8, mode='bicubic', align_corners=False)
