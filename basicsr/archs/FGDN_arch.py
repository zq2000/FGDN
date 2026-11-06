"""FGDN architecture.

Frequency-Guided Directional Deraining Network (FGDN) from
"Bridging Wavelet and Fourier Domains for Direction-Aware Single-Image
Deraining".

Paper modules <-> classes in this file:
    FGDN                       top-level encoder-decoder network
    WISE                       Wavelet-Informed Subband Enhancement
    DSPB                       Dual-Scale Pinwheel Block (low-frequency enhancement)
    DWAB                       Directional Wavelet Attention Block (DTSA + WTBlock)
    SGFFN                      Spectral-Gated Feed-Forward Network
    PConv / PConvLight         Pinwheel-shaped convolutions
    WaveletUnit / Wavelet_Feature_Extractor / WTBlock   wavelet convolution branch
    Downsample                 wavelet-based downsampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import ptwt

from einops import rearrange
from basicsr.utils.registry import ARCH_REGISTRY

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class PConv(nn.Module):
    ''' Pinwheel-shaped Convolution using the Asymmetric Padding method. '''

    def __init__(self, c1, c2, k, s):
        super().__init__()

        # self.k = k
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        #self.cat = Conv(c2, c2, 2, s=1, p=0)
        self.cat_dw = Conv(c2, c2, 2, s=1, p=0, g=c2)   # D x 1 x 4 = 4D params
        self.cat_pw = Conv(c2, c2, 1, s=1)

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        out = torch.cat([yw0, yw1, yh0, yh1], dim=1)
        return self.cat_pw(self.cat_dw(out))

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class SGFFN(nn.Module):
    """
    Spectral-Gated Feed-Forward Network (SGFFN)

    Spatial VALUE path: multi-scale DWConv (local detail, proven by MSFN)
    Spectral GATE path: FFT -> learnable amplitude filter -> iFFT -> sigmoid
    Fusion: VALUE * GATE (StarNet-style cross-domain star operation)

    Zero BatchNorm. Fully compatible with existing TransformerBlock interface.
    """
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)

        # --- Spatial VALUE: multi-scale local features ---
        self.dw3 = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=bias)
        self.dw5 = nn.Conv2d(hidden, hidden, 5, 1, 2, groups=hidden, bias=bias)

        # --- Spectral GATE: frequency-domain amplitude modulation ---
        # DWConv on amplitude spectrum = learnable local frequency-band filter
        # each channel independently learns which frequency bands to enhance/suppress
        self.amp_filter = nn.Conv2d(hidden, hidden, 3, 1, 1,
                                     groups=hidden, bias=True)
        # 1x1 for inter-channel mixing + gate projection
        self.gate_proj = nn.Conv2d(hidden, hidden, 1, bias=bias)

        self.project_out = nn.Conv2d(hidden * 2, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x_s, x_f = x.chunk(2, dim=1)
        B, C, H, W = x_f.shape

        # VALUE: multi-scale spatial features
        v3 = F.gelu(self.dw3(x_s))
        v5 = F.gelu(self.dw5(x_s))

        # GATE: spectral amplitude modulation
        f = torch.fft.rfft2(x_f, norm='ortho')
        f_amp = f.abs()                            # amplitude spectrum [B,C,H,W//2+1]
        f_phase = f / (f_amp + 1e-8)               # normalized phase (preserved)
        f_amp = F.relu(self.amp_filter(f_amp))      # learnable frequency response
        f = f_amp * f_phase                         # reconstruct with filtered amplitude
        x_spec = torch.fft.irfft2(f, s=(H, W), norm='ortho')
        gate = torch.sigmoid(self.gate_proj(x_spec))

        # Cross-domain star operation: spatial VALUE x spectral GATE
        out = torch.cat([v3 * gate, v5 * gate], dim=1)
        return self.project_out(out)

class FeedforwardNetwork(nn.Module):
    def __init__(self, dim):
        super(FeedforwardNetwork, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(dim, dim*2, 1),
            nn.GELU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
        )
        self.conv_dw = nn.Sequential(
            nn.Conv2d(dim*2, dim*2, kernel_size=3, padding=3 // 2, groups=dim*2,
                      padding_mode='reflect'),
            nn.GELU()
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_dw(x)
        x = self.conv_2(x)
        return x

class WaveletUnit(nn.Module):
    def __init__(self, in_channels, out_channels, wavelet='haar', mode='reflect'):
        super(WaveletUnit, self).__init__()
        self.wavelet = wavelet
        self.spatial_channels = in_channels // 4

        # PixelShuffle for channel <-> space conversion
        self.unshuffle = nn.PixelUnshuffle(4)  # space -> channels
        self.shuffle = nn.PixelShuffle(4)      # channels -> space

        self.pconv = PConv(self.spatial_channels, self.spatial_channels, k=3, s=1)
        self.bn = nn.BatchNorm2d(in_channels * 4)

    def forward(self, x):
        B, C, H, W = x.shape

        coeffs = ptwt.wavedec2(x, wavelet=self.wavelet, level=1, mode='reflect')
        cA, (cH, cV, cD) = coeffs[0], coeffs[1]
        combined = torch.cat([cA, cH, cV, cD], dim=1)  # [B, 4C, h, w]

        # channels -> space: [B, 4C, h, w] -> [B, C/4, 4h, 4w]
        combined = self.shuffle(combined)

        # PConv operates on the reduced channel count
        combined = self.pconv(combined)

        # space -> channels: [B, C/4, 4h, 4w] -> [B, 4C, h, w]
        combined = self.unshuffle(combined)

        combined = self.bn(combined)

        cA, cH, cV, cD = torch.chunk(combined, 4, dim=1)
        out = ptwt.waverec2([cA, (cH, cV, cD)], wavelet=self.wavelet)
        return out[:, :, :H, :W]

class Wavelet_Feature_Extractor(nn.Module):
    def __init__(self, dim):    # Wavelet Feature Extractor
        super(Wavelet_Feature_Extractor, self).__init__()
        self.dim = dim

        self.ca_conv = nn.Sequential(
            nn.Conv2d(2*dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, padding_mode='reflect'),
            nn.GELU()
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU()
        )
        self.dw_conv_1 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=3 // 2,
                      groups=self.dim, padding_mode='reflect'),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        self.dw_conv_2 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=5, padding=5 // 2,
                      groups=self.dim, padding_mode='reflect'),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )

        self.WU = WaveletUnit(self.dim*2, self.dim*2)

        self.bn = torch.nn.BatchNorm2d(dim*2)
        self.gelu = torch.nn.GELU()


    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim, dim=1))
        x_local_1 = self.dw_conv_1(x[0])
        x_local_2 = self.dw_conv_2(x[1])
        x0 = torch.cat([x_local_1, x_local_2], dim=1)
        #x_local_2 = self.dw_conv_2(x[0])
        x = self.WU(x0) + x0
        x = self.gelu(self.bn(x))
        x = self.ca_conv(x)
        x = self.ca(x) * x

        return x

class WTBlock(nn.Module):
    def __init__(self, dim, depth=2):
        super(WTBlock, self).__init__()
        self.dim = dim
        self.depth = depth
        self.norm1 = nn.ModuleList([torch.nn.BatchNorm2d(dim) for _ in range(depth)])
        self.norm2 = nn.ModuleList([torch.nn.BatchNorm2d(dim) for _ in range(depth)])
        self.mixer = nn.ModuleList([Wavelet_Feature_Extractor(dim=dim) for _ in range(depth)])
        self.ffn = nn.ModuleList([FeedforwardNetwork(dim=dim) for _ in range(depth)])

    def forward(self, x):
        for i in range(self.depth):
            copy = x
            x = self.norm1[i](x)
            x = self.mixer[i](x)
            x = x + copy

            copy = x
            x = self.norm2[i](x)
            x = self.ffn[i](x)
            x = x + copy

        return x

class DWAB(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(DWAB, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_PConv = PConv(dim * 3, dim * 3, k=3, s=1)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

        self.wt_block = WTBlock(dim=dim, depth=2)
        self.fusion_scale = nn.Parameter(torch.zeros(1, dim, 1, 1))  # channel-wise scaling

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_PConv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape

        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        index = torch.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        # wavelet convolution branch (WTBlock)
        wt_output = self.wt_block(x)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        # fuse wavelet-branch features
        out = out + self.fusion_scale * wt_output
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = DWAB(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = SGFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class PConvLight(nn.Module):
    """BN-free lightweight PConv: 4 directional DW branches -> fuse -> PW."""
    def __init__(self, C, k=3):
        super(PConvLight, self).__init__()
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(p[g]) for g in range(4)]
        self.cw = nn.Sequential(
            nn.Conv2d(C, C, (1, k), groups=C, bias=False),
            nn.Conv2d(C, C, 1, bias=False), nn.ReLU(inplace=False))
        self.ch = nn.Sequential(
            nn.Conv2d(C, C, (k, 1), groups=C, bias=False),
            nn.Conv2d(C, C, 1, bias=False), nn.ReLU(inplace=False))
        self.fuse = nn.Sequential(
            nn.Conv2d(C * 4, C * 4, 2, stride=1, groups=C * 4, bias=False),
            nn.Conv2d(C * 4, C, 1, bias=False), nn.ReLU(inplace=False))

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.fuse(torch.cat([yw0, yw1, yh0, yh1], dim=1))


class DSPB(nn.Module):
    """Dual-Scale Pinwheel Block: low-frequency enhancement with 3- and 5-tap PConvLight."""
    def __init__(self, dim, expand_ratio=4):
        super().__init__()
        hidden = dim * expand_ratio
        self.PConv3 = PConvLight(dim, 3)
        self.PConv5 = PConvLight(dim, 5)
        self.expand = nn.Conv2d(dim * 2, hidden, 1, bias=False)
        self.act = nn.GELU()
        self.compress = nn.Conv2d(hidden, dim, 1, bias=False)

    def forward(self, x):
        x3 = self.PConv3(x)
        x5 = self.PConv5(x)
        return self.compress(self.act(self.expand(torch.cat([x3, x5], dim=1))))

class WISE(nn.Module):
    """Wavelet-Informed Subband Enhancement."""
    def __init__(self, dim, wavelet='haar'):
        super().__init__()
        self.wavelet = wavelet

        self.ll_norm1 = LayerNorm(dim, 'WithBias')
        self.ll_ffn1  = DSPB(dim, expand_ratio=4)
        self.ll_norm2 = LayerNorm(dim, 'WithBias')
        self.ll_ffn2  = DSPB(dim, expand_ratio=4)

        # per-subband FiLM gating (replaces a single shared gate generator)
        self.gate_h = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim * 2, 1, bias=False),   # scale + shift
        )
        self.gate_v = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim * 2, 1, bias=False),
        )
        self.gate_d = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim * 2, 1, bias=False),
        )

        self.lh_norm = LayerNorm(dim, 'WithBias')
        self.lh_ffn  = DSPB(dim, expand_ratio=4)
        self.hl_norm = LayerNorm(dim, 'WithBias')
        self.hl_ffn  = DSPB(dim, expand_ratio=4)
        self.hh_norm = LayerNorm(dim, 'WithBias')
        self.hh_ffn  = DSPB(dim, expand_ratio=4)

        self.ffn = SGFFN(dim, 3, bias=False)

        self.cross_subband = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=False),  # spatial context
            nn.GELU(),
            nn.Conv2d(dim * 3, dim * 3, 1, bias=False),  # cross-subband channel mixing
        )

    def forward(self, x):
        B, C, H, W = x.shape

        coeffs = ptwt.wavedec2(x, wavelet=self.wavelet, level=1, mode='reflect')
        cA, (cH, cV, cD) = coeffs[0], coeffs[1]

        cA = self.ll_ffn1(self.ll_norm1(cA)) + cA
        cA = self.ll_ffn2(self.ll_norm2(cA)) + cA

        # each gate sees both cA (semantic context) and the corresponding subband (local high-frequency content)
        s_h, b_h = self.gate_h(torch.cat([cA, cH], dim=1)).chunk(2, dim=1)
        s_v, b_v = self.gate_v(torch.cat([cA, cV], dim=1)).chunk(2, dim=1)
        s_d, b_d = self.gate_d(torch.cat([cA, cD], dim=1)).chunk(2, dim=1)

        cH = self.lh_ffn(self.lh_norm(cH)) * torch.sigmoid(s_h) + b_h + cH
        cV = self.hl_ffn(self.hl_norm(cV)) * torch.sigmoid(s_v) + b_v + cV
        cD = self.hh_ffn(self.hh_norm(cD)) * torch.sigmoid(s_d) + b_d + cD

        high = torch.cat([cH, cV, cD], dim=1)
        high = self.cross_subband(high) + high   # residual
        cH, cV, cD = high.chunk(3, dim=1)
        coeffs_out = [cA, (cH, cV, cD)]
        out = ptwt.waverec2(coeffs_out, wavelet=self.wavelet)[:, :, :H, :W]

        return self.ffn(out) + x

## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

## Wavelet Downsample
class Downsample(nn.Module):
    def __init__(self, n_feat, wavelet='haar', mode='reflect'):
        super(Downsample, self).__init__()
        self.wavelet = wavelet
        self.mode = mode
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(n_feat*4, n_feat*2, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(n_feat*2),
                                    nn.ReLU(inplace=True),
                                    )

    def forward(self, x):
        coeffs = ptwt.wavedec2(x, wavelet=self.wavelet, level=1, mode=self.mode)
        cA = coeffs[0]
        cH, cV, cD = coeffs[1]
        x = torch.cat([cA, cH, cV, cD], dim=1)
        x = self.conv_bn_relu(x)
        return x

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

@ARCH_REGISTRY.register()
class FGDN(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=32,
                 num_blocks=[2, 2, 3, 3],
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=1.6667,
                 bias=False,
                 LayerNorm_type='WithBias'  ## Other option 'BiasFree'
                 ):

        super(FGDN, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level0 = WISE(dim)  ## WISE at the encoder front-end

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = WISE(dim=int(dim*2**1)) ## WISE at the decoder refinement stage

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        inp_enc_level0 = self.encoder_level0(inp_enc_level1) ## WISE at the encoder front-end
        out_enc_level1 = self.encoder_level1(inp_enc_level0)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1) ## WISE at the decoder refinement stage

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1

if __name__ == '__main__':
    input = torch.rand(1, 3, 256, 256)
    model = FGDN()
   # output = model(input)

    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    flops = FlopCountAnalysis(model, input)
    print("FLOPs: ", flops.total())
