import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets import SwinUNETR


def _to_2tuple(value):
    if isinstance(value, tuple):
        return value
    return (value, value)


def _drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return _drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    batch, height, width, channels = x.shape
    x = x.view(
        batch,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, channels)


def window_reverse(windows, window_size, height, width):
    batch = int(windows.shape[0] / (height * width / window_size / window_size))
    x = windows.view(
        batch,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        -1,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(batch, height, width, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = _to_2tuple(window_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        table_shape = (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(table_shape, num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        batch_windows, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_windows, tokens, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(tokens, tokens, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(batch_windows // num_windows, num_windows, self.num_heads, tokens, tokens)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, tokens, tokens)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch_windows, tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = min(window_size, min(input_resolution))
        self.shift_size = 0 if min(input_resolution) <= self.window_size else shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio), drop=drop)

        if self.shift_size > 0:
            self.register_buffer("attn_mask", self._build_attention_mask())
        else:
            self.attn_mask = None

    def _build_attention_mask(self):
        height, width = self.input_resolution
        img_mask = torch.zeros((1, height, width, 1))
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        count = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = count
                count += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)

    def forward(self, x):
        height, width = self.input_resolution
        batch, length, channels = x.shape
        if length != height * width:
            raise ValueError(f"Input token length {length} does not match resolution {height}x{width}")

        shortcut = x
        x = self.norm1(x)
        x = x.view(batch, height, width, channels)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, channels)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        shifted_x = window_reverse(attn_windows, self.window_size, height, width)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(batch, height * width, channels)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=(96, 96), patch_size=4, in_channels=1, embed_dim=24, norm_layer=nn.LayerNorm):
        super().__init__()
        img_size = _to_2tuple(img_size)
        patch_size = _to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        batch, channels, height, width = x.shape
        if (height, width) != self.img_size:
            raise ValueError(f"Swin-Unet 2D branch expects slices sized {self.img_size}, got {(height, width)}")
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        height, width = self.input_resolution
        batch, length, channels = x.shape
        if length != height * width:
            raise ValueError("PatchMerging received an unexpected token count")
        x = x.view(batch, height, width, channels)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1).view(batch, -1, 4 * channels)
        x = self.norm(x)
        return self.reduction(x)


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        height, width = self.input_resolution
        x = self.expand(x)
        batch, length, channels = x.shape
        if length != height * width:
            raise ValueError("PatchExpand received an unexpected token count")
        x = x.view(batch, height, width, channels)
        x = x.view(batch, height, width, self.dim_scale, self.dim_scale, channels // (self.dim_scale ** 2))
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(batch, height * self.dim_scale, width * self.dim_scale, channels // (self.dim_scale ** 2))
        x = x.view(batch, -1, channels // (self.dim_scale ** 2))
        return self.norm(x)


class FinalPatchExpandX4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, (dim_scale ** 2) * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        height, width = self.input_resolution
        x = self.expand(x)
        batch, length, channels = x.shape
        if length != height * width:
            raise ValueError("FinalPatchExpandX4 received an unexpected token count")
        x = x.view(batch, height, width, self.dim_scale, self.dim_scale, channels // (self.dim_scale ** 2))
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(batch, height * self.dim_scale, width * self.dim_scale, channels // (self.dim_scale ** 2))
        x = x.view(batch, -1, self.output_dim)
        return self.norm(x)


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        downsample=None,
    ):
        super().__init__()
        drop_path = [drop_path] * depth if not isinstance(drop_path, list) else drop_path
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (idx % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[idx],
                )
                for idx in range(depth)
            ]
        )
        self.downsample = downsample(input_resolution, dim=dim) if downsample is not None else None

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class BasicLayerUp(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        upsample=None,
    ):
        super().__init__()
        drop_path = [drop_path] * depth if not isinstance(drop_path, list) else drop_path
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (idx % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[idx],
                )
                for idx in range(depth)
            ]
        )
        self.upsample = upsample(input_resolution, dim=dim) if upsample is not None else None

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class SwinUnet2D(nn.Module):
    """
    Local Swin-Unet implementation for 2D slice segmentation.

    This follows the official Swin-Unet design: Swin Transformer encoder,
    symmetric Swin decoder, skip concatenation, patch expansion, and a final
    x4 upsampling projection back to the original image resolution.
    """

    def __init__(
        self,
        img_size=(96, 96),
        patch_size=4,
        in_channels=1,
        num_classes=2,
        embed_dim=24,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            dim = int(embed_dim * 2 ** layer_idx)
            resolution = (
                patches_resolution[0] // (2 ** layer_idx),
                patches_resolution[1] // (2 ** layer_idx),
            )
            layer = BasicLayer(
                dim=dim,
                input_resolution=resolution,
                depth=depths[layer_idx],
                num_heads=num_heads[layer_idx],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:layer_idx]) : sum(depths[: layer_idx + 1])],
                downsample=PatchMerging if layer_idx < self.num_layers - 1 else None,
            )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(int(embed_dim * 2 ** (self.num_layers - 1)))

        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for up_idx in range(self.num_layers):
            source_idx = self.num_layers - 1 - up_idx
            dim = int(embed_dim * 2 ** source_idx)
            resolution = (
                patches_resolution[0] // (2 ** source_idx),
                patches_resolution[1] // (2 ** source_idx),
            )
            if up_idx == 0:
                self.layers_up.append(PatchExpand(resolution, dim=dim, dim_scale=2))
                self.concat_back_dim.append(nn.Identity())
            else:
                self.concat_back_dim.append(nn.Linear(2 * dim, dim))
                self.layers_up.append(
                    BasicLayerUp(
                        dim=dim,
                        input_resolution=resolution,
                        depth=depths[source_idx],
                        num_heads=num_heads[source_idx],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:source_idx]) : sum(depths[: source_idx + 1])],
                        upsample=PatchExpand if up_idx < self.num_layers - 1 else None,
                    )
                )

        self.norm_up = nn.LayerNorm(embed_dim)
        self.up = FinalPatchExpandX4(patches_resolution, dim=embed_dim, dim_scale=4)
        self.output = nn.Conv2d(embed_dim, num_classes, kernel_size=1, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        skips = []
        for layer in self.layers:
            skips.append(x)
            x = layer(x)
        x = self.norm(x)
        return x, skips

    def forward_up_features(self, x, skips):
        for up_idx, layer_up in enumerate(self.layers_up):
            if up_idx == 0:
                x = layer_up(x)
            else:
                skip = skips[self.num_layers - 1 - up_idx]
                x = torch.cat([x, skip], dim=-1)
                x = self.concat_back_dim[up_idx](x)
                x = layer_up(x)
        return self.norm_up(x)

    def up_x4(self, x):
        height, width = self.patches_resolution
        x = self.up(x)
        batch, _, channels = x.shape
        x = x.view(batch, height * 4, width * 4, channels)
        x = x.permute(0, 3, 1, 2).contiguous()
        return self.output(x)

    def forward(self, x):
        x, skips = self.forward_features(x)
        x = self.forward_up_features(x, skips)
        return self.up_x4(x)


def _make_swinunetr(
    spatial_dims,
    in_channels,
    out_channels,
    feature_size,
    use_checkpoint,
    img_size=None,
):
    params = inspect.signature(SwinUNETR).parameters
    kwargs = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "feature_size": feature_size,
        "use_checkpoint": use_checkpoint,
    }
    if "spatial_dims" in params:
        kwargs["spatial_dims"] = spatial_dims
    elif spatial_dims != 3:
        raise RuntimeError("The installed MONAI SwinUNETR only supports the 3D branch.")
    if img_size is not None and "img_size" in params:
        kwargs["img_size"] = img_size
    return SwinUNETR(**kwargs)


class DualSwinUNetFusion(nn.Module):
    """
    Dual-branch lung nodule segmentation model.

    The 3D branch keeps the existing SwinUNETR volumetric context. The 2D branch
    is a local Swin-Unet implementation applied slice by slice. Slice logits are
    stacked back into a volume, then a 1x1x1 fuser produces the comprehensive 3D
    logits expected by the existing training and inference code.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=2,
        feature_size_3d=48,
        feature_size_2d=24,
        use_checkpoint=True,
        plane="axial",
        fusion_mode="conv",
        slice_batch_size=16,
        roi_size=(96, 96, 96),
        swin2d_depths=(2, 2, 2, 2),
        swin2d_heads=(3, 6, 12, 24),
        swin2d_window_size=6,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.plane = plane.lower()
        self.fusion_mode = fusion_mode.lower()
        self.slice_batch_size = int(slice_batch_size)

        if self.plane not in {"axial", "coronal", "sagittal"}:
            raise ValueError(f"Unsupported 2D plane: {plane}")
        if self.fusion_mode not in {"conv", "mean", "weighted"}:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")

        self.branch_3d = _make_swinunetr(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size_3d,
            use_checkpoint=use_checkpoint,
            img_size=roi_size,
        )
        self.branch_2d = SwinUnet2D(
            img_size=self._slice_size_for_plane(roi_size, self.plane),
            in_channels=in_channels,
            num_classes=out_channels,
            embed_dim=feature_size_2d,
            depths=swin2d_depths,
            num_heads=swin2d_heads,
            window_size=swin2d_window_size,
        )

        if self.fusion_mode == "conv":
            self.fuser = nn.Conv3d(out_channels * 2, out_channels, kernel_size=1)
            self._init_average_fuser()
        elif self.fusion_mode == "weighted":
            self.branch_logits = nn.Parameter(torch.zeros(2))
        else:
            self.fuser = None

    @staticmethod
    def _slice_size_for_plane(roi_size, plane):
        depth, height, width = roi_size
        if plane == "axial":
            return (height, width)
        if plane == "coronal":
            return (depth, width)
        return (depth, height)

    def _init_average_fuser(self):
        with torch.no_grad():
            self.fuser.weight.zero_()
            self.fuser.bias.zero_()
            for class_idx in range(self.out_channels):
                self.fuser.weight[class_idx, class_idx, 0, 0, 0] = 0.5
                self.fuser.weight[class_idx, class_idx + self.out_channels, 0, 0, 0] = 0.5

    def _volume_to_slices(self, x):
        if self.plane == "axial":
            batch, channels, depth, height, width = x.shape
            slices = x.permute(0, 2, 1, 3, 4).reshape(batch * depth, channels, height, width)
            return slices, (batch, depth, height, width)
        if self.plane == "coronal":
            batch, channels, depth, height, width = x.shape
            slices = x.permute(0, 3, 1, 2, 4).reshape(batch * height, channels, depth, width)
            return slices, (batch, depth, height, width)

        batch, channels, depth, height, width = x.shape
        slices = x.permute(0, 4, 1, 2, 3).reshape(batch * width, channels, depth, height)
        return slices, (batch, depth, height, width)

    def _slices_to_volume(self, y, volume_shape):
        batch, depth, height, width = volume_shape
        channels = y.shape[1]
        if self.plane == "axial":
            return y.reshape(batch, depth, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()
        if self.plane == "coronal":
            return y.reshape(batch, height, channels, depth, width).permute(0, 2, 3, 1, 4).contiguous()
        return y.reshape(batch, width, channels, depth, height).permute(0, 2, 3, 4, 1).contiguous()

    def _run_2d_branch(self, x):
        slices, volume_shape = self._volume_to_slices(x)
        if self.slice_batch_size <= 0 or self.slice_batch_size >= slices.shape[0]:
            slice_logits = self.branch_2d(slices)
        else:
            chunks = []
            for start in range(0, slices.shape[0], self.slice_batch_size):
                chunks.append(self.branch_2d(slices[start : start + self.slice_batch_size]))
            slice_logits = torch.cat(chunks, dim=0)
        return self._slices_to_volume(slice_logits, volume_shape)

    def fuse_logits(self, logits_3d, logits_2d):
        if self.fusion_mode == "mean":
            return 0.5 * (logits_3d + logits_2d)
        if self.fusion_mode == "weighted":
            weights = torch.softmax(self.branch_logits, dim=0)
            return weights[0] * logits_3d + weights[1] * logits_2d
        return self.fuser(torch.cat([logits_3d, logits_2d], dim=1))

    def forward(self, x, return_aux=False):
        logits_3d = self.branch_3d(x)
        logits_2d = self._run_2d_branch(x)
        fused_logits = self.fuse_logits(logits_3d, logits_2d)
        if return_aux:
            return {
                "fused": fused_logits,
                "logits_3d": logits_3d,
                "logits_2d": logits_2d,
            }
        return fused_logits

    def pretrained_target(self):
        return self.branch_3d

    def load_swinunetr_3d_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        compatible = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[7:]
            target_key = key if key.startswith("branch_3d.") else f"branch_3d.{key}"
            if target_key in own_state and tuple(value.shape) == tuple(own_state[target_key].shape):
                compatible[target_key] = value

        if not compatible:
            raise RuntimeError("No compatible 3D SwinUNETR parameters were found for the dual model.")

        updated = dict(own_state)
        updated.update(compatible)
        self.load_state_dict(updated, strict=strict)
        return {"matched": len(compatible), "missing": len(own_state) - len(compatible)}


class CoarseToFineResidualSwinFusion(nn.Module):
    """
    Coarse-to-fine residual segmentation model.

    The 3D SwinUNETR branch produces a high-recall coarse probability volume.
    A light 2.5D Swin-Unet refiner then sees local CT slice context together
    with the coarse foreground prior and uncertainty map. The final logits are
    the coarse logits plus a bounded residual correction, so the fine branch is
    encouraged to repair missed/boundary regions without freely expanding the
    mask everywhere.
    """

    supports_aux_loss = True

    def __init__(
        self,
        in_channels=1,
        out_channels=2,
        feature_size_3d=48,
        feature_size_2d=24,
        use_checkpoint=True,
        plane="axial",
        slice_batch_size=16,
        roi_size=(96, 96, 96),
        context_slices=3,
        residual_scale=0.35,
        detach_coarse_prior=True,
        swin2d_depths=(2, 2, 2, 2),
        swin2d_heads=(3, 6, 12, 24),
        swin2d_window_size=6,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.plane = plane.lower()
        self.slice_batch_size = int(slice_batch_size)
        self.context_slices = max(1, int(context_slices))
        if self.context_slices % 2 == 0:
            self.context_slices += 1
        self.detach_coarse_prior = bool(detach_coarse_prior)

        if self.plane not in {"axial", "coronal", "sagittal"}:
            raise ValueError(f"Unsupported 2D plane: {plane}")

        self.branch_3d = _make_swinunetr(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size_3d,
            use_checkpoint=use_checkpoint,
            img_size=roi_size,
        )
        self.refiner_2d = SwinUnet2D(
            img_size=self._slice_size_for_plane(roi_size, self.plane),
            in_channels=self.context_slices + 2,
            num_classes=out_channels,
            embed_dim=feature_size_2d,
            depths=swin2d_depths,
            num_heads=swin2d_heads,
            window_size=swin2d_window_size,
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    @staticmethod
    def _slice_size_for_plane(roi_size, plane):
        depth, height, width = roi_size
        if plane == "axial":
            return (height, width)
        if plane == "coronal":
            return (depth, width)
        return (depth, height)

    def _volume_to_plane_stack(self, x):
        batch, channels, depth, height, width = x.shape
        if self.plane == "axial":
            stack = x.permute(0, 2, 1, 3, 4).contiguous()
            return stack, (batch, depth, height, width)
        if self.plane == "coronal":
            stack = x.permute(0, 3, 1, 2, 4).contiguous()
            return stack, (batch, depth, height, width)
        stack = x.permute(0, 4, 1, 2, 3).contiguous()
        return stack, (batch, depth, height, width)

    def _plane_stack_to_volume(self, y, volume_shape):
        batch, depth, height, width = volume_shape
        channels = y.shape[2]
        if self.plane == "axial":
            return y.reshape(batch, depth, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()
        if self.plane == "coronal":
            return y.reshape(batch, height, channels, depth, width).permute(0, 2, 3, 1, 4).contiguous()
        return y.reshape(batch, width, channels, depth, height).permute(0, 2, 3, 4, 1).contiguous()

    def _context_channels(self, stack):
        batch, slices, channels, height, width = stack.shape
        if channels != 1:
            raise ValueError("The 2.5D refiner currently expects a single CT image channel.")
        radius = self.context_slices // 2
        padded = F.pad(stack[:, :, 0], (0, 0, 0, 0, radius, radius), mode="replicate")
        context = []
        for offset in range(self.context_slices):
            context.append(padded[:, offset : offset + slices])
        return torch.stack(context, dim=2).reshape(batch * slices, self.context_slices, height, width)

    def _single_channel_slices(self, x):
        stack, _ = self._volume_to_plane_stack(x)
        batch, slices, channels, height, width = stack.shape
        if channels != 1:
            raise ValueError("Expected one channel when slicing coarse prior maps.")
        return stack.reshape(batch * slices, 1, height, width)

    def _run_refiner(self, image, coarse_prob, uncertainty):
        image_stack, volume_shape = self._volume_to_plane_stack(image)
        batch, slices, _, height, width = image_stack.shape
        image_context = self._context_channels(image_stack)
        coarse_slices = self._single_channel_slices(coarse_prob)
        uncertainty_slices = self._single_channel_slices(uncertainty)
        refiner_input = torch.cat([image_context, coarse_slices, uncertainty_slices], dim=1)

        if self.slice_batch_size <= 0 or self.slice_batch_size >= refiner_input.shape[0]:
            refined_slices = self.refiner_2d(refiner_input)
        else:
            chunks = []
            for start in range(0, refiner_input.shape[0], self.slice_batch_size):
                chunks.append(self.refiner_2d(refiner_input[start : start + self.slice_batch_size]))
            refined_slices = torch.cat(chunks, dim=0)

        refined_stack = refined_slices.reshape(batch, slices, self.out_channels, height, width)
        return self._plane_stack_to_volume(refined_stack, volume_shape)

    def forward(self, x, return_aux=False):
        logits_3d = self.branch_3d(x)
        coarse_prob = torch.softmax(logits_3d, dim=1)[:, 1:2]
        coarse_prior = coarse_prob.detach() if self.detach_coarse_prior else coarse_prob
        uncertainty = 1.0 - torch.abs(2.0 * coarse_prior - 1.0)
        logits_refine = self._run_refiner(x, coarse_prior, uncertainty)
        fused_logits = logits_3d + torch.tanh(self.residual_scale) * logits_refine

        if return_aux:
            return {
                "fused": fused_logits,
                "logits_3d": logits_3d,
                "logits_refine": logits_refine,
                "coarse_prob": coarse_prob,
                "uncertainty": uncertainty,
                "residual_scale": torch.tanh(self.residual_scale).detach(),
            }
        return fused_logits

    def pretrained_target(self):
        return self.branch_3d

    def load_swinunetr_3d_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        compatible = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[7:]
            target_key = key if key.startswith("branch_3d.") else f"branch_3d.{key}"
            if target_key in own_state and tuple(value.shape) == tuple(own_state[target_key].shape):
                compatible[target_key] = value

        if not compatible:
            raise RuntimeError("No compatible 3D SwinUNETR parameters were found for the coarse-to-fine model.")

        updated = dict(own_state)
        updated.update(compatible)
        self.load_state_dict(updated, strict=strict)
        return {"matched": len(compatible), "missing": len(own_state) - len(compatible)}
