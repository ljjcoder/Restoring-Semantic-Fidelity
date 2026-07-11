import inspect
import math
from typing import Callable, List, Optional, Tuple, Union
from diffusers.models.attention_processor import Attention

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import os
from torchvision.transforms import GaussianBlur

import pdb
import pywt
import torch.fft as fft
import math
import random
from einops import rearrange, reduce, repeat

def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _infer_hw_from_image_tokens(image_token_len: int) -> Tuple[int, int]:
    # Mirror the heuristic used in this repo for non-square cases.
    s = int(math.isqrt(image_token_len))
    if s * s == image_token_len:
        return s, s
    h = int(math.sqrt(image_token_len / 2))
    w = int(math.sqrt(image_token_len * 2))
    if h * w != image_token_len:
        # Fallback: treat as 1D (will likely be wrong, but avoids crashing).
        return image_token_len, 1
    return h, w


def _resolve_image_hw(image_token_len: int, image_hw: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    if image_hw is not None:
        h, w = int(image_hw[0]), int(image_hw[1])
        if h * w == image_token_len:
            return h, w
    return _infer_hw_from_image_tokens(image_token_len)


class NPAAttention2D:
    """
    Patchify image tokens into local attention blocks (Q small, KV larger neighborhood),
    aligned with ScaleDiff NPA: only Q is padded (for jitter); K/V stay on original grid;
    text RoPE uses the image-token start position of each patch (KV patch start).
    """

    def __init__(self, q_patch_hw: int = 32, kv_patch_hw: int = 64, jitter: bool = False):
        self.q_patch_hw = q_patch_hw
        self.kv_patch_hw = kv_patch_hw
        self.query_random_jitter = True
        self.base_q_patch_len = 1024
        self.base_kv_patch_len = 4096
        self.base_height = 64
        self.base_width = 64
        # print( self.q_patch_hw,self.kv_patch_hw,self.jitter)
        # exit()

    def _compute_patch_specs(self, h: int, w: int, device: torch.device):
        ph = self.q_patch_hw
        pw = self.q_patch_hw
        kh = self.kv_patch_hw
        kw = self.kv_patch_hw
        # print(self.q_patch_hw,self.kv_patch_hw,self.jitter)
        # exit()
        # Jitter: only for Q, same as ScaleDiff (random offset up to ph/2, pw/2).
        if self.jitter:
            off_h = int(torch.randint(0, ph, (1,), device=device).item())
            off_w = int(torch.randint(0, pw, (1,), device=device).item())
            h_pad = h + off_h
            w_pad = w + off_w
        else:
            off_h, off_w = 0, 0
            h_pad, w_pad = h, w

        # Pad Q grid so it is divisible by (ph, pw) for patch coverage.
        pad_h = (_ceil_div(h_pad, ph) * ph) - h_pad
        pad_w = (_ceil_div(w_pad, pw) * pw) - w_pad
        h_pad += pad_h
        w_pad += pad_w

        num_ph = h_pad // ph
        num_pw = w_pad // pw
        num_patches = num_ph * num_pw

        return (ph, pw, kh, kw, off_h, off_w, h_pad, w_pad, num_ph, num_pw, num_patches)

    @torch.no_grad()
    def patchify(self, q, k, v, image_rotary_emb):
        B, H, L, C = q.shape
        self.device = q.device
        self.dtype = q.dtype
        image_rotary_emb_cos, image_rotary_emb_sin = image_rotary_emb
        C_emb = image_rotary_emb_cos.shape[1]
        assert q.shape == k.shape == v.shape
        assert self.height * self.width == L == image_rotary_emb_cos.shape[0] == image_rotary_emb_sin.shape[0]

        q = q.reshape(B, H, self.height, self.width, C)
        k = k.reshape(B, H, self.height, self.width, C)
        v = v.reshape(B, H, self.height, self.width, C)
        image_rotary_emb_cos = image_rotary_emb_cos.reshape(self.height, self.width, C_emb)
        image_rotary_emb_sin = image_rotary_emb_sin.reshape(self.height, self.width, C_emb)

        height = self.height
        width = self.width

        if self.query_random_jitter:
            q_pad = torch.zeros(B, H, self.height + self.base_height // 2, self.width + self.base_width // 2, C, device=self.device, dtype=self.dtype)
            random_h = random.randint(0, self.base_height // 2)
            random_w = random.randint(0, self.base_width // 2)
            q_pad[:, :, random_h:random_h + self.height, random_w:random_w + self.width, :] = q
            q = q_pad
            coord = (random_h, random_w)
            height += self.base_height // 2
            width += self.base_width // 2
        else:
            coord = (0, 0)
        
        assert height % (self.base_height // 2) == 0 and width % (self.base_width // 2) == 0

        num_patch_height = height // (self.base_height // 2)
        num_patch_width = width // (self.base_width // 2)
        num_total_patch = num_patch_height * num_patch_width

        q_patches = []
        k_patches = []
        v_patches = []
        txt_rotary_emb_cos = []
        txt_rotary_emb_sin = []
        
        for i in range(num_patch_height):
            for j in range(num_patch_width):
                h_start_q = i * (self.base_height // 2)
                w_start_q = j * (self.base_width // 2)

                h_start_kv = torch.clamp(torch.tensor(h_start_q - coord[0] - self.base_height // 4), 0, self.height - self.base_height).item()
                w_start_kv = torch.clamp(torch.tensor(w_start_q - coord[1] - self.base_width  // 4), 0, self.width  - self.base_width ).item()

                q_patches.append(q[:, :, h_start_q  : h_start_q +self.base_height // 2, w_start_q  : w_start_q +self.base_width // 2, :])
                k_patches.append(k[:, :, h_start_kv : h_start_kv+self.base_height     , w_start_kv : w_start_kv+self.base_width,      :])
                v_patches.append(v[:, :, h_start_kv : h_start_kv+self.base_height     , w_start_kv : w_start_kv+self.base_width,      :])
                txt_rotary_emb_cos.append(image_rotary_emb_cos[h_start_kv, w_start_kv, :])
                txt_rotary_emb_sin.append(image_rotary_emb_sin[h_start_kv, w_start_kv, :])

        q_patches = torch.cat(q_patches).reshape(B * num_total_patch, H, self.base_q_patch_len, C)
        k_patches = torch.cat(k_patches).reshape(B * num_total_patch, H, self.base_kv_patch_len, C)
        v_patches = torch.cat(v_patches).reshape(B * num_total_patch, H, self.base_kv_patch_len, C)
        txt_rotary_emb_cos = torch.stack(txt_rotary_emb_cos)
        txt_rotary_emb_sin = torch.stack(txt_rotary_emb_sin)

        return q_patches, k_patches, v_patches, (txt_rotary_emb_cos, txt_rotary_emb_sin), num_total_patch, coord


    @torch.no_grad()    
    def unpatchify(self, o_patches, num_total_patch, coord):
        B_prime, H, L_patch, C = o_patches.shape   
        assert L_patch == self.base_q_patch_len
        assert B_prime % num_total_patch == 0
        o_patches = o_patches.reshape(B_prime, H, self.base_height // 2, self.base_width // 2, C)
        o_patches = o_patches.chunk(num_total_patch)
        
        B = B_prime // num_total_patch
        height = self.height
        width = self.width
        if self.query_random_jitter:
            height += self.base_height // 2
            width += self.base_width // 2
        assert height % (self.base_height // 2) == 0 and width % (self.base_width // 2) == 0
        num_patch_height = height // (self.base_height // 2)
        num_patch_width = width // (self.base_width // 2)
        assert num_patch_height * num_patch_width == num_total_patch

        o = torch.zeros(B, H, height, width, C, device=o_patches[0].device, dtype=o_patches[0].dtype)

        for i in range(num_patch_height):
            for j in range(num_patch_width):
                h_start = i * (self.base_height // 2)
                w_start = j * (self.base_width // 2)

                o[:, :, h_start:h_start + self.base_height // 2, w_start:w_start + self.base_width // 2, :] = o_patches[i * num_patch_width + j]
        
        o = o[:, :, coord[0]:coord[0] + self.height, coord[1]:coord[1] + self.width, :]
        o = o.reshape(B, H, self.height * self.width, C)
        return o


def apply_rotary_emb_text_patches(
    x: torch.Tensor,
    txt_rotary_emb_cos: torch.Tensor,
    txt_rotary_emb_sin: torch.Tensor,
    num_patches: int,
) -> torch.Tensor:
    """
    Apply per-patch RoPE to text tokens. x: (B*np, H, T, D); cos/sin: (np, D).
    Each patch uses the RoPE at its KV start position (ScaleDiff style).
    """
    B_np, H, T, D = x.shape
    device = x.device
    dtype = x.dtype
    patch_idx = torch.arange(B_np, device=device) % num_patches
    cos = txt_rotary_emb_cos[patch_idx].to(device)
    sin = txt_rotary_emb_sin[patch_idx].to(device)
    cos = cos.view(B_np, 1, 1, -1).expand(B_np, H, T, D)
    sin = sin.view(B_np, 1, 1, -1).expand(B_np, H, T, D)
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(-2)
    out = (x.float() * cos + x_rotated.float() * sin).to(dtype)
    return out

def apply_rotary_emb_NPA(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    text = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.
    """
    if text:
        B, H, S, D = x.shape
        cos, sin = freqs_cis # [nv, D]
        b = B//cos.shape[0]
        cos = repeat(cos, 'nv D -> (b nv) 1 S D', b=b, S=S)
        sin = repeat(sin, 'nv D -> (b nv) 1 S D', b=b, S=S)
        cos, sin = cos.to(x.device), sin.to(x.device)
    else:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, H, S, D//2]
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)

    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

    return out

def get_views_local(
    height: int,
    width: int,
    h_window_size: int = 64,
    w_window_size: int = 64,
    random_jitter: bool = False,
) -> List[Tuple[int, int, int, int]]:
    """
    FreeScale-style overlapping window views for local attention.
    Stride = window_size // 2 (50% overlap).
    Returns list of (h_start, h_end, w_start, w_end).
    """
    height, width = int(height), int(width)

    h_window_stride = h_window_size // 2
    w_window_stride = w_window_size // 2
    # print(height, width,w_window_stride)

    num_blocks_height = int((height - h_window_size) / h_window_stride - 1e-6) + 2 if height > h_window_size else 1
    num_blocks_width = int((width - w_window_size) / w_window_stride - 1e-6) + 2 if width > w_window_size else 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    h_jitter_range = h_window_size // 8 if random_jitter else 0
    w_jitter_range = w_window_size // 8 if random_jitter else 0

    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * h_window_stride)
        h_end = h_start + h_window_size
        w_start = int((i % num_blocks_width) * w_window_stride)
        w_end = w_start + w_window_size

        if h_end > height:
            h_start = int(h_start + height - h_end)
            h_end = int(height)
        if w_end > width:
            w_start = int(w_start + width - w_end)
            w_end = int(width)
        if h_start < 0:
            h_end = int(h_end - h_start)
            h_start = 0
        if w_start < 0:
            w_end = int(w_end - w_start)
            w_start = 0

        if random_jitter:
            h_jitter = 0
            w_jitter = 0
            if (w_start != 0) and (w_end != width):
                w_jitter = random.randint(-w_jitter_range, w_jitter_range)
            elif (w_start == 0) and (w_end != width):
                w_jitter = random.randint(-w_jitter_range, 0)
            elif (w_start != 0) and (w_end == width):
                w_jitter = random.randint(0, w_jitter_range)
            if (h_start != 0) and (h_end != height):
                h_jitter = random.randint(-h_jitter_range, h_jitter_range)
            elif (h_start == 0) and (h_end != height):
                h_jitter = random.randint(-h_jitter_range, 0)
            elif (h_start != 0) and (h_end == height):
                h_jitter = random.randint(0, h_jitter_range)
            h_start += (h_jitter + h_jitter_range)
            h_end += (h_jitter + h_jitter_range)
            w_start += (w_jitter + w_jitter_range)
            w_end += (w_jitter + w_jitter_range)

        views.append((h_start, h_end, w_start, w_end))
    return views, h_jitter_range, w_jitter_range


def local_window_attention_2d(
    q_img: torch.Tensor,
    k_img: torch.Tensor,
    v_img: torch.Tensor,
    text_k: torch.Tensor,
    text_v: torch.Tensor,
    H_img: int,
    W_img: int,
    h_window_size: int = 64,
    w_window_size: int = 64,
    scale: float = None,
    random_jitter: bool = False,
) -> torch.Tensor:
    """
    FreeScale-style local attention: partition image into overlapping windows,
    run full self-attention within each window (image tokens attend to text + local image),
    average overlapping regions.
    q_img, k_img, v_img: [B, H, N_img, D]
    text_k, text_v: [B, H, T, D]
    """
    B, Hh, N_img, D = q_img.shape
    T = text_k.shape[2]
    device = q_img.device
    dtype = q_img.dtype

    views, h_jitter_range, w_jitter_range = get_views_local(
        H_img, W_img, h_window_size, w_window_size, random_jitter
    )

    q_2d = q_img.reshape(B, Hh, H_img, W_img, D)
    k_2d = k_img.reshape(B, Hh, H_img, W_img, D)
    v_2d = v_img.reshape(B, Hh, H_img, W_img, D)

    if random_jitter:
        q_2d = F.pad(q_2d, (0, 0, w_jitter_range, w_jitter_range, h_jitter_range, h_jitter_range), "constant", 0)
        k_2d = F.pad(k_2d, (0, 0, w_jitter_range, w_jitter_range, h_jitter_range, h_jitter_range), "constant", 0)
        v_2d = F.pad(v_2d, (0, 0, w_jitter_range, w_jitter_range, h_jitter_range, h_jitter_range), "constant", 0)

    value = torch.zeros_like(q_2d)
    count = torch.zeros_like(q_2d)
    # print(len(views))
    # if H_img>128:
    #     print(views)
    #     print(H_img, W_img)
    # exit()
    for h_start, h_end, w_start, w_end in views:
        q_win = q_2d[:, :, h_start:h_end, w_start:w_end, :]
        k_win = k_2d[:, :, h_start:h_end, w_start:w_end, :]
        v_win = v_2d[:, :, h_start:h_end, w_start:w_end, :]

        win_h, win_w = q_win.shape[2], q_win.shape[3]
        q_win = rearrange(q_win, "b h ph pw d -> b h (ph pw) d")
        k_win = rearrange(k_win, "b h ph pw d -> b h (ph pw) d")
        v_win = rearrange(v_win, "b h ph pw d -> b h (ph pw) d")

        k_full = torch.cat([text_k, k_win], dim=2)
        v_full = torch.cat([text_v, v_win], dim=2)

        local_out = F.scaled_dot_product_attention(
            q_win, k_full, v_full, dropout_p=0.0, is_causal=False, scale=scale
        )
        local_out = rearrange(local_out, "b h (ph pw) d -> b h ph pw d", ph=win_h, pw=win_w)

        value[:, :, h_start:h_end, w_start:w_end, :] += local_out
        count[:, :, h_start:h_end, w_start:w_end, :] += 1

    if random_jitter:
        value = value[:, :, h_jitter_range:-h_jitter_range, w_jitter_range:-w_jitter_range, :]
        count = count[:, :, h_jitter_range:-h_jitter_range, w_jitter_range:-w_jitter_range, :]

    out = torch.where(count > 0, value / count, value)
    return out.reshape(B, Hh, N_img, D)


def tvsda_attention_2d(
    q, k, v, text_k, text_v,
    H_img, W_img,
    window_size,
    scale,
    chunk_size=256,
):
    """
    window_size: full window size (e.g. 64), radius = window_size // 2
    """
    B, Hh, N, D = q.shape
    device = q.device
    out = torch.zeros_like(q)

    r = window_size // 2
    # print(H_img,W_img)
    # exit()

    # -------- 1. Cache 2D coords + index grid --------
    # This function is called for many layers/steps; caching saves overhead.
    cache_key = (int(H_img), int(W_img), str(device))
    if not hasattr(tvsda_attention_2d, "_cache"):
        tvsda_attention_2d._cache = {}
    cache = tvsda_attention_2d._cache
    if cache_key in cache:
        ys, xs, index_grid = cache[cache_key]
    else:
        coords = torch.arange(N, device=device)
        ys = coords // W_img          # [N]
        xs = coords % W_img           # [N]
        index_grid = torch.arange(N, device=device).reshape(int(H_img), int(W_img))
        cache[cache_key] = (ys, xs, index_grid)

    T = text_k.shape[2]  # text token length
    # print(text_k.shape)
    # exit()

    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        C = end - start

        q_chunk = q[:, :, start:end, :]      # [B,H,C,D]

        # -------- 3. query 坐标 --------
        qy = ys[start:end][:, None]           # [C,1]
        qx = xs[start:end][:, None]

        # -------- 4. chunk 的粗 key 范围（image 部分）--------
        ky_min = max(0, int(qy.min()) - r)
        ky_max = min(H_img - 1, int(qy.max()) + r)
        kx_min = max(0, int(qx.min()) - r)
        kx_max = min(W_img - 1, int(qx.max()) + r)

        # Use prebuilt 2D index grid slice to avoid per-chunk meshgrid/arange.
        img_key_indices = index_grid[ky_min : ky_max + 1, kx_min : kx_max + 1].reshape(-1)

        # Gather only needed image keys, then concatenate text keys (avoid building full_k/full_v).
        k_chunk_img = k[:, :, img_key_indices, :]
        v_chunk_img = v[:, :, img_key_indices, :]
        k_chunk = torch.cat([text_k, k_chunk_img], dim=2)   # [B,H,T+K_img,D]
        v_chunk = torch.cat([text_v, v_chunk_img], dim=2)

        # -------- 5. 构造 additive mask（关键优化点）--------
        ky = (img_key_indices // W_img)[None, :]   # [1,K_img]
        kx = (img_key_indices % W_img)[None, :]

        img_mask = (
            (ky >= qy - r) &
            (ky <= qy + r) &
            (kx >= qx - r) &
            (kx <= qx + r)
        )  # [C,K_img]
        # print(img_mask.shape)
        # exit()
        # text 永远可见
        text_mask = torch.ones((C, T), device=device, dtype=torch.bool)

        full_mask = torch.cat([text_mask, img_mask], dim=1)  # [C,K]

        # bool -> additive mask（FlashAttention 友好）
        attn_mask = torch.zeros_like(full_mask, dtype=q.dtype)
        attn_mask[~full_mask] = -1e4

        # -------- 6. 一次 SDPA --------
        out[:, :, start:end, :] = F.scaled_dot_product_attention(
            q_chunk,
            k_chunk,
            v_chunk,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )

    return out
# def sliding_window_attention_2d(
#     q, k, v, text_k, text_v,
#     H_img, W_img,
#     window_size,
#     scale,
#     chunk_size=256,
# ):
#     """
#     window_size: full window size (e.g. 64), radius = window_size // 2
#     """
#     B, Hh, N, D = q.shape
#     device = q.device
#     out = torch.zeros_like(q)

#     r = window_size // 2
#     # print(H_img,W_img)
#     # exit()

#     # -------- 1. Cache 2D coords + index grid --------
#     # This function is called for many layers/steps; caching saves overhead.
#     cache_key = (int(H_img), int(W_img), str(device))
#     if not hasattr(sliding_window_attention_2d, "_cache"):
#         sliding_window_attention_2d._cache = {}
#     cache = sliding_window_attention_2d._cache
#     if cache_key in cache:
#         ys, xs, index_grid = cache[cache_key]
#     else:
#         coords = torch.arange(N, device=device)
#         ys = coords // W_img          # [N]
#         xs = coords % W_img           # [N]
#         index_grid = torch.arange(N, device=device).reshape(int(H_img), int(W_img))
#         cache[cache_key] = (ys, xs, index_grid)

#     T = text_k.shape[2]  # text token length
#     # print(text_k.shape)
#     # exit()

#     for start in range(0, N, chunk_size):
#         end = min(N, start + chunk_size)
#         C = end - start

#         q_chunk = q[:, :, start:end, :]      # [B,H,C,D]

#         # -------- 3. query 坐标 --------
#         qy = ys[start:end][:, None]           # [C,1]
#         qx = xs[start:end][:, None]

#         # -------- 4. chunk 的粗 key 范围（image 部分）--------
#         ky_min = max(0, int(qy.min()) - r)
#         ky_max = min(H_img - 1, int(qy.max()) + r)
#         kx_min = max(0, int(qx.min()) - r)
#         kx_max = min(W_img - 1, int(qx.max()) + r)

#         # Use prebuilt 2D index grid slice to avoid per-chunk meshgrid/arange.
#         img_key_indices = index_grid[ky_min : ky_max + 1, kx_min : kx_max + 1].reshape(-1)

#         # Gather only needed image keys, then concatenate text keys (avoid building full_k/full_v).
#         k_chunk_img = k[:, :, img_key_indices, :]
#         v_chunk_img = v[:, :, img_key_indices, :]
#         k_chunk = torch.cat([text_k, k_chunk_img], dim=2)   # [B,H,T+K_img,D]
#         v_chunk = torch.cat([text_v, v_chunk_img], dim=2)

#         # -------- 5. 构造 additive mask（关键优化点）--------
#         ky = (img_key_indices // W_img)[None, :]   # [1,K_img]
#         kx = (img_key_indices % W_img)[None, :]

#         img_mask = (
#             (ky >= qy - r) &
#             (ky <= qy + r) &
#             (kx >= qx - r) &
#             (kx <= qx + r)
#         )  # [C,K_img]
#         # print(img_mask.shape)
#         # exit()
#         # text 永远可见
#         text_mask = torch.ones((C, T), device=device, dtype=torch.bool)

#         full_mask = torch.cat([text_mask, img_mask], dim=1)  # [C,K]

#         # bool -> additive mask（FlashAttention 友好）
#         attn_mask = torch.zeros_like(full_mask, dtype=q.dtype)
#         attn_mask[~full_mask] = -1e4

#         # -------- 6. 一次 SDPA --------
#         out[:, :, start:end, :] = F.scaled_dot_product_attention(
#             q_chunk,
#             k_chunk,
#             v_chunk,
#             attn_mask=attn_mask,
#             dropout_p=0.0,
#             is_causal=False,
#             scale=scale,
#         )

#     return out

class Custom_FluxAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self,control_params):
        self.use_tvsda_attention = control_params['use_tvsda_attention']
        self.use_npa_attention = control_params.get('use_npa_attention', False)
        self.use_local_attention = control_params.get('use_local_attention', False)
        self.local_window_h = control_params.get('local_window_h', 64)
        self.local_window_w = control_params.get('local_window_w', 64)
        self.local_random_jitter = control_params.get('local_random_jitter', False)
        self.image_hw = None
        self.text_seq_len = None
        # print('use_npa_attention:',self.use_npa_attention)
        # print(control_params.get("npa_q_patch_hw", 32))
        # print(control_params.get("npa_kv_patch_hw", 64))
        # exit()
        # NPA settings (image tokens only)
        self.npa = NPAAttention2D(
            q_patch_hw=control_params.get("npa_q_patch_hw", 32),
            kv_patch_hw=control_params.get("npa_kv_patch_hw", 64),
            jitter=control_params.get("npa_jitter", False),
        )
        # print('use_swin_attention:',self.use_swin_attention)
        # exit()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        proportional_attention = True,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # if encoder_hidden_states is None:
            # print('encoder_hidden_states: ',query.shape)
            # exit()
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
        
        if image_rotary_emb is not None and not getattr(self, "use_npa_attention", False):
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
            
        train_seq_len = 64 ** 2 + 512
        # print('train_seq_len: ',train_seq_len)
        # exit()
        # print('proportional_attention:',proportional_attention, math.sqrt((math.log(key.size(2), train_seq_len)) / head_dim))
        # exit()
        if proportional_attention:
            attention_scale = math.sqrt((math.log(key.size(2), train_seq_len)) / head_dim)
            # print(key.size(2),train_seq_len)
            # exit()
            # attention_scale_window = math.sqrt((math.log(key.size(2), train_seq_len)) / head_dim)
            attention_scale_window = math.sqrt(1 / head_dim)
            # print(attention_scale)
        else:
            attention_scale = math.sqrt(1 / head_dim)
        
        query_batch = False 
        if query_batch:
            query_batch_size = 256 ** 2
            query_batch_num = int((query.size(2) - 1e3) // query_batch_size + 1)
            hidden_states = []
            for qb in range(query_batch_num):
                query_batch = query[:, :, qb * query_batch_size: (qb + 1) * query_batch_size]
                hidden_states.append(F.scaled_dot_product_attention(query_batch, key, value, dropout_p=0.0, is_causal=False, scale=attention_scale))
            hidden_states = torch.cat(hidden_states, dim=2)
        else:
            # print(query.shape,key.shape)
            # exit()
            tvsda_attention = self.use_tvsda_attention
            npa_attention = self.use_npa_attention
            local_attention = self.use_local_attention
            # print('window: ',window_attention)
            # exit()

            if local_attention:
                # FreeScale-style local attention: overlapping windows, full attn per window, average overlap
                # print('local 2d')
                # print(image_rotary_emb is not None and not getattr(self, "use_npa_attention", False))
                # exit()
                text_len = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else (self.text_seq_len or 512)
                image_token_len = query.shape[2] - text_len
                h_latent, w_latent = _resolve_image_hw(image_token_len, self.image_hw)
                text_query = query[:, :, :text_len]
                text_hidden_states = F.scaled_dot_product_attention(
                    text_query, key, value, dropout_p=0.0, is_causal=False, scale=attention_scale
                )
                image_query = query[:, :, text_len:]
                text_key, text_value = key[:, :, :text_len], value[:, :, :text_len]
                image_key, image_value = key[:, :, text_len:], value[:, :, text_len:]
                # print(self.local_random_jitter)
                # exit()
                self.local_random_jitter =True
                image_hidden_states = local_window_attention_2d(
                    image_query, image_key, image_value, text_key, text_value,
                    h_latent, w_latent, self.local_window_h, self.local_window_w,
                    attention_scale, random_jitter=self.local_random_jitter,
                )
                # print(attention_scale_window)
                # exit()
                hidden_states = torch.cat([text_hidden_states, image_hidden_states], dim=2)
            elif npa_attention:
                # NPA: patchify image tokens (Q small, KV neighborhood); text RoPE = patch's image start position.
                text_len = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else (self.text_seq_len or 512)
                image_token_len = query.shape[2] - text_len
                h_latent, w_latent = _resolve_image_hw(image_token_len, self.image_hw)
                self.npa.height = h_latent
                self.npa.width = w_latent
                q_img = query[:, :, text_len:, :]
                k_img = key[:, :, text_len:, :]
                v_img = value[:, :, text_len:, :]
                query_text = query[:, :, :text_len, :]
                key_text = key[:, :, :text_len, :]
                value_text = value[:, :, :text_len, :]                
                pure_image_rotary_emb = [image_rotary_emb[0][ text_len:],image_rotary_emb[1][ text_len:]]
                q_img = apply_rotary_emb(q_img, pure_image_rotary_emb)
                k_img = apply_rotary_emb(k_img, pure_image_rotary_emb)

                query_image, key_image, value_image, txt_rotary_emb, num_total_patch, coord = self.npa.patchify(q_img, k_img, v_img, pure_image_rotary_emb)
                # print(coord)
                # exit()




                query_text = repeat(query_text, 'b h nt d -> (b np) h nt d', np=num_total_patch)
                key_text = repeat(key_text, 'b h nt d -> (b np) h nt d', np=num_total_patch)
                value_text = repeat(value_text, 'b h nt d -> (b np) h nt d', np=num_total_patch)

                query_text = apply_rotary_emb_NPA(query_text, txt_rotary_emb, text=True)
                key_text = apply_rotary_emb_NPA(key_text, txt_rotary_emb, text=True)


                query = torch.cat([query_text, query_image], dim=2)
                key = torch.cat([key_text, key_image], dim=2)
                value = torch.cat([value_text, value_image], dim=2)
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
                hidden_states_text, hidden_states_image = hidden_states[:, :, :text_len, :], hidden_states[:, :, text_len:, :]
                hidden_states_text = reduce(hidden_states_text, '(b np) c h w -> b c h w', 'mean', np=num_total_patch)
                hidden_states_image = self.npa.unpatchify(hidden_states_image, num_total_patch, coord)
                hidden_states = torch.cat([hidden_states_text, hidden_states_image], dim=2)
            elif tvsda_attention:
                # 修正2k_4K推理时候的bug
                text_len = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else (self.text_seq_len or 512)
                image_token_len = query.shape[2] - text_len
                h_latent, w_latent = _resolve_image_hw(image_token_len, self.image_hw)
                # print(h_latent,w_latent)
                # exit()
                # img_resolution = int(math.sqrt(image_token_len))
                # print(image_token_len,img_resolution)
                # exit()
                # print(img_resolution)
                # exit()
                # print(query.shape)
                # exit()
                text_query = query[:, :, :text_len]
                # print(text_query.shape)
                # exit() 
                text_hidden_states = F.scaled_dot_product_attention(text_query, key, value, dropout_p=0.0, is_causal=False, scale = attention_scale)

                image_query = query[:, :, text_len:]
                text_key, text_value = key[:, :, :text_len], value[:, :, :text_len]
                image_key, image_value = key[:, :, text_len:], value[:, :, text_len:]
                # image_text_hidden_states = F.scaled_dot_product_attention(image_query, text_key, text_value, dropout_p=0.0, is_causal=False, scale = 1)
                # image_image_hidden_states = F.scaled_dot_product_attention(image_query, image_key, image_value, dropout_p=0.0, is_causal=False, scale = attention_scale)
                # print(image_query.shape)
                # exit()
                image_image_hidden_states = tvsda_attention_2d(image_query, image_key, image_value, text_key, text_value,h_latent,w_latent,64,attention_scale,4096)          
                # hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False, scale = attention_scale)
                # print('finished split attention')
                # image_hidden_states = image_text_hidden_states+image_image_hidden_states
                image_hidden_states = image_image_hidden_states
                hidden_states = torch.cat([text_hidden_states,image_hidden_states],dim=2)
            else:
                # print('attention_scale: ',attention_scale)
                # exit()
                hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False, scale = attention_scale)

            # print(image_text_hidden_states.shape,image_image_hidden_states.shape)
            # exit()
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    # print('use_real:',use_real)
    # exit()
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None] # equal to .unsqueeze(0).unsqueeze(0)
        sin = sin[None, None]
        # print(cos.shape,sin.shape)
        # exit()
        cos, sin = cos.to(x.device), sin.to(x.device)
        # print(use_real_unbind_dim)
        # exit()
        # print(x.shape)
        # exit()
        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            # print(x_real.shape,x_imag.shape)
            # print(torch.stack([-x_imag, x_real], dim=-1).shape,torch.stack([-x_imag, x_real], dim=-1).flatten(2).shape)
            # exit()
            # exit()
            # torch.stack([-x_imag, x_real], dim=-1) shape is [1,24,17153,64,2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
            # print(x_rotated.shape)
            # print(torch.stack([-x_imag, x_real], dim=-1).shape)
            # exit(0)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")
        # print('kkk')
        # print(x.shape,cos.shape,x_rotated.shape)
        # exit()
        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        # used for lumina
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)

def prep_attn_processor(transformer,control_params):
    atten_name_2_processor = {}
    for name, module in transformer.named_modules():
        module_name = module.__class__.__name__  
        if name.split('.')[-1] == 'attn':
            atten_name_2_processor[name]=module.processor 
            attn_number = name.split('.')[-2]
            module.processor = Custom_FluxAttnProcessor2_0(control_params)
            module.processor.attn_number = attn_number
            # print(name,module.processor.attn_number,attn_number)
    # exit()
    return atten_name_2_processor
def reset_attn_processor(transformer,atten_name_2_processor):
    # atten_name_2_processor = {}
    for name, module in transformer.named_modules():
        module_name = module.__class__.__name__  
        if name.split('.')[-1] == 'attn':
            # atten_name_2_processor[name]=module.processor 
            attn_number = name.split('.')[-2]
            module.processor = atten_name_2_processor[name]
            # module.processor.attn_number = attn_number
            # print(name,module.processor.attn_number,attn_number)
    # exit()
    return atten_name_2_processor
def get_filter(shape, device, ratio = 0.75):
    h, w = shape[-2:]
    LPF = torch.zeros(shape).to(device)
    center_h, center_w = h // 2, w // 2
    region_size = (int(ratio * h), int(ratio * w))
    LPF[..., center_h-region_size[0]//2:center_h+region_size[0]//2, center_w-region_size[1]//2:center_w+region_size[1]//2] = 1
    return LPF

def butterworth_low_pass_filter_2d(shape, device, n=4, ratio=0.25):
    """
    Compute the Butterworth low-pass filter mask for a 2D image.

    Args:
        shape: (H, W) shape of the filter
        n: order of the filter, larger n ~ ideal, smaller n ~ gaussian
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
    """
    H, W = int(shape[-2]), int(shape[-1])
    d_s = float(ratio)

    # Cache by (H,W,order,ratio,device)
    cache_key = (H, W, int(n), round(d_s, 6), str(device))
    if not hasattr(butterworth_low_pass_filter_2d, "_cache"):
        butterworth_low_pass_filter_2d._cache = {}
    cache = butterworth_low_pass_filter_2d._cache
    if cache_key in cache:
        base = cache[cache_key]
        return base.expand(*shape).to(device)

    base = torch.zeros((H, W), device=device, dtype=torch.float32)
    if d_s == 0.0:
        cache[cache_key] = base
        return base.expand(*shape).to(device)

    # Vectorized grid in [-1, 1] range (approx), matching original formula.
    yy = (2.0 * torch.arange(H, device=device, dtype=torch.float32) / float(H) - 1.0) ** 2  # [H]
    xx = (2.0 * torch.arange(W, device=device, dtype=torch.float32) / float(W) - 1.0) ** 2  # [W]
    d_square = yy[:, None] + xx[None, :]  # [H,W]
    base = 1.0 / (1.0 + (d_square / (d_s**2)) ** int(n))

    cache[cache_key] = base
    return base.expand(*shape).to(device)

def set_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def gaussian_blur_image_sharpening(image, kernel_size=3, sigma=(0.1, 2.0), alpha=1):
    gaussian_blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    image_blurred = gaussian_blur(image)
    image_sharpened = (alpha + 1) * image - alpha * image_blurred

    return image_sharpened

def split_frequency_components_dwt(x, wavelet='haar', level=1):
    device = x.device
    dtype = x.dtype
    x = x.type(torch.float32)
    x = x.cpu().numpy()

    B, C, H, W = x.shape
    low_freq_components = []

    # Using list comprehension to improve performance
    for b in range(B):
        for c in range(C):
            coeffs = pywt.wavedec2(x[b, c], wavelet=wavelet, level=level)
            low_freq, *high_freq = coeffs
            low_freq_components.append([low_freq] + [(np.zeros_like(detail[0]), np.zeros_like(detail[1]), np.zeros_like(detail[2])) for detail in high_freq])

    # Convert list of numpy arrays to a single numpy array for better performance
    x_low_freq = np.stack([pywt.waverec2(low_freq_components[i], wavelet=wavelet) for i in range(B * C)])

    # Convert the numpy array to a tensor
    x_low_freq = torch.from_numpy(x_low_freq).view(B, C, H, W).type(dtype).to(device)

    return x_low_freq

def split_frequency_components_fft(x, freq_filter, is_low = True):
    x_freq = fft.fftshift(fft.fft2(x.to(freq_filter.dtype)))
    x_split_freq = x_freq * freq_filter if is_low else x_freq * (1 - freq_filter)
    x_split = fft.ifft2(fft.ifftshift(x_split_freq)).real
    return x_split
