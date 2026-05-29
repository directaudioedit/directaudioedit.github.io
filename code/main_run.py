import argparse
import calendar
import matplotlib.pyplot as plt
import os
import time
import torch
import torchaudio
import warnings
import wandb
from torch import inference_mode

from ddm_inversion.inversion_utils import inversion_forward_process, inversion_reverse_process
from ddm_inversion.ddim_inversion import ddim_inversion, text2image_ldm_stable
from models import load_model
from utils import set_reproducability, load_audio
from tqdm import tqdm
import json

from utils import get_text_embeddings
from typing import List, Optional


def ddm_edit(args, save_name):
    args.eta = 1.
    args.numerical_fix = True
    args.x_prev_mode = False
    args.test_rand_gen = False

    set_reproducability(args.seed, extreme=False)
    cfg_scale_src = args.cfg_src
    cfg_scale_tar = args.cfg_tar


    save_path = os.path.join(args.results_path)
    os.makedirs(save_path, exist_ok=True)
    save_full_path_wave = os.path.join(save_path, save_name)



    eta = args.eta  # = 1
    if len(args.tstart) != len(args.target_prompt):
        if len(args.tstart) == 1:
            args.tstart *= len(args.target_prompt)
        else:
            raise ValueError("T-start amount and target prompt amount don't match.")
    args.tstart = torch.tensor(args.tstart, dtype=torch.int)
    skip = args.num_diffusion_steps - args.tstart

    # ldm_stable = load_model(model_id, device, args.num_diffusion_steps)
    x0 = load_audio(args.init_aud, ldm_stable.get_fn_STFT(), device=device)
    torch.cuda.empty_cache()
    with inference_mode():
        w0 = ldm_stable.vae_encode(x0)

    if args.mode == "ddim":
        if len(cfg_scale_src) > 1:
            raise ValueError("DDIM only supports one cfg_scale_src value")
        wT = ddim_inversion(ldm_stable, w0, args.source_prompt, cfg_scale_src[0],
                            num_inference_steps=args.num_diffusion_steps, skip=skip[0])
    elif args.mode == "ddpm":
        wt, zs, wts = inversion_forward_process(ldm_stable, w0, etas=eta,
                                                prompts=args.source_prompt, cfg_scales=cfg_scale_src,
                                                prog_bar=True,
                                                num_inference_steps=args.num_diffusion_steps,
                                                cutoff_points=args.cutoff_points,
                                                numerical_fix=args.numerical_fix,
                                                x_prev_mode=args.x_prev_mode)
    else:
        raise NotImplementedError



    if args.mode == "ddpm":
        # reverse process (via Zs and wT)
        w0, _ = inversion_reverse_process(ldm_stable,
                                          xT=wts if not args.test_rand_gen else torch.randn_like(wts),
                                          skips=args.num_diffusion_steps - skip,
                                          fix_alpha=args.fix_alpha,
                                          etas=eta, prompts=args.target_prompt,
                                          neg_prompts=args.target_neg_prompt,
                                          cfg_scales=cfg_scale_tar, prog_bar=True,
                                          zs=zs[:int(args.num_diffusion_steps - min(skip))]
                                          if not args.test_rand_gen else torch.randn_like(
                                              zs[:int(args.num_diffusion_steps - min(skip))]),
                                          #   zs=zs[skip:],
                                          cutoff_points=args.cutoff_points)
    
    elif args.mode == "ddim":  # ddim
        if skip != 0:
            warnings.warn("Plain DDIM Inversion should be run with t_start == num_diffusion_steps. "
                          "You are now running partial DDIM inversion.", RuntimeWarning)
        if len(cfg_scale_tar) > 1:
            raise ValueError("DDIM only supports one cfg_scale_tar value")
        if len(args.source_prompt) > 1:
            raise ValueError("DDIM only supports one args.source_prompt value")
        if len(args.target_prompt) > 1:
            raise ValueError("DDIM only supports one args.target_prompt value")
        

        w0 = text2image_ldm_stable(ldm_stable, args.target_prompt,
                               args.num_diffusion_steps, cfg_scale_tar[0],
                               wT,
                               skip=skip)

    else:
        raise NotImplementedError

    # vae decode image
    with inference_mode():
        x0_dec = ldm_stable.vae_decode(w0)
    if x0_dec.dim() < 4:
        x0_dec = x0_dec[None, :, :, :]

    with torch.no_grad():
        audio = ldm_stable.decode_to_mel(x0_dec)


    torchaudio.save(save_full_path_wave, audio, sample_rate=16000)





from typing import Tuple, Union, Optional, List


@torch.no_grad()
def text2image(ldm_model, prompt: List[str], num_inference_steps: int = 50,
                          guidance_scale: float = 7.5, xt: Optional[torch.FloatTensor] = None, skip: int = 0):

    # import pdb; pdb.set_trace()

    _, text_emb, uncond_emb = get_text_embeddings(prompt, [""], ldm_model)

    for t in tqdm(ldm_model.model.scheduler.timesteps[skip:]):
        noise_pred_uncond, _, _ = ldm_model.unet_forward(
                xt,
                timestep=t,
                encoder_hidden_states=uncond_emb.embedding_hidden_states,
                class_labels=uncond_emb.embedding_class_lables,
                encoder_attention_mask=uncond_emb.boolean_prompt_mask,
            )

        noise_prediction_text, _, _ = ldm_model.unet_forward(
                xt,
                timestep=t,
                encoder_hidden_states=text_emb.embedding_hidden_states,
                class_labels=text_emb.embedding_class_lables,
                encoder_attention_mask=text_emb.boolean_prompt_mask,
            )

        noise_pred = noise_pred_uncond.sample + guidance_scale * (noise_prediction_text.sample - noise_pred_uncond.sample)
        xt = ldm_model.model.scheduler.step(noise_pred, t, xt, eta=0).prev_sample

    return xt



def ldm_step_noise(ldm_model, xt, t, text_emb, uncond_emb, guidance_scale):
    noise_pred_uncond, _, _ = ldm_model.unet_forward(
                xt,
                timestep=t,
                encoder_hidden_states=uncond_emb.embedding_hidden_states,
                class_labels=uncond_emb.embedding_class_lables,
                encoder_attention_mask=uncond_emb.boolean_prompt_mask,
            )

    noise_prediction_text, _, _ = ldm_model.unet_forward(
                xt,
                timestep=t,
                encoder_hidden_states=text_emb.embedding_hidden_states,
                class_labels=text_emb.embedding_class_lables,
                encoder_attention_mask=text_emb.boolean_prompt_mask,
            )

    noise_pred = noise_pred_uncond.sample + guidance_scale * (noise_prediction_text.sample - noise_pred_uncond.sample)
    return noise_pred






def pad_to_size(tensor, target_size, dim=1):
    paddings = [0] * (2 * tensor.dim())
    rev_dim = tensor.dim() - 1 - dim
    paddings[rev_dim * 2 + 1] = max(0, target_size - tensor.shape[dim])
    
    fill_val = False if tensor.dtype == torch.bool else 0
    return torch.nn.functional.pad(tensor, tuple(paddings), value=fill_val)


def pad_and_cat(tensor1, tensor2):
    if tensor1 == None and tensor2 == None:
        return None
    
    L1 = tensor1.shape[1]
    L2 = tensor2.shape[1]


    assert tensor1.shape[0] == 1
    assert tensor2.shape[0] == 1
    
    if L1 == L2:
        return torch.cat([tensor1, tensor2])
    
    max_len = max(L1, L2)
    
    if L1 < max_len:
        tensor1 = pad_to_size(tensor1, max_len)
        
    if L2 < max_len:
        tensor2 = pad_to_size(tensor2, max_len)
        
    return torch.cat([tensor1, tensor2])





def ldm_step_noise_2(ldm_model, t, data_tar, data_src):
    xt_src, text_emb_src, uncond_emb_src, guidance_scale_src = data_src
    xt_tar, text_emb_tar, uncond_emb_tar, guidance_scale_tar = data_tar

    xt = torch.cat([xt_tar, xt_src])
    uncond_emb_embedding_hidden_states = pad_and_cat(uncond_emb_tar.embedding_hidden_states, uncond_emb_src.embedding_hidden_states)
    # import pdb; pdb.set_trace()
    uncond_emb_embedding_class_lables = pad_and_cat(uncond_emb_tar.embedding_class_lables, uncond_emb_src.embedding_class_lables)
    uncond_emb_boolean_prompt_mask = pad_and_cat(uncond_emb_tar.boolean_prompt_mask, uncond_emb_src.boolean_prompt_mask)
    text_emb_embedding_hidden_states = pad_and_cat(text_emb_tar.embedding_hidden_states, text_emb_src.embedding_hidden_states)
    text_emb_embedding_class_lables = pad_and_cat(text_emb_tar.embedding_class_lables, text_emb_src.embedding_class_lables)
    text_emb_boolean_prompt_mask = pad_and_cat(text_emb_tar.boolean_prompt_mask, text_emb_src.boolean_prompt_mask)


    # import pdb; pdb.set_trace()

    noise_pred_uncond, _, _ = ldm_model.unet_forward(
                xt,
                timestep=t,
                encoder_hidden_states=uncond_emb_embedding_hidden_states,
                class_labels=uncond_emb_embedding_class_lables,
                encoder_attention_mask=uncond_emb_boolean_prompt_mask,
            )

    noise_prediction_text, _, _ = ldm_model.unet_forward(
                xt,
                timestep=t,
                encoder_hidden_states=text_emb_embedding_hidden_states,
                class_labels=text_emb_embedding_class_lables,
                encoder_attention_mask=text_emb_boolean_prompt_mask,
            )

    noise_pred_tar = noise_pred_uncond.sample[:1] + guidance_scale_tar * (noise_prediction_text.sample[:1] - noise_pred_uncond.sample[:1])
    noise_pred_src = noise_pred_uncond.sample[1:] + guidance_scale_src * (noise_prediction_text.sample[1:] - noise_pred_uncond.sample[1:])
    return noise_pred_tar, noise_pred_src




def scale_noise(ldm_model, sample, timestep, noise):
    scheduler = ldm_model.model.scheduler
    alpha_t = scheduler.alphas_cumprod[timestep]
    zt_src = alpha_t**0.5 * sample + (1 - alpha_t)**0.5 * noise
    return zt_src    






@torch.no_grad()
def text2image_daedit_theory(ldm_model, prompt_src: List[str], prompt_tgt: List[str],
                          guidance_scale_src=3, guidance_scale_tar=8, x_src: Optional[torch.FloatTensor] = None, fix_noise=False, guidance_scale_strategy="cos"):
    
    _, src_emb, src_emb_uncond = get_text_embeddings(prompt_src, [""], ldm_model)
    _, tgt_emb, tgt_emb_uncond = get_text_embeddings(prompt_tgt, [""], ldm_model)

    zt_edit = x_src.clone()

    fwd_noise = torch.randn_like(x_src).to(x_src.device)
    
    for t in tqdm(ldm_model.model.scheduler.timesteps):
        current_guidance_scale_src = resolve_daedit_guidance_scale(
            guidance_scale_src,
            t,
            strategy=guidance_scale_strategy,
        )
        current_guidance_scale_tar = resolve_daedit_guidance_scale(
            guidance_scale_tar,
            t,
            strategy=guidance_scale_strategy,
        )


        if not fix_noise:
            fwd_noise = torch.randn_like(x_src).to(x_src.device)
        zt_src = scale_noise(ldm_model, x_src, t, fwd_noise)
        zt_tar = scale_noise(ldm_model, zt_edit, t, fwd_noise)
        
        eps_tgt, eps_src = ldm_step_noise_2(ldm_model, t, (zt_tar, tgt_emb, tgt_emb_uncond, current_guidance_scale_tar), (zt_src, src_emb, src_emb_uncond, current_guidance_scale_src))
        
        zt_src_prev = ldm_model.model.scheduler.step(eps_src, t, zt_src, eta=0).prev_sample
        zt_tar_prev = ldm_model.model.scheduler.step(eps_tgt, t, zt_tar, eta=0).prev_sample
        zt_src_delta = zt_src_prev - zt_src
        zt_tar_delta = zt_tar_prev - zt_tar
        zt_edit += zt_tar_delta - zt_src_delta



    return zt_edit






import math
def cal_scale(t, w_min, w_max):
    current_w = w_min + (w_max - w_min) * math.cos((math.pi / 2) * t)
    return current_w


def cal_scale_with_strategy(t, w_min, w_max, strategy="cos"):
    t = max(0.0, min(1.0, float(t)))

    if strategy == "linear":
        weight = t
    elif strategy == "cos":
        weight = 1 - math.cos((math.pi / 2) * t)
    else:
        raise ValueError(f"Unknown daedit guidance strategy: {strategy}")

    return w_min + (w_max - w_min) * weight


def cal_scale_piecewise(t, guidance_scales, strategy="cos"):
    if len(guidance_scales) == 1:
        return guidance_scales[0]

    t = max(0.0, min(1.0, float(t)))
    num_segments = len(guidance_scales) - 1

    if t >= 1.0:
        return guidance_scales[-1]

    segment_width = 1.0 / num_segments
    segment_idx = min(int(t / segment_width), num_segments - 1)
    segment_t_start = segment_idx * segment_width
    local_t = (t - segment_t_start) / segment_width

    return cal_scale_with_strategy(
        local_t,
        guidance_scales[segment_idx],
        guidance_scales[segment_idx + 1],
        strategy=strategy,
    )


def resolve_daedit_guidance_scale(guidance_scale, timestep, strategy="cos"):
    if isinstance(guidance_scale, (list, tuple)):
        if len(guidance_scale) >= 1:
            return cal_scale_piecewise(1 - timestep / 1000, guidance_scale, strategy=strategy)
    return guidance_scale



def w2audio(w):
    # vae decode image
    with inference_mode():
        x0_dec = ldm_stable.vae_decode(w)
    if x0_dec.dim() < 4:
        x0_dec = x0_dec[None, :, :, :]

    with torch.no_grad():
        audio = ldm_stable.decode_to_mel(x0_dec)

    return audio



def daedit(args, save_name):
    args.eta = 1.
    args.numerical_fix = True
    args.x_prev_mode = False
    args.test_rand_gen = False

    set_reproducability(args.seed, extreme=False)

    cfg_scale_src = args.cfg_src
    cfg_scale_tar = args.cfg_tar


    save_path = os.path.join(args.results_path)
    os.makedirs(save_path, exist_ok=True)
    save_full_path_wave = os.path.join(save_path, save_name)



    eta = args.eta  # = 1
    if len(args.tstart) != len(args.target_prompt):
        if len(args.tstart) == 1:
            args.tstart *= len(args.target_prompt)
        else:
            raise ValueError("T-start amount and target prompt amount don't match.")
    args.tstart = torch.tensor(args.tstart, dtype=torch.int)
    skip = args.num_diffusion_steps - args.tstart

    x0 = load_audio(args.init_aud, ldm_stable.get_fn_STFT(), device=device)
    torch.cuda.empty_cache()
    with inference_mode():
        w0 = ldm_stable.vae_encode(x0)
        w0_src = w0


    if skip != 0:
        warnings.warn("Plain DDIM Inversion should be run with t_start == num_diffusion_steps. "
                      "You are now running partial DDIM inversion.", RuntimeWarning)


    basescale_cfg_src = cfg_scale_src if len(cfg_scale_src) > 1 else cfg_scale_src[0]
    basescale_cfg_tar = cfg_scale_tar if len(cfg_scale_tar) > 1 else cfg_scale_tar[0]
    w0_daedit = text2image_daedit_theory(ldm_stable, args.source_prompt, args.target_prompt,
                           basescale_cfg_src, basescale_cfg_tar,
                           w0_src,
                           fix_noise=args.fixnoise,
                           guidance_scale_strategy=args.daedit_guidance_strategy)


    audio_daedit = w2audio(w0_daedit)


    torchaudio.save(f"{save_full_path_wave}", audio_daedit, sample_rate=16000)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run text-based audio editing.')

    parser.add_argument("--interval", type=int, default=0)
    parser.add_argument("--device_num", type=int, default=0, help="GPU device number")
    parser.add_argument('-s', "--seed", type=int, default=42, help="GPU device number")
    parser.add_argument("--model_id", type=str, choices=["cvssp/audioldm2",
                                                         "cvssp/audioldm2-music",
                                                         "declare-lab/tango2-full"],
                        default="declare-lab/tango2-full", help='Audio diffusion model to use')

    parser.add_argument("--init_aud", type=str, required=True, help='Audio to invert and extract PCs from')
    parser.add_argument("--cfg_src", type=float, nargs='+', default=[3],
                        help='Classifier-free guidance strength for forward process. Supports one value or multiple control points for piecewise scheduling.')
    parser.add_argument("--cfg_tar", type=float, nargs='+', default=[8],
                        help='Classifier-free guidance strength for reverse process. Supports one value or multiple control points for piecewise scheduling, e.g. 4 12 4 for low-high-low.')
    parser.add_argument("--num_diffusion_steps", type=int, default=50,
                        help="Number of diffusion steps. TANGO and AudioLDM2 are recommended to be used with 200 steps"
                             ", while AudioLDM is recommeneded to be used with 100 steps")
    parser.add_argument("--target_prompt", type=str, nargs='+', default=[""], required=True,
                        help="Prompt to accompany the reverse process. Should describe the wanted edited audio.")
    parser.add_argument("--source_prompt", type=str, nargs='+', default=[""],
                        help="Prompt to accompany the forward process. Should describe the original audio.")
    parser.add_argument("--target_neg_prompt", type=str, nargs='+', default=[""],
                        help="Negative prompt to accompany the inversion and generation process")
    parser.add_argument("--tstart", type=int, nargs='+', default=[50],
                        help="Diffusion timestep to start the reverse process from. Controls editing strength.")
    parser.add_argument("--results_path", type=str, default="results", help="path to dump results")

    parser.add_argument("--cutoff_points", type=float, nargs='*', default=None)
    parser.add_argument("--mode", default="direct_audio_edit", choices=['ddim', 'ddpm', 'direct_audio_edit'],
                        help="Run our editing or DDIM inversion based editing.")
    parser.add_argument("--fix_alpha", type=float, default=0.1)

    parser.add_argument('--fixnoise', action='store_true')

    parser.add_argument('--daedit_guidance_strategy', type=str,
                        choices=['cos', 'linear'],
                        default='cos',
                        help='Interpolation strategy used between consecutive CFG control points when --cfg_src/--cfg_tar provide multiple values.')


    parser.add_argument('--n_avg', type=int, default=1, help='Number of noise samples to average over when computing the CVC correction. Higher values can improve stability at the cost of speed.')

    parser.add_argument('--fe_ver', type=str, default="v1", help='Version of daedit to run. 1 for the version in the original paper, 2 for the latest version with improved stability and optional alignment regularization.')

    args = parser.parse_args()



    device = f"cuda:{args.device_num}"
    torch.cuda.set_device(args.device_num)

    model_id = args.model_id

    ldm_stable = load_model(model_id, device, args.num_diffusion_steps)


    name = os.path.split(args.init_aud)[1]
    if args.mode == 'direct_audio_edit':
        daedit(args, save_name=name)
    else:
        ddm_edit(args, save_name=name)

