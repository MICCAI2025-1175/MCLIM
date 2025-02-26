import datetime
import math
import sys
import time
import warnings
from functools import partial
from typing import List

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

import dist
from models import build_sparse_encoder, build_decoder
from sampler import DistInfiniteBatchSampler, worker_init_fn
from mclim import MCLIM
from utils import arg_util, misc, lamb
from utils.mri_v5 import build_dataset_to_pretrain, custom_list_data_collate
from utils.lr_control import lr_wd_annealing, get_param_groups

import SimpleITK as sitk


class LocalDDP(torch.nn.Module):
    def __init__(self, module):
        super(LocalDDP, self).__init__()
        self.module = module
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def main_pt():
    warnings.filterwarnings("ignore") 
    
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    print(f'initial args:\n{str(args)}')
    args.log_epoch()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # build data
    print(f'[build data for pre-training] ...\n')
    dataset_train = build_dataset_to_pretrain(args.data_path, args.input_size, args.mim_ratio, args.patch_size)
    data_loader_train = DataLoader(
        dataset=dataset_train, num_workers=args.dataloader_workers, pin_memory=True,
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(dataset_train), glb_batch_size=args.glb_batch_size,
            shuffle=True, filling=True, rank=dist.get_rank(), world_size=dist.get_world_size(),
        ), worker_init_fn=worker_init_fn, persistent_workers=True
    )
    itrt_train, iters_train = iter(data_loader_train), len(data_loader_train)
    print(f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size_per_gpu}, iters_train={iters_train}')
    
    model_without_ddp = MCLIM(rank=dist.get_rank(), world_size=dist.get_world_size()).to(args.device)
    print(f'[PT model] model = {model_without_ddp}\n')
    if dist.initialized():
        model: DistributedDataParallel = DistributedDataParallel(model_without_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=True, broadcast_buffers=False)
    else:
        model = LocalDDP(model_without_ddp)
    
    # build optimizer and lr_scheduler
    param_groups: List[dict] = get_param_groups(model_without_ddp)
    opt_clz = {
        'sgd': partial(torch.optim.SGD, momentum=0.9, nesterov=True),
        'adamw': partial(torch.optim.AdamW, betas=(0.9, args.ada)),
        'lamb': partial(lamb.TheSameAsTimmLAMB, betas=(0.9, args.ada), max_grad_norm=5.0),
    }[args.opt]
    optimizer = opt_clz(params=param_groups, lr=args.lr, weight_decay=0.0)
    print(f'[optimizer] optimizer({opt_clz}) ={optimizer}\n')
    
    # try to resume
    ep_start, performance_desc = misc.load_checkpoint(args.resume_from, model_without_ddp, optimizer)
    if ep_start >= args.ep: # load from a complete checkpoint file
        print(f'  [*] [PT already done]    Min/Last Loss: {performance_desc}')
    else:   # perform pre-training
        tb_lg = misc.TensorboardLogger(args.tb_lg_dir, is_master=dist.is_master(), prefix='pt')
        min_loss = 1e9
        print(f'[PT start] from ep{ep_start}')

        if args.amp:
            scaler = GradScaler()
        else:
            scaler = None
        
        pt_start_time = time.time()
        for ep in range(ep_start, args.ep):
            ep_start_time = time.time()
            tb_lg.set_step(ep * iters_train)
            if hasattr(itrt_train, 'set_epoch'):
                itrt_train.set_epoch(ep)
            
            stats = pre_train_one_ep(ep, args, tb_lg, itrt_train, iters_train, model, optimizer, scaler)
            last_loss = stats['last_loss']
            min_loss = min(min_loss, last_loss)
            performance_desc = f'{min_loss:.4f} {last_loss:.4f}'
            misc.save_checkpoint(f'{args.model}_still_pretraining.pth', args, ep, performance_desc, model_without_ddp.state_dict(), optimizer.state_dict())
            if ep % 50 == 0 and ep != 0:
                misc.save_checkpoint(f'{args.model}_{ep}.pth', args, ep, performance_desc, model_without_ddp.state_dict(), optimizer.state_dict())
            
            ep_cost = round(time.time() - ep_start_time, 2) + 1    # +1s: approximate the following logging cost
            remain_secs = (args.ep-1 - ep) * ep_cost
            remain_time = datetime.timedelta(seconds=round(remain_secs))
            finish_time = time.strftime("%m-%d %H:%M", time.localtime(time.time() + remain_secs))
            print(f'  [*] [ep{ep}/{args.ep}]    Min/Last Loss {performance_desc},    Cost: {ep_cost}s,    Remain: {remain_time},    Finish @ {finish_time}')
            
            args.cur_ep = f'{ep + 1}/{args.ep}'
            args.remain_time, args.finish_time = str(remain_time), str(finish_time)
            args.last_loss = last_loss
            args.log_epoch()
            
            tb_lg.update(min_loss=min_loss, head='train', step=ep)
            tb_lg.update(rest_hours=round(remain_secs/60/60, 2), head='z_burnout', step=ep)
            tb_lg.flush()
        
        # finish pre-training
        tb_lg.update(min_loss=min_loss, head='result', step=ep_start)
        tb_lg.update(min_loss=min_loss, head='result', step=args.ep)
        tb_lg.flush()
        print(f'final args:\n{str(args)}')
        print('\n\n')
        print(f'  [*] [PT finished]    Min/Last Loss: {performance_desc},    Total Cost: {(time.time() - pt_start_time) / 60 / 60:.1f}h\n')
        print('\n\n')
        tb_lg.close()
        time.sleep(10)
    
    args.remain_time, args.finish_time = '-', time.strftime("%m-%d %H:%M", time.localtime(time.time()))
    args.log_epoch()


def pre_train_one_ep(ep, args: arg_util.Args, tb_lg: misc.TensorboardLogger, itrt_train, iters_train, model: DistributedDataParallel, optimizer, scaler):
    model.train()
    me = misc.MetricLogger(delimiter='  ')
    me.add_meter('max_lr', misc.SmoothedValue(window_size=1, fmt='{value:.5f}'))
    header = f'[PT] Epoch {ep}:'
    
    warnings.filterwarnings("ignore") 
    optimizer.zero_grad()
    early_clipping = args.clip > 0 and not hasattr(optimizer, 'global_grad_norm')
    print('Early Clipping:', early_clipping)
    late_clipping = hasattr(optimizer, 'global_grad_norm')
    if early_clipping:
        params_req_grad = [p for p in model.parameters() if p.requires_grad]
    
    for it, (inp) in enumerate(me.log_every(iters_train, itrt_train, 3, header)):
        # adjust lr and wd
        min_lr, max_lr, min_wd, max_wd = lr_wd_annealing(optimizer, args.lr, args.wd, args.wde, it + ep * iters_train, args.wp_ep * iters_train, args.ep * iters_train)
        
        # forward and backward
        inp_t = torch.cat([t['image'] for t in inp], dim=0).to(args.device, non_blocking=True)
        label_t = torch.cat([t['label'] for t in inp], dim=0).to(args.device, non_blocking=True)
        if args.mim_ratio>0:
            mask_t = torch.cat([t['mask'] for t in inp], dim=0).to(args.device, non_blocking=True)
            mask_image_t = torch.cat([t['mask_image'] for t in inp], dim=0).to(args.device, non_blocking=True)
        else:
            mask_t = None
            mask_image_t = None

        with autocast(enabled=scaler is not None, dtype=torch.bfloat16):
            loss, clip_loss, match_loss, recon_im_loss = model(inp_t, label_t, mask_t, mask_image_t, args.weight_recon)
            grad_norm = None

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if early_clipping: grad_norm = torch.nn.utils.clip_grad_norm_(params_req_grad, args.clip).item()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if early_clipping: grad_norm = torch.nn.utils.clip_grad_norm_(params_req_grad, args.clip).item()
                optimizer.step()
                if late_clipping: grad_norm = optimizer.global_grad_norm

            loss = loss.item()
    
            optimizer.zero_grad()
            
            torch.cuda.synchronize()
        
        # log
        me.update(last_loss=loss)
        me.update(clip_loss=clip_loss)
        me.update(recon_im_loss=recon_im_loss)
        me.update(match_loss=match_loss)
        me.update(max_lr=max_lr)
        tb_lg.update(loss=me.meters['last_loss'].global_avg, head='train_loss')
        tb_lg.update(loss=me.meters['clip_loss'].global_avg, head='clip_loss')
        tb_lg.update(loss=me.meters['recon_im_loss'].global_avg, head='recon_im_loss')
        tb_lg.update(loss=me.meters['match_loss'].global_avg, head='match_loss')
        tb_lg.update(sche_lr=max_lr, head='train_hp/lr_max')
        tb_lg.update(sche_lr=min_lr, head='train_hp/lr_min')
        tb_lg.update(sche_wd=max_wd, head='train_hp/wd_max')
        tb_lg.update(sche_wd=min_wd, head='train_hp/wd_min')
        
        if grad_norm is not None:
            me.update(orig_norm=grad_norm)
            tb_lg.update(orig_norm=grad_norm, head='train_hp')
        tb_lg.set_step()
    
    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}


if __name__ == '__main__':
    main_pt()
