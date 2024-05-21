# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import math
import os
import time
from collections import defaultdict, deque
import datetime

import numpy as np
import torch
import torch.distributed as dist


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.ave = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.ave = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_k = correct[:1].view(-1).float()

    return correct_k


def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B * N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B * N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError


def get_index(idx, image_size=224, patch_size1=32, patch_size2=16):
    '''
    get index of fine stage corresponding to coarse stage 
    '''
    H1 = int(image_size / patch_size1)
    H2 = int(image_size / patch_size2)
    y = idx % H1
    idx1 = 4 * idx - 2 * y
    idx2 = idx1 + 1
    idx3 = idx1 + H2
    idx4 = idx3 + 1
    idx_finnal = torch.cat((idx1, idx2, idx3, idx4), dim=1)  # transformer对位置不敏感，位置随意
    return idx_finnal


def smooth_loss(disp):
    gamma = 10
    l1 = torch.abs(disp[:, :, :, :, :-1] - disp[:, :, :, :, 1:])
    l2 = torch.abs(disp[:, :, :, :-1, :] - disp[:, :, :, 1:, :])
    l3 = torch.abs(disp[:, :, :, 1:, :-1] - disp[:, :, :, :-1, 1:])
    l4 = torch.abs(disp[:, :, :, :-1, :-1] - disp[:, :, :, 1:, 1:])
    return (((l1.mean(dim=(-2, -1)) + l2.mean(dim=(-2, -1)) + l3.mean(dim=(-2, -1)) + l4.mean(dim=(-2, -1))) / 4).transpose(1, 2)).mean(dim=-1)


def generate(num: int, edge: int):
    '''
    :param num:  region_size
    :param edge: location_stage_size
    :return: regions
    '''
    information_results = []
    uninformation_results = []
    for i in range(num):
        for j in range(num):
            if (i + edge) <= num and (j + edge) <= num:
                temp1 = []
                temp2 = []
                for k in range(j + i * num, (j + i * num) + edge):
                    temp1.append(k)
                for k in range(0, j):
                    temp2.append(k + i * num)
                for k in range(j + edge + i * num, i * num + num):
                    temp2.append(k)

                temp1 = np.array(temp1)
                temp2 = np.array(temp2)
                temp_information = list(torch.tensor(temp1, dtype=int))
                temp_uninformation = list(torch.tensor(temp2, dtype=int))
                for m in range(1, edge):
                    temp_information.extend(torch.tensor(temp1 + num * m, dtype=int))
                    temp_uninformation.extend(torch.tensor(temp2 + num * m, dtype=int))

                for m in range(0, i):
                    temp_uninformation.extend(torch.tensor([h for h in range(num * m, num * m + num)], dtype=int))

                for m in range(i + edge, num):
                    temp_uninformation.extend(torch.tensor([h for h in range(num * m, num * m + num)], dtype=int))

                uninformation_results.append(torch.stack(temp_uninformation))
                information_results.append(torch.stack(temp_information))

    return information_results, uninformation_results


def block_information(batch_cls_attn, x, alpha, region_size, image_size=224, patch_size1=32, patch_size2=16):
    H1 = int(image_size / patch_size1)
    H2 = int(image_size / patch_size2)

    information_blocks, uninformation_blocks = generate(H1, region_size)

    information_blocks = torch.stack(information_blocks).to(torch.device('cuda'))
    uninformation_blocks = torch.stack(uninformation_blocks).to(torch.device('cuda'))

    regions_scores = []
    for block in information_blocks:
        regions_scores.append(batch_cls_attn[:, block].sum(dim=1))

    policy_indices = torch.argsort(torch.stack(regions_scores, dim=1), dim=1, descending=True)[:, :1]
    if x.size(0) == 1:
        # Important tokens in the original image
        important_indices = information_blocks[policy_indices.squeeze()].unsqueeze(0)
        # Unimportant tokens in the original image
        unimportant_indices = uninformation_blocks[policy_indices.squeeze()].unsqueeze(0)
    else:
        important_indices = information_blocks[policy_indices.squeeze()]
        unimportant_indices = uninformation_blocks[policy_indices.squeeze()]

    import_token_num = int(math.ceil(region_size * region_size * alpha))
    important_tokens = batch_index_select(x, important_indices + 1)
    unimportant_tokens = batch_index_select(x, unimportant_indices + 1)
    important_scores = torch.stack([cls[index] for cls, index in zip(batch_cls_attn, important_indices)])
    sorted_important_indices = torch.argsort(important_scores, dim=1, descending=True)
    new_unimportan_index = sorted_important_indices[:, import_token_num:]
    new_unimportan_tokens = batch_index_select(important_tokens, new_unimportan_index)

    important_tokens_indices = torch.stack([index[cls] for cls, index in zip(sorted_important_indices, important_indices)])
    new_important_index = important_tokens_indices[:, :import_token_num]
    final_unimportant_tokens = torch.cat((unimportant_tokens, new_unimportan_tokens), dim=1)

    y1 = new_important_index % H1
    idx1 = 4 * new_important_index - 2 * y1
    idx2 = idx1 + 1
    idx3 = idx1 + H2
    idx4 = idx3 + 1
    idx_finnal = torch.cat((idx1, idx2, idx3, idx4), dim=1)
    return idx_finnal, final_unimportant_tokens
