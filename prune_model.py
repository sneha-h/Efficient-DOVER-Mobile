import torch
import torch.nn as nn
import argparse
import pickle as pkl

#import decord
import numpy as np
import yaml

import cv2
import random
import os.path as osp
import argparse
from scipy.stats import spearmanr, pearsonr
from scipy.stats.stats import kendalltau as kendallr
import numpy as np
from time import time
from tqdm import tqdm
import pickle
import math
import wandb
import yaml
import copy

import torch.nn.utils.prune as prune
import torch.optim as optim

from collections import OrderedDict
from matplotlib import pyplot as plt
from functools import reduce
from thop import profile

from gpustat import GPUStatCollection
import psutil
import time

import dover.models as models
import dover.datasets as datasets

from dover.models import DOVER

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

def get_model_macs(model, inputs) -> int:
    return profile_macs(model, inputs)


def get_sparsity(tensor: torch.Tensor) -> float:
    """
    calculate the sparsity of the given tensor
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    return 1 - float(tensor.count_nonzero()) / tensor.numel()


def get_model_sparsity(model: nn.Module) -> float:
    """
    calculate the sparsity of the given model
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    num_nonzeros, num_elements = 0, 0
    for param in model.parameters():
        num_nonzeros += param.count_nonzero()
        num_elements += param.numel()
    return 1 - float(num_nonzeros) / num_elements


def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    """
    calculate the total number of parameters of model
    :param count_nonzero_only: only count nonzero weights
    """
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements



def get_model_size(model: nn.Module, data_width=32, count_nonzero_only=False) -> int:
    """
    calculate the model size in bits
    :param data_width: #bits per element
    :param count_nonzero_only: only count nonzero weights
    """
    return get_num_parameters(model, count_nonzero_only) * data_width

def plot_weight_distribution(model, bins=256, count_nonzero_only=False):
    fig, axes = plt.subplots(3,3, figsize=(10, 6))
    axes = axes.ravel()
    plot_index = 0
    for name, param in model.named_parameters():
        if param.dim() > 1:
            ax = axes[plot_index]
            if count_nonzero_only:
                param_cpu = param.detach().view(-1).cpu()
                param_cpu = param_cpu[param_cpu != 0].view(-1)
                ax.hist(param_cpu, bins=bins, density=True,
                        color = 'blue', alpha = 0.5)
            else:
                ax.hist(param.detach().view(-1).cpu(), bins=bins, density=True,
                        color = 'blue', alpha = 0.5)
            ax.set_xlabel(name)
            ax.set_ylabel('density')
            plot_index += 1
    fig.suptitle('Histogram of Weights')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.show()

def plot_num_parameters_distribution(model):
    num_parameters = dict()
    for name, param in model.named_parameters():
        if param.dim() > 1:
            num_parameters[name] = param.numel()
    fig = plt.figure(figsize=(8, 6))
    plt.grid(axis='y')
    plt.bar(list(num_parameters.keys()), list(num_parameters.values()))
    plt.title('#Parameter Distribution')
    plt.ylabel('Number of Parameters')
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def sensitivity_scan(model, dataloader, scan_step=0.1, scan_start=0.4, scan_end=1.0, verbose=True):
    sparsities = np.arange(start=scan_start, stop=scan_end, step=scan_step)
    accuracies = []
    named_conv_weights = [(name, param) for (name, param) \
                          in model.named_parameters() if param.dim() > 1]
    for i_layer, (name, param) in enumerate(named_conv_weights):
        param_clone = param.detach().clone()
        accuracy = []
        for sparsity in tqdm(sparsities, desc=f'scanning {i_layer}/{len(named_conv_weights)} weight - {name}'):
            fine_grained_prune(param.detach(), sparsity=sparsity)
            acc = evaluate(model, dataloader, verbose=False)
            if verbose:
                print(f'\r    sparsity={sparsity:.2f}: accuracy={acc:.2f}%', end='')
            # restore
            param.copy_(param_clone)
            accuracy.append(acc)
        if verbose:
            print(f'\r    sparsity=[{",".join(["{:.2f}".format(x) for x in sparsities])}]: accuracy=[{", ".join(["{:.2f}%".format(x) for x in accuracy])}]', end='')
        accuracies.append(accuracy)
    return sparsities, accuracies

def train_test_split(dataset_path, ann_file, ratio=0.8, seed=42):
    random.seed(seed)
    print(seed)
    video_infos = []
    with open(ann_file, "r") as fin:
        for line in fin.readlines():
            line_split = line.strip().split(",")
            filename, _, _, label = line_split
            label = float(label)
            filename = osp.join(dataset_path, filename)
            video_infos.append(dict(filename=filename, label=label))
    random.shuffle(video_infos)
    return (
        video_infos[: int(ratio * len(video_infos))],
        video_infos[int(ratio * len(video_infos)) :],
    )


def rank_loss(y_pred, y):
    ranking_loss = torch.nn.functional.relu(
        (y_pred - y_pred.t()) * torch.sign((y.t() - y))
    )
    scale = 1 + torch.max(ranking_loss)
    return (
        torch.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0] - 1) / scale
    ).float()


def gaussian(y, eps=1e-8):
    return (y - y.mean()) / (y.std() + 1e-8)


def plcc_loss(y_pred, y):
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()


def rescaled_l2_loss(y_pred, y):
    y_pred_rs = (y_pred - y_pred.mean()) / y_pred.std()
    y_rs = (y - y.mean()) / (y.std() + eps)
    return torch.nn.functional.mse_loss(y_pred_rs, y_rs)


def rplcc_loss(y_pred, y, eps=1e-8):
    ## Literally (1 - PLCC) / 2
    y_pred, y = gaussian(y_pred), gaussian(y)
    cov = torch.sum(y_pred * y) / y_pred.shape[0]
    # std = (torch.std(y_pred) + eps) * (torch.std(y) + eps)
    return (1 - cov) / 2


def self_similarity_loss(f, f_hat, f_hat_detach=False):
    if f_hat_detach:
        f_hat = f_hat.detach()
    return 1 - torch.nn.functional.cosine_similarity(f, f_hat, dim=1).mean()


def contrastive_similarity_loss(f, f_hat, f_hat_detach=False, eps=1e-8):
    if f_hat_detach:
        f_hat = f_hat.detach()
    intra_similarity = torch.nn.functional.cosine_similarity(f, f_hat, dim=1).mean()
    cross_similarity = torch.nn.functional.cosine_similarity(f, f_hat, dim=0).mean()
    return (1 - intra_similarity) / (1 - cross_similarity + eps)


def rescale(pr, gt=None):
    if gt is None:
        pr = (pr - np.mean(pr)) / np.std(pr)
    else:
        pr = ((pr - np.mean(pr)) / np.std(pr)) * np.std(gt) + np.mean(gt)
    return pr


sample_types = ["aesthetic", "technical"]


def finetune_epoch(
    ft_loader,
    model,
    model_ema,
    optimizer,
    scheduler,
    device,
    epoch=-1,
    need_upsampled=False,
    need_feat=False,
    need_fused=False,
    need_separate_sup=True,
):
    model.train()
    for i, data in enumerate(tqdm(ft_loader, desc=f"Training in epoch {epoch}")):
        optimizer.zero_grad()
        video = {}
        for key in sample_types:
            if key in data:
                video[key] = data[key].to(device)

        y = data["gt_label"].float().detach().to(device).unsqueeze(-1)

        scores = model(video, inference=False, reduce_scores=False)
        if len(scores) > 1:
            y_pred = reduce(lambda x, y: x + y, scores)
        else:
            y_pred = scores[0]
        y_pred = y_pred.mean((-3, -2, -1))

        frame_inds = data["frame_inds"]


        loss = 0  # p_loss + 0.3 * r_loss

        if need_separate_sup:
            p_loss_a = plcc_loss(scores[0].mean((-3, -2, -1)), y)
            p_loss_b = plcc_loss(scores[1].mean((-3, -2, -1)), y)
            r_loss_a = rank_loss(scores[0].mean((-3, -2, -1)), y)
            r_loss_b = rank_loss(scores[1].mean((-3, -2, -1)), y)
            loss += (
                p_loss_a + p_loss_b + 0.3 * r_loss_a + 0.3 * r_loss_b
            )  # + 0.2 * o_loss
            wandb.log(
                {
                    "train/plcc_loss_a": p_loss_a.item(),
                    "train/plcc_loss_b": p_loss_b.item(),
                }
            )

        wandb.log(
            {"train/total_loss": loss.item(),}
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        # ft_loader.dataset.refresh_hypers()

        if model_ema is not None:
            model_params = dict(model.named_parameters())
            model_ema_params = dict(model_ema.named_parameters())
            for k in model_params.keys():
                model_ema_params[k].data.mul_(0.999).add_(
                    model_params[k].data, alpha=1 - 0.999
                )
    model.eval()


def profile_inference(inf_set, model, device):
    video = {}
    data = inf_set[0]
    for key in sample_types:
        if key in data:
            video[key] = data[key].to(device).unsqueeze(0)
    with torch.no_grad():

        flops, params = profile(model, (video,))
    print(
        f"The FLOps of the Variant is {flops/1e9:.1f}G, with Params {params/1e6:.2f}M."
    )


def inference_set(
    inf_loader,
    model,
    device,
    best_,
    save_model=False,
    suffix="s",
    save_name="divide",
    save_type="head",
):

    results = []

    best_s, best_p, best_k, best_r = best_

    for i, data in enumerate(tqdm(inf_loader, desc="Validating")):
        result = dict()
        video, video_up = {}, {}
        for key in sample_types:
            if key in data:
                video[key] = data[key].to(device)
                ## Reshape into clips
                b, c, t, h, w = video[key].shape
                video[key] = (
                    video[key]
                    .reshape(
                        b, c, data["num_clips"][key], t // data["num_clips"][key], h, w
                    )
                    .permute(0, 2, 1, 3, 4, 5)
                    .reshape(
                        b * data["num_clips"][key], c, t // data["num_clips"][key], h, w
                    )
                )
            if key + "_up" in data:
                video_up[key] = data[key + "_up"].to(device)
                ## Reshape into clips
                b, c, t, h, w = video_up[key].shape
                video_up[key] = (
                    video_up[key]
                    .reshape(b, c, data["num_clips"], t // data["num_clips"], h, w)
                    .permute(0, 2, 1, 3, 4, 5)
                    .reshape(b * data["num_clips"], c, t // data["num_clips"], h, w)
                )
            # .unsqueeze(0)
        with torch.no_grad():
            result["pr_labels"] = model(video, reduce_scores=True).cpu().numpy()
            if len(list(video_up.keys())) > 0:
                result["pr_labels_up"] = model(video_up).cpu().numpy()

        result["gt_label"] = data["gt_label"].item()
        del video, video_up
        results.append(result)

    ## generate the demo video for video quality localization
    gt_labels = [r["gt_label"] for r in results]
    pr_labels = [np.mean(r["pr_labels"][:]) for r in results]
    pr_labels = rescale(pr_labels, gt_labels)

    s = spearmanr(gt_labels, pr_labels)[0]
    p = pearsonr(gt_labels, pr_labels)[0]
    k = kendallr(gt_labels, pr_labels)[0]
    r = np.sqrt(((gt_labels - pr_labels) ** 2).mean())

    wandb.log(
        {
            f"val_{suffix}/SRCC-{suffix}": s,
            f"val_{suffix}/PLCC-{suffix}": p,
            f"val_{suffix}/KRCC-{suffix}": k,
            f"val_{suffix}/RMSE-{suffix}": r,
        }
    )

    del results, result  # , video, video_up
    torch.cuda.empty_cache()

    if s + p > best_s + best_p and save_model:
        state_dict = model.state_dict()

        if save_type == "head":
            head_state_dict = OrderedDict()
            for key, v in state_dict.items():
                if "head" in key:
                    head_state_dict[key] = v
            print("Following keys are saved (for head-only):", head_state_dict.keys())
            torch.save(
                {"state_dict": head_state_dict, "validation_results": best_,},
                f"pretrained_weights/{save_name}_{suffix}_finetuned.pth",
            )
        else:
            torch.save(
                {"state_dict": state_dict, "validation_results": best_,},
                f"pretrained_weights/{save_name}_{suffix}_finetuned.pth",
            )

    best_s, best_p, best_k, best_r = (
        max(best_s, s),
        max(best_p, p),
        max(best_k, k),
        min(best_r, r),
    )

    wandb.log(
        {
            f"val_{suffix}/best_SRCC-{suffix}": best_s,
            f"val_{suffix}/best_PLCC-{suffix}": best_p,
            f"val_{suffix}/best_KRCC-{suffix}": best_k,
            f"val_{suffix}/best_RMSE-{suffix}": best_r,
        }
    )

    print(
        f"For {len(inf_loader)} videos, \nthe accuracy of the model: [{suffix}] is as follows:\n  SROCC: {s:.4f} best: {best_s:.4f} \n  PLCC:  {p:.4f} best: {best_p:.4f}  \n  KROCC: {k:.4f} best: {best_k:.4f} \n  RMSE:  {r:.4f} best: {best_r:.4f}."
    )

    return best_s, best_p, best_k, best_r



def transfer_learning(model, opt, args):
    # code from transfer_learning to train model 
    target_set = "val-kv1k"
    device = 'cuda'
    bests_ = []

    if opt.get("split_seed", -1) > 0:
        num_splits = 10
    else:
        num_splits = 1

    print(opt["split_seed"])

    for split in range(10):
        #model = getattr(models, opt["model"]["type"])(**opt["model"]["args"]).to(device)
        if opt.get("split_seed", -1) > 0:
            opt["data"]["train"] = copy.deepcopy(opt["data"][target_set])
            opt["data"]["eval"] = copy.deepcopy(opt["data"][target_set])

            split_duo = train_test_split(
                opt["data"][target_set]["args"]["data_prefix"],
                opt["data"][target_set]["args"]["anno_file"],
                seed=opt["split_seed"] * (split + 1),
            )
            (
                opt["data"]["train"]["args"]["anno_file"],
                opt["data"]["eval"]["args"]["anno_file"],
            ) = split_duo
            opt["data"]["train"]["args"]["sample_types"]["technical"]["num_clips"] = 1

        train_datasets = {}
        for key in opt["data"]:
            if key.startswith("train"):
                train_dataset = getattr(datasets, opt["data"][key]["type"])(
                    opt["data"][key]["args"]
                )
                train_datasets[key] = train_dataset
                print(len(train_dataset.video_infos))

        train_loaders = {}
        for key, train_dataset in train_datasets.items():
            train_loaders[key] = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=opt["batch_size"],
                num_workers=opt["num_workers"],
                shuffle=True,
            )

        sensitivity_scan(model, train_loaders)
        val_datasets = {}
        for key in opt["data"]:
            if key.startswith("eval"):
                val_dataset = getattr(datasets, opt["data"][key]["type"])(
                    opt["data"][key]["args"]
                )
                print(len(val_dataset.video_infos))
                val_datasets[key] = val_dataset

        val_loaders = {}
        for key, val_dataset in val_datasets.items():
            val_loaders[key] = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=1,
                num_workers=opt["num_workers"],
                pin_memory=True,
            )
        sensitivity_scan(model, val_loaders)

        run = wandb.init(
            project=opt["wandb"]["project_name"],
            name=opt["name"] + f"_target_{target_set}_split_{split}"
            if num_splits > 1
            else opt["name"],
            reinit=True,
            settings=wandb.Settings(start_method="thread"),
        )

        #state_dict = torch.load(opt["test_load_path"], map_location=device)

        state_dict = model.state_dict()
        head_removed_state_dict = OrderedDict()
        for key, v in state_dict.items():
            if "head" not in key:
                head_removed_state_dict[key] = v

        # Allowing empty head weight
        #model.load_state_dict(state_dict, strict=False)
        model.load_state_dict(state_dict, strict=False)
        sparse_model_size = get_model_size(model, count_nonzero_only=True)
        sparse_model_parameters = get_num_parameters(model, count_nonzero_only=True)
        model_sparsity = get_model_sparsity(model)
        print(f"sparse - model size: {sparse_model_size / MiB:.2f}, number of parameters: {sparse_model_parameters} \
        model_sparsity - {model_sparsity}")
        if opt["ema"]:
            from copy import deepcopy

            model_ema = deepcopy(model)
        else:
            model_ema = None

        # profile_inference(val_dataset, model, device)

        # finetune the model

        param_groups = []

        for key, value in dict(model.named_children()).items():
            if "backbone" in key:
                param_groups += [
                    {
                        "params": value.parameters(),
                        "lr": opt["optimizer"]["lr"]
                        * opt["optimizer"]["backbone_lr_mult"],
                    }
                ]
            else:
                param_groups += [
                    {"params": value.parameters(), "lr": opt["optimizer"]["lr"]}
                ]

        optimizer = torch.optim.AdamW(
            lr=opt["optimizer"]["lr"],
            params=param_groups,
            weight_decay=opt["optimizer"]["wd"],
        )
        warmup_iter = 0
        for train_loader in train_loaders.values():
            warmup_iter += int(opt["warmup_epochs"] * len(train_loader))
        max_iter = int((opt["num_epochs"] + opt["l_num_epochs"]) * len(train_loader))
        lr_lambda = (
            lambda cur_iter: cur_iter / warmup_iter
            if cur_iter <= warmup_iter
            else 0.5 * (1 + math.cos(math.pi * (cur_iter - warmup_iter) / max_iter))
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda,)

        bests = {}
        bests_n = {}
        for key in val_loaders:
            bests[key] = -1, -1, -1, 1000
            bests_n[key] = -1, -1, -1, 1000

        for key, value in dict(model.named_children()).items():
            if "backbone" in key:
                for param in value.parameters():
                    param.requires_grad = False


        optimizer = torch.optim.AdamW(
            lr=opt["optimizer"]["lr"],
            params=param_groups,
            weight_decay=opt["optimizer"]["wd"],
        )
        warmup_iter = 0
        for train_loader in train_loaders.values():
            warmup_iter += int(opt["warmup_epochs"] * len(train_loader))
        max_iter = int((opt["num_epochs"] + opt["l_num_epochs"]) * len(train_loader))
        lr_lambda = (
            lambda cur_iter: cur_iter / warmup_iter
            if cur_iter <= warmup_iter
            else 0.5 * (1 + math.cos(math.pi * (cur_iter - warmup_iter) / max_iter))
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda,)

        bests = {}
        bests_n = {}
        for key in val_loaders:
            bests[key] = -1, -1, -1, 1000
            bests_n[key] = -1, -1, -1, 1000

        for key, value in dict(model.named_children()).items():
            if "backbone" in key:
                for param in value.parameters():
                    param.requires_grad = False
                    
        # Freeze pruned weights
        for name, param in model.named_parameters():
            if 'weight_orig' in name:
                print(f"Freeze: {name}")
                param.requires_grad = False

        for epoch in range(opt["l_num_epochs"]):
            print(f"Linear Epoch {epoch}:")
            start_time = time.time()
            cpu_usage_before = psutil.cpu_percent()
            gpu_stats_before = GPUStatCollection.new_query()

            for key, train_loader in train_loaders.items():
                finetune_epoch(
                    train_loader,
                    model,
                    model_ema,
                    optimizer,
                    scheduler,
                    device,
                    epoch,
                    opt.get("need_upsampled", False),
                    opt.get("need_feat", False),
                    opt.get("need_fused", False),
                )
            for key in val_loaders:
                bests[key] = inference_set(
                    val_loaders[key],
                    model_ema if model_ema is not None else model,
                    device,
                    bests[key],
                    save_model=opt["save_model"],
                    save_name=opt["name"] + "_head_" + target_set + f"_{split}",
                    suffix=key + "_s",
                )
                if model_ema is not None:
                    bests_n[key] = inference_set(
                        val_loaders[key],
                        model,
                        device,
                        bests_n[key],
                        save_model=opt["save_model"],
                        save_name=opt["name"]
                        + "_head_"
                        + target_set
                        + f"_{split}",
                        suffix=key + "_n",
                    )
                else:
                    bests_n[key] = bests[key]

            cpu_usage_after = psutil.cpu_percent()
            gpu_stats_after = GPUStatCollection.new_query()
            end_time = time.time()

            print(f"Epoch {epoch}: Time {end_time - start_time}s, \
            CPU Usage {cpu_usage_after - cpu_usage_before}%")
            for gpu_before, gpu_after in zip(gpu_stats_before, gpu_stats_after):
                print(f"GPU {gpu_before['index']} usage increased from \
                {gpu_before['utilization.gpu']}% to {gpu_after['utilization.gpu']}%")

        if opt["l_num_epochs"] >= 0:
            for key in val_loaders:
                print(
                    f"""For the linear transfer process on {key} with {len(val_loaders[key])} videos,
                    the best validation accuracy of the model-s is as follows:
                    SROCC: {bests[key][0]:.4f}
                    PLCC:  {bests[key][1]:.4f}
                    KROCC: {bests[key][2]:.4f}
                    RMSE:  {bests[key][3]:.4f}."""
                )

                print(
                    f"""For the linear transfer process on {key} with {len(val_loaders[key])} videos,
                    the best validation accuracy of the model-n is as follows:
                    SROCC: {bests_n[key][0]:.4f}
                    PLCC:  {bests_n[key][1]:.4f}
                    KROCC: {bests_n[key][2]:.4f}
                    RMSE:  {bests_n[key][3]:.4f}."""
                )

        for key, value in dict(model.named_children()).items():
            if "backbone" in key:
                for param in value.parameters():
                    param.requires_grad = True

        print("run finish")

        #run.finish()

def fine_grained_pruning(model, amount):
    cnt = 0
    for module_name, module in model.named_modules():
        #print(f"type - {type(module)}")
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear):
            cnt += 1
            #print(f"module name - {module_name}, {module.weight.nelement()}, count - {cnt}")
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')

def check_sparsity(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear):
            print(f"Sparcity in {name}.weight: \
            {100. * float(torch.sum(module.weight ==0)) / float(module.weight.nelement()):.3f}%")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o", "--opt", type=str, default="./dover.yml", help="the option file"
    )

    ## can be your own
    parser.add_argument(
        "-v",
        "--video_path",
        type=str,
        default="./demo/17734.mp4",
        help="the input video path",
    )

    parser.add_argument(
        "-d", "--device", type=str, default="cuda", help="the running device"
    )

    #parser.add_argument(
        #"-f", "--fusion", action="store_true",
    #)

    args = parser.parse_args()

    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)


    model = DOVER(**opt["model"]["args"]).to(args.device)
    checkpoint =  torch.load(opt["test_load_path"], map_location=args.device)
    model.load_state_dict(checkpoint)
    recover_model = lambda: model.load_state_dict(checkpoint)
    
    dense_model_size = get_model_size(model)
    model_parameters = get_num_parameters(model)
    
    print(f"model size: {dense_model_size / MiB:.2f}, number of parameters: {model_parameters}")
    #plot_weight_distribution(model)
    #plot_num_parameters_distribution(model)

    fine_grained_pruning(model, 0.5)
    #check_sparsity(model)
    sparse_model_size = get_model_size(model, count_nonzero_only=True)
    sparse_model_parameters = get_num_parameters(model, count_nonzero_only=True)
    model_sparsity = get_model_sparsity(model)
    print(f"sparse - model size: {sparse_model_size / MiB:.2f}, number of parameters: {sparse_model_parameters} \
    model_sparsity - {model_sparsity}")
    print(f"Sparse model has size={sparse_model_size / MiB:.2f} \
    MiB = {sparse_model_size / dense_model_size * 100:.2f}% of dense model size")


    #model.load_state_dict
        #torch.load("./pretrained_weights/DOVER.pth", map_location=args.device)
    #)

    #pruned_model_path = "pruned_model.pth"
    #torch.save(model.state_dict, pruned_model_path)
    

    transfer_learning(model, opt, args)