import argparse
import os
import datetime
import numpy as np
import torch
import torchvision
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import imageio.v2 as imageio

from utils import Nptranspose, Rotation, H_Mirror, V_Mirror
from datasets import TrainDataset
import trainer
from network.Net import Resnet50


# ----------------------------
# Helpers
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()

    # Mode / IO
    p.add_argument("--mode", default="train", choices=["train", "eval", "infer"])
    p.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint (.pkl from torch.save(model, ...))")
    p.add_argument("--ckpt_dir", type=str, default="./model", help="Directory to save checkpoints during training")
    p.add_argument("--pred_dir", type=str, default="./predictions", help="Directory to save inference images")


    # Data
    p.add_argument("--train_image_file", "--train_image", default="../dataset/train-augment/image/", type=str)
    p.add_argument("--train_label_file", "--train_label", default="../dataset/train-augment/dsm/", type=str)
    p.add_argument("--test_image_file", "--test_image", default="../dataset/evensmallertest/image/", type=str)
    p.add_argument("--test_label_file", "--test_label", default="../dataset/evensmallertest/dsm/", type=str)

    # Training
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epoch", type=int, default=0)
    p.add_argument("--num_epochs", "--start_epoch", type=int, default=30)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=123)

    return p.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloader(image_dir, label_dir, mode, batch_size, num_workers):
    if mode == "train":
        tfm = torchvision.transforms.Compose([Rotation(), H_Mirror(), V_Mirror(), Nptranspose()])
        ds = TrainDataset(image_dir, label_dir, tfm)
        return DataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        tfm = torchvision.transforms.Compose([Nptranspose()])
        ds = TrainDataset(image_dir, label_dir, tfm)
        return DataLoader(
            dataset=ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

def extract_state_dict(ckpt):
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unexpected checkpoint type: {type(ckpt)}")

    # common nested keys first
    for key in ["model", "model_state", "state_dict", "model_state_dict", "net"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]

    # raw state_dict fallback
    if all(isinstance(k, str) for k in ckpt.keys()):
        return ckpt

    raise KeyError(f"Could not find state_dict. Top-level keys: {list(ckpt.keys())}")

def load_model(args, device):
    model = Resnet50().to(device)

    if args.mode in ["eval", "infer"]:
        if not args.ckpt:
            raise ValueError("Provide --ckpt path/to/ckpt.pt")

        ckpt = torch.load(args.ckpt, map_location=device)
        state_dict = extract_state_dict(ckpt)

        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        print("Missing:", missing[:20])
        print("Unexpected:", unexpected[:20])

        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint mismatch. Missing={len(missing)}, Unexpected={len(unexpected)}"
            )

        model.eval()

    return model


@torch.no_grad()
def eval_only(model, dataloader, criterion, device):
    from cal_acc import cal_psnr, cal_ssim, cal_mae, cal_rmse, cal_zncc

    model.eval()
    totals = dict(loss=0.0, psnr=0.0, ssim=0.0, mae=0.0, rmse=0.0, zncc=0.0)
    n = 0

    for sample in dataloader:
        image = sample["image"].to(torch.float32).to(device)
        label = sample["label"].to(torch.float32).to(device)

        output, _, _, _ = model(image)

        totals["loss"] += criterion(output, label).item()
        totals["psnr"] += cal_psnr(output, label).item()
        totals["ssim"] += float(cal_ssim(output, label))
        totals["mae"]  += float(cal_mae(output, label))
        totals["rmse"] += float(cal_rmse(output, label))
        totals["zncc"] += float(cal_zncc(output, label))
        n += 1

    for k in totals:
        totals[k] /= max(n, 1)

    print(
        f"eval_loss={totals['loss']:.6f}, "
        f"eval_psnr={totals['psnr']:.4f}, "
        f"eval_ssim={totals['ssim']:.4f}, "
        f"eval_mae={totals['mae']:.4f}, "
        f"eval_rmse={totals['rmse']:.4f}, "
        f"eval_zncc={totals['zncc']:.4f}"
    )


@torch.no_grad()
def run_inference(model, dataloader, device, pred_dir):
    os.makedirs(pred_dir, exist_ok=True)
    model.eval()

    for idx, sample in enumerate(dataloader):
        image = sample["image"].to(torch.float32).to(device)
        output, _, _, _ = model(image)  # [B,1,H,W]
        out_np = output.detach().cpu().numpy()

        for b in range(out_np.shape[0]):
            pred = out_np[b, 0]  # [H,W]

            # 16-bit PNG, per-image normalization (visualization)
            pmin, pmax = float(pred.min()), float(pred.max())
            if pmax > pmin:
                pred_u16 = ((pred - pmin) / (pmax - pmin) * 65535.0).astype(np.uint16)
            else:
                pred_u16 = np.zeros_like(pred, dtype=np.uint16)

            imageio.imwrite(os.path.join(pred_dir, f"pred_{idx:05d}_{b}.png"), pred_u16)

    print(f"Saved predictions to: {pred_dir}")



def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = load_model(args, device)
    criterion = nn.MSELoss()

    start = datetime.datetime.now()

    if args.mode == "train":
        train_loader = build_dataloader(args.train_image_file, args.train_label_file,
                                        mode="train", batch_size=args.batch_size, num_workers=args.num_workers)
        test_loader = build_dataloader(args.test_image_file, args.test_label_file,
                                       mode="test", batch_size=1, num_workers=0)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # Note: you create scheduler but trainer.py doesn't use it; keep if you later wire it in.
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

        print("len(train_dataloader)", len(train_loader))
        print("len(eval_dataloader)", len(test_loader))
        print("begin training!")
        trainer.train_model(model, args, train_loader, test_loader, criterion, optimizer, device)

    elif args.mode == "eval":
        test_loader = build_dataloader(args.test_image_file, args.test_label_file,
                                       mode="test", batch_size=1, num_workers=0)
        print("len(eval_dataloader)", len(test_loader))
        print("begin eval only!")
        eval_only(model, test_loader, criterion, device)

    elif args.mode == "infer":
        test_loader = build_dataloader(args.test_image_file, args.test_label_file,
                                       mode="test", batch_size=1, num_workers=0)
        print("len(infer_dataloader)", len(test_loader))
        print("begin inference only!")
        run_inference(model, test_loader, device, args.pred_dir)

    end = datetime.datetime.now()
    print("run time {}".format(end - start))


if __name__ == "__main__":
    main()
