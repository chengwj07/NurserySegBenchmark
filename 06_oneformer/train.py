"""
06_oneformer/train.py

OneFormer training script (implemented based on HuggingFace transformers, because OneFormer
Not officially supported by mmsegmentation). By default, OneFormer-Swin-Large pre-trained by shi-labs is used
Weight (ADE20K) for fine-tuning multi-category semantic segmentation (default is 5 categories of seedlings, also supports two categories, see the number of categories/category names)
engine.py::CLASSES) - OneFormer has the largest weight and has been published on HuggingFace.
And the configuration can be used without additional native operators (natten).

Usage:
    # Start training from scratch
    python train.py --data_root /path/to/work_data_root --output_dir ./work_dirs/oneformer         [--epochs 50] [--batch_size 2] [--lr 1e-5]         [--pretrained shi-labs/oneformer_ade20k_swin_large]         [--amp] [--gpu_mem_fraction 0.4]

    # Resume training from breakpoint (automatically restore model/optimizer/scheduler/epoch from output_dir/last + trainer_state.pt)
    python train.py --data_root /path/to/work_data_root --output_dir ./work_dirs/oneformer --resume

Dependencies:
    pip install transformers accelerate torch torchvision pillow numpy

Only two weights are saved: output_dir/best (the highest mIoU in the verification set) and output_dir/last (the most recent epoch),
It does not save a copy for every epoch like earlier versions, and takes up less disk space.

After training, if you just want to use a certain checkpoint (such as best) to recalculate the indicators on the verification set,
There is no need to re-run train.py, it is faster to use eval.py in the same directory:
    python eval.py --checkpoint ./work_dirs/oneformer/best --data_root ...
"""
import argparse
import os
import sys
import time

# Shared Graphics: Allows the allocator to be closer to "what's actually used" rather than the large reserved cache.
# The available values of this environment variable vary with the PyTorch version and must be set before any CUDA allocation occurs.
# But `import torch` itself will not initialize the CUDA context, so the import here is only used to detect the version number.
import torch  # noqa: E402


def _set_default_alloc_conf():
    try:
        major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    except Exception:
        return  # If the version number parsing fails, no settings will be made, and PyTorch default behavior will be used without risking an error.
    if (major, minor) >= (2, 0):
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    elif (major, minor) >= (1, 10):
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    # PyTorch < 1.10 does not recognize this environment variable, skip it directly, and use the default allocator behavior


_set_default_alloc_conf()

from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.dirname(__file__))
from dataset import SegFolderDataset
from engine import (
    NUM_CLASSES,
    BEST_DIR_NAME,
    LAST_DIR_NAME,
    TRAINER_STATE_NAME,
    AvgMeter,
    collate_fn,
    evaluate,
    find_resume_checkpoint,
    format_eta,
    format_metrics,
    gpu_memory_mb,
    load_model_and_processor,
    load_trainer_state,
    log,
    move_to_device,
    save_checkpoint,
    save_trainer_state,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./work_dirs/oneformer")
    parser.add_argument(
        "--pretrained", type=str, default="shi-labs/oneformer_ade20k_swin_large",
        help='OneFormer pre-trained weight id on HuggingFace (only used when training from scratch; will be used when --resume'
             'Ignore this parameter and restore directly from the checkpoint in output_dir). Default Swin-Large '
             '(No additional dependencies required); if you need higher accuracy and can install natten, you can replace it with '
             "shi-labs/oneformer_ade20k_dinat_large",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument(
        "--resume", action="store_true",
        help='Resume training from breakpoint: automatically restore model weights, optimizer status, etc. from output_dir/last + trainer_state.pt'
             'Learning rate scheduler status, AMP scaler status, current epoch and historical optimal mIoU, followed by training.'
             'When there is no recoverable checkpoint in output_dir, an error will be reported and exited (to avoid mistakenly thinking that training has been continued when in fact it is'
             'start from scratch).',
    )
    parser.add_argument(
        "--log_interval", type=int, default=20,
        help='Print the training log every iter (mmseg style: lr/eta/time/data_time/memory/loss)',
    )
    parser.add_argument(
        "--gpu_mem_fraction", type=float, default=None,
        help='Limit the maximum proportion of GPU memory available to this process (0~1). It is recommended to set this when sharing the GPU.',
    )
    parser.add_argument(
        "--cudnn_benchmark", action="store_true",
        help='Turn on cudnn to automatically select the fastest convolution algorithm. It can speed up when using the exclusive GPU. It is turned off by default when sharing the GPU.'
             'Predictable memory usage proportional to batch_size',
    )
    parser.add_argument(
        "--amp", action="store_true",
        help='Turn on automatic mixed precision training (fp16 + GradScaler). Off by default (fp32); turning it on can significantly reduce GPU memory,'
             'Speed up, the effect is obvious on large models such as Swin-Large/DiNAT-Large',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    else:
        # Shared graphics card: Turn off benchmark to avoid unstable graphics memory/speed each time due to different remaining video memory on the graphics card at that time.
        torch.backends.cudnn.benchmark = False

    if args.gpu_mem_fraction is not None and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction, device=0)
        log(f"[Video memory limit] This process can use up to the current GPU memory {args.gpu_mem_fraction:.0%}")

    # --- Decide whether to start from scratch with pre-trained weights or continue training from the checkpoint in output_dir ---
    resume_model_dir, resume_state_path = (None, None)
    if args.resume:
        resume_model_dir, resume_state_path = find_resume_checkpoint(args.output_dir)
        if resume_model_dir is None:
            raise FileNotFoundError(
                f"--resume but in {args.output_dir} No recoverable checkpoint found "
                f"(Need to exist simultaneously {LAST_DIR_NAME}/ directory and {TRAINER_STATE_NAME} file)."
                f"If this is the first time training, remove the --resume parameter."
            )
        log(f"[Continued training] From {resume_model_dir} Restore model weights")

    model, processor = load_model_and_processor(args.pretrained, device, checkpoint=resume_model_dir)

    train_set = SegFolderDataset(args.data_root, "train", processor, args.image_size, num_classes=NUM_CLASSES)
    val_set = SegFolderDataset(args.data_root, "val", processor, args.image_size, num_classes=NUM_CLASSES)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                             num_workers=2, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * total_steps),
                                                 num_training_steps=total_steps)

    # AMP: Only meaningful on CUDA devices, GradScaler will be skipped on CPU (no operation when enabled=False)
    use_amp = args.amp and device == "cuda"
    if args.amp and device != "cuda":
        log('[AMP] Current device is not CUDA, ignore --amp, train as fp32')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        log('[AMP] Automatic mixed precision training (fp16 + GradScaler) turned on')

    start_epoch, global_step, best_miou = 0, 0, -1.0
    if resume_state_path is not None:
        start_epoch, global_step, best_miou = load_trainer_state(resume_state_path, optimizer,
                                                                   scheduler, scaler)
        log(f"[Continued training] The optimizer/scheduler state has been restored from epoch {start_epoch} continue "
            f"(Historical best mIoU={best_miou * 100:.2f})")

    total_iters_per_epoch = len(train_loader)
    total_iters_all = args.epochs * total_iters_per_epoch

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        iter_time_meter = AvgMeter()
        data_time_meter = AvgMeter()
        prev_end = time.time()

        for i, batch in enumerate(train_loader):
            data_time_meter.update(time.time() - prev_end, args.log_interval)

            model_inputs = move_to_device(batch, device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(**model_inputs)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1
            iter_time_meter.update(time.time() - prev_end, args.log_interval)
            prev_end = time.time()

            if (i + 1) % args.log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                eta = format_eta((total_iters_all - global_step) * iter_time_meter.avg)
                log(
                    f"Epoch(train) [{epoch + 1}][{i + 1}/{total_iters_per_epoch}]  "
                    f"lr: {lr:.4e}  eta: {eta}  time: {iter_time_meter.avg:.4f}  "
                    f"data_time: {data_time_meter.avg:.4f}  memory: {gpu_memory_mb()}  "
                    f"loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss / max(1, len(train_loader))
        log(f"Epoch(train) [{epoch + 1}] Completed, average loss: {avg_loss:.4f}")

        metrics = evaluate(model, processor, val_loader, device)
        log(f"Epoch(val) [{epoch + 1}]  loss: {metrics['loss']:.4f}  {format_metrics(metrics)}")

        # Only keep two weights: best and last
        save_checkpoint(model, processor, os.path.join(args.output_dir, LAST_DIR_NAME))
        if metrics["mIoU"] > best_miou:
            best_miou = metrics["mIoU"]
            save_checkpoint(model, processor, os.path.join(args.output_dir, BEST_DIR_NAME))
            log(f"Epoch(val) [{epoch + 1}] The new optimal model has been saved to {BEST_DIR_NAME}/ "
                f"(mIoU={best_miou * 100:.2f})")

        # Training status (optimizer/scheduler/scaler/epoch/best_miou) is updated every epoch,
        # With the weights in last/, --resume can truly resume training from the breakpoint instead of losing the optimizer momentum and restarting.
        save_trainer_state(
            os.path.join(args.output_dir, TRAINER_STATE_NAME),
            epoch=epoch, global_step=global_step, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, best_miou=best_miou,
        )

    log(f"Training completed. Optimal mIoU={best_miou * 100:.2f}, see weight "
        f"{os.path.join(args.output_dir, BEST_DIR_NAME)}")


if __name__ == "__main__":
    main()
