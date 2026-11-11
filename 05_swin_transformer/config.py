"""
05_swin_transformer/config.py
Swin Transformer (Swin-Large, window12_384) + UpperNet segmentation configuration (default 5 categories of seedlings, see common/plant_dataset.py; if you need to switch back to two categories, see the root directory README)
——The standard configuration with the largest number of parameters and the highest accuracy among the mmseg Swin series
Backbone: SwinTransformer (window self-focus + shift window)
Decode Head: UPerHead (PPM + FPN multi-scale fusion)
Auxiliary Head: FCN Head (deep supervision)
"""
custom_imports = dict(imports=["common.plant_dataset", "common.seg_loss"], allow_failed_imports=False)

dataset_type = "PlantSegDataset"
data_root = "REPLACE_AT_RUNTIME"
crop_size = (1024, 1024)
# Category imbalance (background is about 84%, Ligustrum quihoui is about 1%): CrossEntropyLoss plus the inverse frequency of the pixel ratio of the training set class_weight
# (Mean normalization, the rarer the category, the greater the weight) to prevent the model from collapsing into a "full prediction background". Recalculation is required when the distribution of the data set changes.

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="RandomResize", scale=(1024, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1024, 1024), keep_ratio=True),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]

# mmseg.apis.inference_model() hardcoded to read cfg.test_pipeline, independent of val_dataloader configuration
test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=1, num_workers=4, persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(type=dataset_type, data_root=data_root,
                 data_prefix=dict(img_path="img_dir/train", seg_map_path="ann_dir/train"),
                 pipeline=train_pipeline),
)
val_dataloader = dict(
    batch_size=1, num_workers=4, persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(type=dataset_type, data_root=data_root,
                 data_prefix=dict(img_path="img_dir/val", seg_map_path="ann_dir/val"),
                 pipeline=val_pipeline),
)
test_dataloader = val_dataloader
val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU", "mDice", "mFscore"])
test_evaluator = val_evaluator

norm_cfg = dict(type="SyncBN", requires_grad=True)
# The PSPModule of decode_head (UPerHead) contains the "global pooling to 1x1" branch with pool_scales=1, during training
# batch_size=1 (default for this configuration, limited by Swin-Large memory) will cause BatchNorm to report an error due to insufficient number of samples.
# "Expected more than 1 value per channel when training". The head part uses GroupNorm to avoid it;
# Backbone's LayerNorm/SyncBN is not affected and does not need to be changed.
head_norm_cfg = dict(type="GN", num_groups=32, requires_grad=True)
data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True, pad_val=0, seg_pad_val=255, size=crop_size,
)
model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    # ImageNet-22K pre-trained weights are loaded by default (mmseg official Swin-Large window12_384 checkpoint)
    pretrained="https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth",
    backbone=dict(
        type="SwinTransformer",
        pretrain_img_size=384,
        embed_dims=192,
        patch_size=4,
        window_size=12,
        mlp_ratio=4,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type="GELU"),
        norm_cfg=dict(type="LN"),
    ),
    decode_head=dict(
        type="UPerHead",
        in_channels=[192, 384, 768, 1536],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=5,
        norm_cfg=head_norm_cfg,
        align_corners=False,
        loss_decode=dict(type="SafeCrossEntropyLoss", use_sigmoid=False, loss_weight=1.0, class_weight=[0.0364, 0.4529, 0.6537, 0.9887, 2.8683]),
    ),
    auxiliary_head=dict(
        type="FCNHead",
        in_channels=768, in_index=2, channels=256, num_convs=1,
        concat_input=False, dropout_ratio=0.1, num_classes=5,
        norm_cfg=head_norm_cfg, align_corners=False,
        loss_decode=dict(type="SafeCrossEntropyLoss", use_sigmoid=False, loss_weight=0.4, class_weight=[0.0364, 0.4529, 0.6537, 0.9887, 2.8683]),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="slide", crop_size=crop_size, stride=(768, 768)),
)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)
# Training by epoch: 50 epochs by default, verification once per round, LR is scheduled by epoch.
# warmup spans 2 epochs (end-begin must >= 2): LinearLR is a continuous multiplication scheduler (total_iters=end-begin-1),
# If it only spans 1 epoch, total_iters=0, lr will only be multiplied by start_factor once and will never be pulled back (stopped at ~1e-10 permanently).
max_epochs = 50  # The total number of training rounds; change this or use --max_epochs to overwrite it during training, and the PolyLR endpoint will be synchronized
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=True, begin=0, end=3),
    dict(type="PolyLR", eta_min=0.0, power=1.0, begin=1, end=max_epochs, by_epoch=True),
]
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", by_epoch=True, interval=1, max_keep_ckpts=3, save_best="mIoU"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)
# Early stop: If there is no improvement in mIoU for 15 consecutive epochs (the improvement is > min_delta), the training will be stopped and can be turned off with --no_early_stop
custom_hooks = [
    dict(type="EarlyStoppingHook", monitor="mIoU", rule="greater", min_delta=1e-3, patience=15),
]
env_cfg = dict(
    # Shared graphics: turn off cudnn_benchmark for predictable memory usage proportional to batch_size
    # (benchmark=True will select the algorithm based on the remaining video memory of the graphics card at each run, and is unstable on a card shared by multiple people)
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(type="SegLocalVisualizer", vis_backends=vis_backends, name="visualizer")
log_processor = dict(by_epoch=True)
log_level = "INFO"
default_scope = "mmseg"
load_from = None
resume = False
