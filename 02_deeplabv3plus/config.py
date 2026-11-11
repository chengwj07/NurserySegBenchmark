"""
02_deeplabv3plus/config.py
DeepLabV3+ (native implementation of mmsegmentation) segmentation configuration (default 5 categories of seedlings, see common/plant_dataset.py; if you need to switch back to two categories, see the root directory README)
Backbone: ResNetV1c-101 (dilated, ImageNet pre-trained) - the largest number of parameters in the mmseg DeepLabV3+ series,
          The standard backbone with the highest accuracy (twice the depth compared to R50, the official benchmark has higher accuracy)
Decode Head: DepthwiseSeparableASPPHead (ASPP + depthwise separable convolutional decoder)
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
    batch_size=2, num_workers=4, persistent_workers=True,
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
# The ASPP of decode_head contains the "image-level global pooling" branch (pooling the feature map to 1x1 and then connecting to BN),
# When batch_size=1 (common configuration for single card/small graphics memory), the BatchNorm training mode in 1x1 space will be affected by insufficient number of samples.
# The error "Expected more than 1 value per channel when training" is reported. Use GroupNorm for head part instead
# (Not limited by batch_size) Avoid this problem; backbone's SyncBN is not affected and does not need to be modified.
# num_groups needs to be divided by channels(512) and low-level feature projection channels c1_channels(48) at the same time, and the greatest common divisor of the two is 16.
head_norm_cfg = dict(type="GN", num_groups=16, requires_grad=True)
data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True, pad_val=0, seg_pad_val=255, size=crop_size,
)
model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    pretrained="open-mmlab://resnet101_v1c",  # ImageNet pre-trained weights are loaded by default
    backbone=dict(
        type="ResNetV1c",
        depth=101, num_stages=4, out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4), strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg, norm_eval=False, style="pytorch", contract_dilation=True,
    ),
    decode_head=dict(
        type="DepthwiseSeparableASPPHead",
        in_channels=2048, in_index=3, channels=512,
        dilations=(1, 12, 24, 36), c1_in_channels=256, c1_channels=48,
        dropout_ratio=0.1, num_classes=5, norm_cfg=head_norm_cfg, align_corners=False,
        loss_decode=dict(type="SafeCrossEntropyLoss", use_sigmoid=False, loss_weight=1.0, class_weight=[0.0364, 0.4529, 0.6537, 0.9887, 2.8683]),
    ),
    auxiliary_head=dict(
        type="FCNHead",
        in_channels=1024, in_index=2, channels=256, num_convs=1,
        concat_input=False, dropout_ratio=0.1, num_classes=5,
        norm_cfg=head_norm_cfg, align_corners=False,
        loss_decode=dict(type="SafeCrossEntropyLoss", use_sigmoid=False, loss_weight=0.4, class_weight=[0.0364, 0.4529, 0.6537, 0.9887, 2.8683]),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="slide", crop_size=crop_size, stride=(768, 768)),
)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0005),
    clip_grad=None,
)
# Training by epoch: 50 epochs by default, verification once per round, LR is scheduled by epoch.
# warmup spans 2 epochs (end-begin must >= 2): LinearLR is a continuous multiplication scheduler (total_iters=end-begin-1),
# If it only spans 1 epoch, total_iters=0, lr will only be multiplied by start_factor once and will never be pulled back (stopped at ~1e-10 permanently).
max_epochs = 50  # The total number of training rounds; change this or use --max_epochs to overwrite it during training, and the PolyLR endpoint will be synchronized
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=True, begin=0, end=3),
    dict(type="PolyLR", eta_min=0.0001, power=0.9, begin=1, end=max_epochs, by_epoch=True),
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
