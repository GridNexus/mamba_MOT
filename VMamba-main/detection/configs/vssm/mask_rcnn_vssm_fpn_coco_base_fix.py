# VMamba Base config for Mask R-CNN - matches checkpoint
# Checkpoint architecture: dims=128, depths=(2, 2, 15, 2), v2 patchembed, v3 downsample
_base_ = [
    '../swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py'
]

model = dict(
    type='MaskRCNN',
    backbone=dict(
        _delete_=True,
        type='MM_VSSM',
        out_indices=(0, 1, 2, 3),
        # VMamba Base architecture matching checkpoint
        dims=128,
        depths=(2, 2, 15, 2),
        ssm_d_state=16,
        ssm_dt_rank="auto",
        ssm_ratio=2.0,
        mlp_ratio=0.0,
        downsample_version="v3",
        patchembed_version="v2",
        forward_type="v05_noz",
        ssm_conv=3,
        ssm_conv_bias=False,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        norm_layer="ln2d",
        patch_norm=True,
        use_checkpoint=False,
    ),
    neck=dict(in_channels=[128, 256, 512, 1024]),
)
