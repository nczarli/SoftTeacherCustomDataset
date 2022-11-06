_base_ = "base.py"
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(checkpoint="open-mmlab://detectron2/resnet101_caffe"),
    ),
    roi_head=dict(
      bbox_head=[
        dict(
          type='Shared2FCBBoxHead',
          num_classes=3,
        )
      ],
      mask_head=dict(num_classes=3)
    )
    
  
)

# 1. sataset settings
dataset_type = 'Cocodataset'
classes = ('pitted', 'not_pitted', 'try_again')
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
      type = dataset_type,
      ann_file="./labels_generated/train/annotations/instances_default.json",
      img_prefix="./labels_generated/train/images",
    ),
)



optimizer = dict(lr=0.02)
lr_config = dict(step=[120000 * 4, 160000 * 4])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=180000 * 4)
