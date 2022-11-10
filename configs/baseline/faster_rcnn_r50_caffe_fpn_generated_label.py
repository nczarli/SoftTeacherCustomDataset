_base_ = "base.py"

model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            num_classes=3,
            )
    )
)

dataset_type = 'CocoDataset'
classes = ('pitted', 'not_pitted', 'try_again')
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
      type = dataset_type,
      classes=classes,
      ann_file="labels_generated/train/annotations/instances_default.json",
      img_prefix="labels_generated/train/images/",
    ),
    val=dict(
      type = dataset_type,
      classes=classes,
      ann_file="labels_generated/test/annotations/instances_default.json",
      img_prefix="labels_generated/test/images/",
    ),
    test=dict(
      type = dataset_type,
      classes=classes,
      ann_file="labels_generated/test/annotations/instances_default.json",
      img_prefix="labels_generated/test/images/",
    ),
)


optimizer = dict(lr=0.02)
lr_config = dict(
  step=[120000 * 4, 160000 * 4],
  interval=1,
)
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=200)
