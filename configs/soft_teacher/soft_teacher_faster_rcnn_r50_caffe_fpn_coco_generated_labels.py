_base_="base.py"



num_classes = 3
dataset_type = 'CocoDataset'
classes = ('pitted', 'not_pitted', 'try_again')
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        sup=dict(
            type = dataset_type,
            classes=classes,
            ann_file="labels_generated/train/annotations/instances_default.json",
            img_prefix="labels_generated/train/images/",

        ),
        unsup=dict(
            type = dataset_type,
            classes=classes,
            ann_file="labels_generated/unlabelled/annotations/instances_default.json",
            img_prefix="labels_generated/unlabelled/images/",
        ),
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
    sampler=dict(
        train=dict(
            sample_ratio=[1, 1],
        )
    ),
)


model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            num_classes=num_classes,
            )
    )
)

semi_wrapper = dict(
    train_cfg=dict(
        unsup_weight=2.0,
    )
)

lr_config = dict(
  step=[120000 * 4, 160000 * 4],
)
log_config=dict(
  interval=1
)
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=500)

