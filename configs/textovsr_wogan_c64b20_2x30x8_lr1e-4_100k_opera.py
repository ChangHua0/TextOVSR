exp_name = 'textovsr_wogan_c64b20_2x30x8_lr1e-4_100k_opera'

scale = 4

# model settings
model = dict(
    type='TextOVSR_stage1',
    generator=dict(
        type='Real_TextOVSRNet',
        mid_channels=64,
        num_propagation_blocks=10,
        num_cleaning_blocks=20,
        dynamic_refine_thres=255,  # change to 1.5 for test
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'basicvsr/spynet_20210409-c6c1bd09.pth',
        is_fix_cleaning=False,
        is_sequential_cleaning=False),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    cleaning_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    neg_loss=dict(type='L1Loss', loss_weight=0.5, reduction='mean'),
    is_use_sharpened_gt_in_pixel=True,
    is_use_ema=True,
)

# model training and testing settings
train_cfg = dict()
test_cfg = dict(metrics=['PSNR'], crop_border=0)  # change to [] for test

# dataset settings
train_dataset_type = 'SRFolderMultipleGTDataset'
val_dataset_type = 'SRFolderMultipleGTDataset'

train_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['gt']),
    # dict(type='MirrorSequence', keys=['gt']),
    dict(
        type='UnsharpMasking',
        keys=['gt'],
        kernel_size=51,
        sigma=0,
        weight=0.5,
        threshold=10),

    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    # dict(type='MirrorSequence', keys=['lq']),
    dict(
        type='RandomResize',
        params=dict(
            target_size=(64, 64),
            resize_opt=['bilinear', 'area', 'bicubic'],
            resize_prob=[1 / 3., 1 / 3., 1 / 3.])
            , keys=['lq']
    ),
    # 读取退化描述
    dict(type='LoadDegLevelsFromJSON',),
    # dict(type='MirrorSequence', keys=['degrade_prompts']),

    # 需要把captions.json 复制到退化集的目录下，目前还没有生成完全只复制了三个
    # 读取Caption描述
    dict(type='LoadCaptionsFromJSON',),
    # dict(type='MirrorSequence', keys=['captions']),

    dict(type='Quantize', keys=['lq']), # 像量化并剪切到[0,1]
    dict(type='FramesToTensor', keys=['lq', 'gt', 'gt_unsharp']),
    dict(
        type='Collect', keys=['lq', 'gt', 'gt_unsharp'], meta_keys=['gt_path','lq_path','degrade_prompts', 'captions'])
    

]


val_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key'])
]

test_pipeline = [
    dict(
        type='GenerateSegmentIndices',
        interval_list=[1],
        filename_tmpl='{:08d}.png'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='FramesToTensor', keys=['lq']),
    dict(type='Collect', keys=['lq'], meta_keys=['lq_path', 'key'])
]

data = dict(
    workers_per_gpu=10,
    # train_dataloader=dict(
    #     samples_per_gpu=2, drop_last=True, persistent_workers=False),
    # val_dataloader=dict(samples_per_gpu=1, persistent_workers=False),
    # test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),
    train_dataloader=dict(
        samples_per_gpu=1, drop_last=True, persistent_workers=False),
    val_dataloader=dict(samples_per_gpu=1, persistent_workers=False),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train=dict(
        type='RepeatDataset',
        times=150,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='/opera_data_realworld/degraded_hr_7_3rd',
            gt_folder='/opera_data_realworld/argument_hr',
            num_input_frames=30,
            pipeline=train_pipeline,
            scale=4,
            test_mode=False)),
    # val
    val=dict(
        type=val_dataset_type,
        lq_folder='/UDM10/UDM10/BIx4',
        gt_folder='/UDM10/UDM10/GT',
        pipeline=val_pipeline,
        scale=4,
        test_mode=True),
    # test
    test=dict(
        type=val_dataset_type,
        lq_folder='/OperaLQ/',
        gt_folder='/OperaLQ/',
        pipeline=test_pipeline,
        scale=4,
        test_mode=True),
)

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99)))

# learning policy
total_iters = 100000
lr_config = dict(policy='Step', by_epoch=False, step=[400000], gamma=1)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)

# remove gpu_collect=True in non distributed training
evaluation = dict(interval=5000, save_image=False, gpu_collect=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
    ])
visual_config = None

# custom hook
custom_hooks = [
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema', ),
        interval=1,
        interp_cfg=dict(momentum=0.999),
    )
]

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./experiments/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
