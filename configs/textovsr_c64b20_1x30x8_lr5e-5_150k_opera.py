exp_name = 'textovsr_c64b20_1x30x8_lr5e-5_150k_opera'

scale = 4

# model settings
model = dict(
    type='TextOVSR_stage2',
    generator=dict(
        type='Real_TextOVSRNet',
        mid_channels=64,
        num_propagation_blocks=10, #20
        num_cleaning_blocks=20,
        dynamic_refine_thres=255,  # change to 5 for test
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'basicvsr/spynet_20210409-c6c1bd09.pth',
        is_fix_cleaning=False,
        is_sequential_cleaning=False),
    discriminator=dict(
        type='TED',
        in_channels=3,
        text_dim=768,
        mid_channels=64,
        skip_connection=True),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    cleaning_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    neg_loss=dict(type='L1Loss', loss_weight=0.5, reduction='mean'),
    # CLIPIQALoss 
    clipiqa_loss=dict(type='CLIPIQALoss',loss_weight=0.5),
    perceptual_loss=dict(
        type='PerceptualLoss',
        layer_weights={
            '2': 0.1,
            '7': 0.1,
            '16': 1.0,
            '25': 1.0,
            '34': 1.0,
        },
        vgg_type='vgg19',
        perceptual_weight=1.0,
        style_weight=0,
        norm_img=False),
    gan_loss=dict(
        type='GANLoss',
        gan_type='vanilla',
        loss_weight=5e-2,
        real_label_val=1.0,
        fake_label_val=0),
    is_use_sharpened_gt_in_pixel=True,
    is_use_sharpened_gt_in_percep=True,
    is_use_sharpened_gt_in_gan=False,
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
    dict(type='LoadDegLevelsFromJSON',),
    dict(type='LoadCaptionsFromJSON',),
    # dict(type='MirrorSequence', keys=['degrade_prompts']),
    
    dict(type='Quantize', keys=['lq']), # 像量化并剪切到[0,1]
    dict(type='FramesToTensor', keys=['lq', 'gt', 'gt_unsharp']),
    dict(
        type='Collect', keys=['lq', 'gt', 'gt_unsharp'], meta_keys=['gt_path','lq_path','degrade_prompts', 'captions'])
    
    
    
    # dict(type='Quantize', keys=['lq']), # 像量化并剪切到[0,1]
    # dict(type='FramesToTensor', keys=['lq', 'gt', 'gt_unsharp']),
    # dict(
    #     type='Collect', keys=['lq', 'gt', 'gt_unsharp'], meta_keys=['gt_path'])
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
    # workers_per_gpu=10,
    workers_per_gpu=1,
    train_dataloader=dict(samples_per_gpu=1, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),


    # train
    train=dict(
        type='RepeatDataset',
        times=150,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='/opera_data_realworld/degraded_hr_7_3rd',
            gt_folder='/opera_data_realworld/argument_hr',
            num_input_frames=22,
            pipeline=train_pipeline,
            scale=4,
            test_mode=False)),
    # val
    val=dict(
        type=val_dataset_type,
        # lq_folder='data/UDM10/BIx4',
        # gt_folder='data/UDM10/GT',
        lq_folder='/UDM10/UDM10/BIx4',
        gt_folder='/UDM10/UDM10/GT',
        pipeline=val_pipeline,
        scale=4,
        test_mode=True),
    # test
    test=dict(
        type=val_dataset_type,
        # lq_folder='data/VideoLQ',
        # gt_folder='data/VideoLQ',
        lq_folder='/OperaLQ/',
        gt_folder='/OperaLQ/',

        pipeline=test_pipeline,
        scale=4,
        test_mode=True),
)


# optimizer
optimizers = dict(
    generator=dict(type='Adam', lr=5e-5, betas=(0.9, 0.99)),
    discriminator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99)))

# learning policy
total_iters = 150000
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


load_from =  '/experiments/textovsr_wogan_c64b20_2x30x8_lr1e-4_100k_opera/iter_100000.pth'

resume_from = None
workflow = [('train', 1)]

