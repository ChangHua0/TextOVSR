exp_name = 'TextOVSR_x4'

# model settings
model = dict(
    type='TextOVSR_stage2',
    generator=dict(
        type='Real_TextOVSRNet',
        mid_channels=64,
        num_propagation_blocks=10,
        num_cleaning_blocks=20,
        dynamic_refine_thres=255,  # change to 5 for test
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'basicvsr/spynet_20210409-c6c1bd09.pth',
        is_fix_cleaning=False,
        is_sequential_cleaning=False),
    discriminator=dict(
        type='TED',
        in_channels=3,
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


val_dataset_type = 'SRFolderMultipleGTDataset'

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
    dict(type='LoadDegLevelsFromJSON',),
    dict(type='LoadCaptionsFromJSON',),
    dict(type='FramesToTensor', keys=['lq']),
    dict(type='Collect', keys=['lq'], meta_keys=['lq_path', 'key','degrade_prompts', 'captions'])
]


data = dict(
    workers_per_gpu=10,
    # test
    test=dict(
        type=val_dataset_type,
        lq_folder='/OperaLQ/',
        gt_folder='/OperaLQ/',
        # num_input_frames=10,
        pipeline=test_pipeline,
        scale=4,
        test_mode=True),
)

# dist_params = dict(backend='nccl')
# test_cfg = dict(metrics=[])
test_cfg = dict(metrics=['NRQM'], crop_border=0)
