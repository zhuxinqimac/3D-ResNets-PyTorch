from datasets.kinetics import Kinetics
from datasets.activitynet import ActivityNet
from datasets.ucf101 import UCF101
from datasets.hmdb51 import HMDB51
from datasets.something import Something


def get_training_set(opt, spatial_transform, temporal_transform,
                     target_transform, score_sens_mode=False):
    assert opt.dataset in ['kinetics', 'activitynet', 
            'ucf101', 'hmdb51', 'something']

    if opt.dataset == 'kinetics':
        training_data = Kinetics(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'activitynet':
        training_data = ActivityNet(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'ucf101':
        training_data = UCF101(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform, 
            score_sens_mode=score_sens_mode)
    elif opt.dataset == 'hmdb51':
        training_data = HMDB51(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'something':
        training_data = Something(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform, 
            score_sens_mode=score_sens_mode)

    return training_data


def get_validation_set(opt, spatial_transform, temporal_transform,
                       target_transform, score_sens_mode=False):
    assert opt.dataset in ['kinetics', 'activitynet', 
            'ucf101', 'hmdb51', 'something']

    if opt.dataset == 'kinetics':
        validation_data = Kinetics(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'activitynet':
        validation_data = ActivityNet(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'ucf101':
        validation_data = UCF101(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration, 
            score_sens_mode=score_sens_mode)
    elif opt.dataset == 'hmdb51':
        validation_data = HMDB51(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'something':
        validation_data = Something(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration, 
            score_sens_mode=score_sens_mode)
    return validation_data


def get_test_set(opt, spatial_transform, temporal_transform, target_transform, 
        score_sens_mode=False, score_inf_mode=False, inner_temp_transform=None):
    assert opt.dataset in ['kinetics', 'activitynet', 
            'ucf101', 'hmdb51', 'something']
    assert opt.test_subset in ['val', 'test']

    if opt.test_subset == 'val':
        subset = 'validation'
    elif opt.test_subset == 'test':
        subset = 'testing'
    if opt.dataset == 'kinetics':
        test_data = Kinetics(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration,
            sample_stride=opt.sample_stride)
    elif opt.dataset == 'activitynet':
        test_data = ActivityNet(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'ucf101':
        test_data = UCF101(
            opt.video_path,
            opt.annotation_path,
            subset,
            # 0,
            1, # sample 1 clip each video
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration, 
            score_sens_mode=score_sens_mode, 
            score_inf_mode=score_inf_mode)
    elif opt.dataset == 'hmdb51':
        test_data = HMDB51(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'something':
        test_data = Something(
            opt.video_path,
            opt.annotation_path,
            subset,
            # 0,
            1, # sample 1 clip each video
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration, 
            score_sens_mode=score_sens_mode, 
            score_inf_mode=score_inf_mode, 
            inner_temp_transform=inner_temp_transform)

    return test_data
