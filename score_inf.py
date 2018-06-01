import argparse
import torch
import os
import time
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import json

from utils import Logger
from model import generate_model
from mean import get_mean, get_std
from dataset import get_training_set, get_validation_set, get_test_set
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import (
    IdentityTransform, 
    LoopPadding, TemporalRandomCrop,
    ShuffleFrames, ReverseFrames, TemporalSparseSample)
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from utils import AverageMeter, calculate_accuracy, accuracy

def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map

def evaluate(test_loader, model1, model2, criterion, val_logger, softmax, 
        analysis_recorder):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    s_n_top1 = AverageMeter()
    s_n_top5 = AverageMeter()

    # switch to evaluate mode
    model1.eval()
    model2.eval()

    end = time.time()
    for i, (inputs, target) in enumerate(test_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(inputs[0], volatile=True)
        # idx_list = inputs[2]
        # t_to_idx = {x:i for i, x in enumerate(idx_list[0])}
        # ab_idx_list = inputs[3]
        target_var = torch.autograd.Variable(target, volatile=True)

        v_path = inputs[1][0].replace(' ', '-')
        # compute output
        norm_output = model1(input_var)
        # print(norm_output.size()) # [1, 16, 512, 7, 7]
        # input('...')
        abnorm_output = model2(input_var)
        # print(abnorm_output[0][0, 0, ...])
        # loss = criterion(output, target_var)
        norm_sm = softmax(norm_output)
        abnorm_sm = softmax(abnorm_output)
        # print('norm_sm:', norm_sm)
        # print('abnorm_sm:', abnorm_sm)
        # input('...')
        loss = criterion(norm_sm, abnorm_sm)
        loss = torch.sqrt(loss)

        prec1, prec5 = accuracy(norm_output.data, target, topk=(1,5))
        top1.update(prec1[0], 1)
        top5.update(prec5[0], 1)
        prec1, prec5 = accuracy(abnorm_output.data, target, topk=(1,5))
        s_n_top1.update(prec1[0], 1)
        s_n_top5.update(prec5[0], 1)

        _, n_n_pred = norm_sm.max(1)
        _, s_n_pred = abnorm_sm.max(1)
        GT_class_name = class_to_name[target.cpu().numpy()[0]]
        if (n_n_pred.data == target).cpu().numpy():
            if_correct = 1
        else:
            if_correct = 0
        # print('v_path:', v_path)
        # print('n_n_pred:', n_n_pred)
        # print('s_n_pred:', s_n_pred)
        # print('target:', target)
        # print('GT_class_name:', GT_class_name)
        # print('if_correct:', if_correct)
        # input('...')


        losses.update(loss.data[0], 1)

        analysis_data_line = ('{path} {if_correct} {loss.val:.4f} '
                '{GT_class_name} {GT_class_index} '
                '{n_n_pred} {s_n_pred}'.format(
                    path=v_path, if_correct=if_correct, loss=losses, 
                    GT_class_name=GT_class_name, 
                    GT_class_index=target.cpu().numpy()[0], 
                    n_n_pred=n_n_pred.data.cpu().numpy()[0], 
                    s_n_pred=s_n_pred.data.cpu().numpy()[0]))
        with open(analysis_recorder, 'a') as f:
            f.write(analysis_data_line+'\n')

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 20 == 0:
            log_line = ('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  's_n_Prec@1 {s_n_top1.val:.3f} ({s_n_top1.avg:.3f})\t'
                  's_n_Prec@5 {s_n_top5.val:.3f} ({s_n_top5.avg:.3f})'.format(
                   i, len(test_loader), batch_time=batch_time, loss=losses, 
                   top1=top1, top5=top5, s_n_top1=s_n_top1, s_n_top5=s_n_top5))
            print(log_line)
            # eval_logger.write(log_line+'\n')
            with open(eval_logger, 'a') as f:
                f.write(log_line+'\n')

    # print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          # .format(top1=top1, top5=top5, loss=losses)))
    log_line = ('Testing Results: Loss {loss.avg:.5f}'
          .format(loss=losses))
    print(log_line)
    with open(eval_logger, 'a') as f:
        f.write(log_line+'\n\n')

    return

class FeatureMapModel(torch.nn.Module):
    def __init__(self, whole_model, consensus_type, modality, num_segments):
        super(FeatureMapModel, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        if self.modality == 'RGB':
            self.new_length = 1
        else:
            self.new_length = 10
        if consensus_type == 'bilinear_att' or consensus_type == 'conv_lstm':
            self.base_model = whole_model.module.base_model
        elif consensus_type == 'lstm' or consensus_type == 'ele_multi':
            removed = list(whole_model.module.base_model.children())[:-1]
            self.base_model = torch.nn.Sequential(*removed)
        elif consensus_type == 'avg' or consensus_type == 'max':
            removed = list(whole_model.module.base_model.children())[:-2]
            self.base_model = torch.nn.Sequential(*removed)
        else:
            ValueError(('Not supported consensus \
                        type {}.'.format(self.consensus_type)))
        # print(self.base_model)

    def forward(self, inputs):
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
        # print(input.size())
        # print(input.view((-1, sample_len) + input.size()[-2:]).size())

        base_out = self.base_model(inputs.view((-1, 
                        sample_len) + inputs.size()[-2:]))
        base_out = base_out.view((-1, self.num_segments) + \
                        base_out.size()[-3:])
        # base_out = base_out.mean(dim=1)
        # return base_out.squeeze(1)
        return base_out



if __name__=='__main__':
    parser = argparse.ArgumentParser(description=
                "Get a model's sensitivity to temporal info.")
    parser.add_argument('first_model_path', type=str, 
                help='Path to a pretrained model 1.')
    parser.add_argument('second_model_path', type=str, 
                help='Path to a pretrained model 2.')
    parser.add_argument('dataset', type=str, 
                choices=['ucf101', 'something'])
    parser.add_argument('annotation_path', type=str, help='Annotation file.')
    parser.add_argument('result_path', default='result', type=str,
                        metavar='LOG_PATH', help='results and log path')
    parser.add_argument('--sample_duration', type=int, default=16)
    # parser.add_argument('--arch', type=str, default="resnet34")
    parser.add_argument('--model_depth', type=int, default=34)
  # ======Modified ======
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    # parser.add_argument('--compared_temp_transform', default='shuffle', 
                        # type=str, help='temp transform to compare', 
                        # choices=['shuffle', 'reverse'])
    parser.add_argument('--gpus', nargs='+', type=int, default=None)
    parser.add_argument('--compared_temp_transform', default='shuffle', 
                        type=str)
# ======= Pasted ========
    parser.add_argument(
        '--no_mean_norm',
        action='store_true',
        help='If true, inputs are not normalized by mean.')
    parser.set_defaults(no_mean_norm=False)
    parser.add_argument(
        '--std_norm',
        action='store_true',
        help='If true, inputs are normalized by standard deviation.')
    parser.set_defaults(std_norm=False)
    parser.add_argument(
        '--sample_size',
        default=112,
        type=int,
        help='Height and width of inputs')
    parser.add_argument(
        '--norm_value',
        default=1,
        type=int,
        help=
        'If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument(
        '--model',
        default='resnet',
        type=str,
        help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument(
        '--n_finetune_classes',
        default=400,
        type=int,
        help=
        'Number of classes for fine-tuning. n_classes is set to the number when pretraining.'
    )
    parser.add_argument(
        '--ft_begin_index',
        default=4,
        type=int,
        help='Begin block index of fine-tuning')
    parser.add_argument(
        '--mean_dataset',
        default='activitynet',
        type=str,
        help=
        'dataset for mean values of mean subtraction (activitynet | kinetics)')
    parser.add_argument(
        '--video_path',
        default='video_kinetics_jpg',
        type=str,
        help='Directory path of Videos')
    parser.add_argument(
        '--n_val_samples',
        default=3,
        type=int,
        help='Number of validation samples for each activity')
    parser.add_argument(
        '--n_threads',
        default=2,
        type=int,
        help='Number of threads for multi-thread loading')
    parser.add_argument(
        '--test_subset',
        default='val',
        type=str,
        help='Used subset in test (val | test)')

    
    args = parser.parse_args()
    args.resnet_shortcut = 'A'
    args.arch = '{}-{}'.format(args.model, args.model_depth)
    args.no_cuda = False
    args.mean = get_mean(args.norm_value, dataset=args.mean_dataset)
    data = load_annotation_data(args.annotation_path)
    class_to_name = data['labels']
    for i in range(len(class_to_name)):
        class_to_name[i] = class_to_name[i].replace(' ', '-')
    if args.dataset == 'ucf101':
        num_class = 101
        args.n_classes = 101
        img_prefix = 'image_'
    else:
        num_class = 174
        args.n_classes = 174
        img_prefix = ''

    args.pretrain_path = args.first_model_path
    first_model, parameters_1 = generate_model(args)
    args.pretrain_path = args.second_model_path
    second_model, parameters_2 = generate_model(args)
    print(first_model)
    # input('...')

    if args.no_mean_norm and not args.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not args.std_norm:
        norm_method = Normalize(args.mean, [1, 1, 1])
    else:
        norm_method = Normalize(args.mean, args.std)
    
    spatial_transform = Compose([
        Scale(args.sample_size),
        CenterCrop(args.sample_size),
        ToTensor(args.norm_value), norm_method
    ])

    # temp_transform = IdentityTransform()
    temp_crop_method = TemporalRandomCrop(args.sample_duration)
    temp_transform = temp_crop_method
    # if opt.train_reverse:
        # temporal_transform = Compose([
            # ReverseFrames(opt.sample_duration), 
            # temp_crop_method
        # ])
    # elif opt.train_shuffle:
        # temporal_transform = Compose([
            # ShuffleFrames(opt.sample_duration), 
            # temp_crop_method
        # ])
    # else:
        # temporal_transform = temp_crop_method
    target_transform = ClassLabel()
    # validation_data = get_validation_set(
        # args, spatial_transform, temp_transform, target_transform, 
        # score_sens_mode=True)
    # val_loader = torch.utils.data.DataLoader(
        # validation_data,
        # batch_size=1,
        # shuffle=False,
        # num_workers=args.n_threads,
        # pin_memory=True)
    test_data = get_test_set(args, spatial_transform, temp_transform,
                             target_transform, 
                             score_inf_mode=True)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.n_threads,
        pin_memory=True)
    val_logger = Logger(
        os.path.join(args.result_path, 'val.log'), ['epoch', 'loss', 'acc'])

    if args.first_model_path:
        print('loading checkpoint {}'.format(args.first_model_path))
        checkpoint = torch.load(args.first_model_path)
        assert args.arch == checkpoint['arch']

        args.begin_epoch = checkpoint['epoch']
        first_model.load_state_dict(checkpoint['state_dict'])
        # if not args.no_train:
            # optimizer.load_state_dict(checkpoint['optimizer'])
    if args.second_model_path:
        print('loading checkpoint {}'.format(args.second_model_path))
        checkpoint = torch.load(args.second_model_path)
        assert args.arch == checkpoint['arch']

        args.begin_epoch = checkpoint['epoch']
        second_model.load_state_dict(checkpoint['state_dict'])
        # if not args.no_train:
            # optimizer.load_state_dict(checkpoint['optimizer'])

    # model = FeatureMapModel()
    f_model = first_model
    s_model = second_model
    # input('...')

    f_model = torch.nn.DataParallel(f_model, device_ids=None).cuda()
    s_model = torch.nn.DataParallel(s_model, device_ids=None).cuda()
    # model = torch.nn.DataParallel(model.cuda(devices[0]), device_ids=args.gpus)

    # cudnn.benchmark = True

    # if args.measure_type == 'KL':
        # criterion = torch.nn.KLDivLoss().cuda()
    # else:
        # criterion = torch.nn.MSELoss().cuda()
    criterion = torch.nn.MSELoss().cuda()
    softmax = torch.nn.Softmax(1)


    eval_logger = os.path.join(args.result_path, 'inf_log.log')
    with open(eval_logger, 'w') as f:
        f.write('')
    analysis_recorder = os.path.join(args.result_path, 'inf_analysis.txt')
    with open(analysis_recorder, 'w') as f:
        f.write('')
    evaluate(test_loader, f_model, s_model, criterion, val_logger=val_logger, 
            softmax=softmax, analysis_recorder=analysis_recorder)
    
