import argparse
import torch
import os
import time
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from model import generate_model
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import (
    LoopPadding, TemporalRandomCrop,
    ShuffleFrames, ReverseFrames, TemporalSparseSample, 
    IdentityTransform)
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose

def evaluate(test_loader, model1, model2, criterion, val_logger, softmax):
    batch_time = AverageMeter()
    losses = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()

    # switch to evaluate mode
    model1.eval()
    model2.eval()

    end = time.time()
    for i, (inputs, target) in enumerate(test_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(inputs)
        # idx_list = inputs[2]
        # t_to_idx = {x:i for i, x in enumerate(idx_list[0])}
        # ab_idx_list = inputs[3]
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        norm_output = model1(input_var)
        print(norm_output.size()) # [1, 16, 512, 7, 7]
        abnorm_output = model2(input_var)
        # print(abnorm_output[0][0, 0, ...])
        # loss = criterion(output, target_var)
        norm_sm = softmax(norm_output)
        abnorm_sm = softmax(abnorm_output)
        loss = criterion(norm_sm, abnorm_sm)
        loss = torch.sqrt(loss)

        # # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1,5))

        losses.update(loss.data[0], 1)
        # losses.update(loss.data[0], len(inputs))
        # top1.update(prec1[0], inputs.size(0))
        # top5.update(prec5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 20 == 0:
            log_line = ('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   i, len(test_loader), batch_time=batch_time, loss=losses))
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
                help='Path to the first pretrained model.')
    parser.add_argument('second_resume_path', type=str, 
                help='Path to the second pretrained model.')
    parser.add_argument('dataset', type=str, 
                choices=['ucf101', 'something'])
    parser.add_argument('annotation_path', type=str, help='Annotation file.')
    parser.add_argument('result_path', default='result', type=str,
                        metavar='LOG_PATH', help='results and log path')
    parser.add_argument('--sample_duration', type=int, default=16)
    # parser.add_argument('--arch', type=str, default="resnet34")
    parser.add_argument('--model_depth', type=int, default=34)

    # ====== Modified ======
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

    
    args = parser.parse_args()
    args.resnet_shortcut = 'A'
    args.no_cuda = False
    args.pretrain_path = args.model_path
    if args.dataset == 'ucf101':
        num_class = 101
        args.n_classes = 101
        img_prefix = 'image_'
    else:
        num_class = 174
        args.n_classes = 174
        img_prefix = ''

    first_model, parameters_1 = generate_model(args)
    second_model, parameters_2 = generate_model(args)
    print(first_model)
    input('...')

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
    # if not args.test_temp_crop == 'sparse':
    # if args.compared_temp_transform == 'shuffle':
        # temp_transform = ShuffleFrames(args.sample_duration)
    # else:
        # temp_transform = ReverseFrames(args.sample_duration)
    temp_transform = IdentityTransform()

    target_transform = ClassLabel()
    validation_data = get_validation_set(
        args, spatial_transform, temp_transform, target_transform, 
        score_sens_mode=False)
    val_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.n_threads,
        pin_memory=True)
    val_logger = Logger(
        os.path.join(args.result_path, 'val.log'), ['epoch', 'loss', 'acc'])

    if args.first_resume_path:
        print('loading checkpoint {}'.format(args.first_resume_path))
        checkpoint = torch.load(args.first_resume_path)
        assert args.arch == checkpoint['arch']

        args.begin_epoch = checkpoint['epoch']
        first_model.load_state_dict(checkpoint['state_dict'])
        if not args.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

    if args.second_resume_path:
        print('loading checkpoint {}'.format(args.second_resume_path))
        checkpoint = torch.load(args.second_resume_path)
        assert args.arch == checkpoint['arch']

        args.begin_epoch = checkpoint['epoch']
        second_model.load_state_dict(checkpoint['state_dict'])
        if not args.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])
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


    eval_logger = os.path.join(args.result_path, 'fm_distance.log')
    evaluate(val_loader, f_model, s_model, criterion, val_logger=val_logger)
    
