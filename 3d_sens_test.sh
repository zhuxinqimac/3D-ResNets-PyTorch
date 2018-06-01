CUDA_VISIBLE_DEVICES=1 python score_sens.py \
    UCF101_results/results_34_n_n/save_20.pth ucf101 \
    /home/xinqizhu/ucfTrainTestlist/ucf101_01.json \
    UCF101_results/sens_reverse --sample_duration 16 --model_depth 34 \
    --n_finetune_classes 101 \
    --video_path /home/xinqizhu/UCF101_frames \
    --compared_temp_transform reverse
#CUDA_VISIBLE_DEVICES=1 python score_sens.py \
    #Something_results/results_34_n_n/save_45.pth something \
    #/home/xinqizhu/repo/TRN-pytorch/video_datasets/something/something.json \
    #Something_results/sens --sample_duration 32 --model_depth 34 \
    #--n_finetune_classes 174 \
    #--video_path /home/xinqizhu/Something_frames \
    #--sample_size 84
    ##--compared_temp_transform shuffle
