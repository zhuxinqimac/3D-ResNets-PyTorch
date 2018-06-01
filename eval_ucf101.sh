CUDA_VISIBLE_DEVICES=1 python main.py --root_path \
    /home/xinqizhu/repo/3D-ResNets-PyTorch --video_path \
    /home/xinqizhu/UCF101_frames --annotation_path \
    /home/xinqizhu/ucfTrainTestlist/ucf101_01.json \
    --result_path UCF101_results/results_34_s_s --dataset ucf101 \
    --n_classes 101 --n_finetune_classes 101 --pretrain_path \
    UCF101_results/results_34_s_s/save_100.pth --ft_begin_index 4 \
    --model resnet --model_depth 34 --resnet_shortcut A \
    --batch_size 16 --n_threads 2 --checkpoint 5 \
    --sample_duration 16 --resume_path \
    UCF101_results/results_34_s_s/save_100.pth \
    --no_train --no_val --test \
    --test_shuffle
