CUDA_VISIBLE_DEVICES=1 python main.py --root_path \
    /home/xinqizhu/repo/3D-ResNets-PyTorch --video_path \
    /home/xinqizhu/Something_frames --annotation_path \
    /home/xinqizhu/repo/TRN-pytorch/video_datasets/something/something.json \
    --result_path Something_results/results_34_s_s --dataset something \
    --n_classes 174 --n_finetune_classes 174 --pretrain_path \
    Something_results/results_34_s_s/save_50.pth \
    --ft_begin_index 3 \
    --model resnet --model_depth 34 --resnet_shortcut A \
    --batch_size 16 --n_threads 2 --checkpoint 5 \
    --sample_duration 32 --resume_path \
    Something_results/results_34_s_s/save_50.pth \
    --sample_size 84 \
    --no_train --no_val --test \
    --test_shuffle
