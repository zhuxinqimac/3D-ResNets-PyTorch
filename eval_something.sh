CUDA_VISIBLE_DEVICES=1 python main.py --root_path \
    /home/xinqizhu/repo/3D-ResNets-PyTorch --video_path \
    /home/xinqizhu/Something_frames --annotation_path \
    /home/xinqizhu/repo/3D-ResNets-PyTorch/annots/something.json \
    --result_path Something_results/test --dataset something \
    --n_classes 174 --n_finetune_classes 174 --pretrain_path \
    Something_results/vanilla_3d_something/save_45.pth \
    --ft_begin_index 3 \
    --model resnet --model_depth 34 --resnet_shortcut A \
    --batch_size 16 --n_threads 2 --checkpoint 5 \
    --sample_duration 32 --resume_path \
    Something_results/vanilla_3d_something/save_45.pth \
    --sample_size 84 \
    --no_train --no_val --test
