CUDA_VISIBLE_DEVICES=0 python main.py --root_path \
    /home/xinqizhu/repo/3D-ResNets-PyTorch --video_path \
    /home/xinqizhu/Something_frames --annotation_path \
    /home/xinqizhu/repo/3D-ResNets-PyTorch/annots/something.json \
    --result_path Something_results/test --dataset something \
    --n_classes 400 --n_finetune_classes 174 --pretrain_path \
    models/resnet-34-kinetics.pth \
    --ft_begin_index 3 \
    --model resnet --model_depth 34 --resnet_shortcut A \
    --batch_size 32 --n_threads 2 --checkpoint 2 \
    --sample_duration 32 --sample_size 84
