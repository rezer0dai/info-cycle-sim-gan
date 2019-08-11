# info-cycle-sim-gan
infogan on cycle domain with new loss function - simgan

more information at : rezer0dai.github.io/info-cycle-sim-gan

This work is under development, need some refactorings and adaptation for general use. Meanwhile you
can run some test on it with : 

python regan.py --dataset_name horse2zebra --img_size 128 --batch_size 32 --latent_dim 196 --sample_interval 4 --space_dim 8 --postfix blog_simgan128

+ you need to download cyclegan datasets ( i recommend to test at first at apple2orange with img_size 64 ) 
