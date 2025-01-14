To get diachronic plots:

1.  Train model after setting random seed.
2.  Modify the Makefile variable `snapshot_subdir`   to point to the new model
3.  Run `make -B generated_data/average_colors/ffhq_test_512`
4.  In ganfaces directory, run diachronic plot target

- Random seeds start a 0 and increase by 1, where 139 is 2.
- Note that due to some failed runs, 143 is seed 5
Code adopted from Nvidia StyleGan3 Implementation: [https://github.com/NVlabs/stylegan3](https://github.com/NVlabs/stylegan3)

# Documentation
discriminator.py has all the functions for using the trained discriminator in stylegan3.
 

# Reminder
* All the images are stored in mnt/data/students in nurvus machine or check Box drive folder for online version: [Link](https://haverford.app.box.com/folder/151383007310) (Needs to update online version)
* Please do not push large images and models to this repository. A .gitignore file is
included, and modify it as you wish. 
* stylegan3/torch_utils/ops/grid_sample_gradfix.py line 60 is changed to remove an error. 
See [Github Issue](https://github.com/NVlabs/stylegan3/issues/188) for more details.

# Results
1. Correlation between discriminator scores and luminance:   

    | Attempt | #1    | #2    |
    | :---:   | :---: | :---: |
    | Seconds | 301   | 283   |

