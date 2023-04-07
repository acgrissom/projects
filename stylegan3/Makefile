pip_dependencies:
	pip install psutil

do_regression:
	@activate_env
	python -c "from regression import plot_regression; plot_regression()"

get_average_color:
	@activate_env
	python -c "from regression import get_average_color; get_average_color()"

miniconda.sh:
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > miniconda.sh
	sh miniconda.sh
	rm miniconda.sh

install_conda_env:
	conda env create --file environment.yml

activate_env:
	conda activate test

fairface_prepare:
	mkdir -p /mnt/data/students/fairface/data/padding_0.25/train_prepared
	python dataset_tool.py \
	--source /mnt/data/students/fairface/data/padding_0.25/train \
	--dest /mnt/data/students/fairface/data/padding_0.25/train_prepared \
	--resolution 512x512


fairface_clean:
	rm -R /mnt/data/students/models/fairface_stylegan3t/training-runs

fairface_train:	
	mkdir -p /mnt/data/students/models/fairface_stylegan3t/training-runs
	python train.py --outdir=/mnt/data/students/models/fairface_stylegan3/training-runs --cfg=stylegan3-t --data=/mnt/data/students/fairface/data/padding_0.25/train/train_prepared --gpus=8 --batch=32 --gamma=8.2 --mirror=1 --snap=5

