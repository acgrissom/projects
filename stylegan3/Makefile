NURVUS_MODELS_DIR := /mnt/data/students/models
KURAN_MODELS_DIR := /share/kuran/models
LOCAL_MODELS_DIR := ~/models
#MODELS_DIR := $(KURAN_MODELS_DIR)
MODELS_DIR := $(NURVUS_MODELS_DIR)

NURVUS_IMAGES_ROOT_DIR :=  /mnt/data/students
KURAN_IMAGES_ROOT_DIR := /share/kuran/gan_data
LOCAL_IMAGES_ROOT_DIR := ~/gan_data
#IMAGES_ROOT_DIR := $(KURAN_IMAGES_ROOT_DIR)
IMAGES_ROOT_DIR := $(NURVUS_IMAGES_ROOT_DIR)

FAIRFACE_IMAGES_DIR := $(IMAGES_ROOT_DIR)/fairface/data/padding_0.25/train_prepared
FFHQ_IMAGES_DIR := $(IMAGES_ROOT_DIR)/fairface/data/padding_0.25/train_prepared
#FAIRFACE_MODELS_DIR :=

pip_dependencies:
	pip install psutil tensorboard

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
	mkdir -p $(IMAGES_ROOT_DIR)/fairface/data/padding_0.25/train_prepared
	python dataset_tool.py \
	--source $(IMAGES_ROOT_DIR)/fairface/data/padding_0.25/train \
	--dest $(IMAGES_ROOT_DIR)/fairface/data/padding_0.25/train_prepared \
	--resolution 512x512


fairface_color_data.csv: get_average_colors.py
	python get_average_colors.py --images_dir $(FAIRFACE_IMAGES_DIR) --dest_csv ./fairface_color_data.csv --gan_model $(MODELS_DIR)/fairface_stylegan3/training-runs/00011-stylegan3-t-train_prepared-gpus8-batch32-gamma8.2/network-snapshot-007000.pkl

fairface_data_ffhq_stylegan3r_model.csv: get_average_colors.py
	python get_average_colors.py \
	--images_dir $(FFHQ_IMAGES_DIR) \
	--dest_csv ./fairface_data_ffhq_styleganr-model.csv \
	--gan_model $(MODELS_DIR)/stylegan3-official/stylegan3-r-ffhq-1024x1024.pkl --resize 1024

fairface_color_data_stylegan3t.csv: get_average_colors.py
	python get_average_colors.py --images_dir /mnt/data/students/fairface/data/padding_0.25/train_prepared --dest_csv ./fairface_color_data.csv --gan_model /mnt/data/students/models/fairface_stylegan3/training-runs/00011-stylegan3-t-train_prepared-gpus8-batch32-gamma8.2/network-snapshot-007000.pkl

#fairface_data_ffhq_stylegan3r_model.csv: get_average_colors.py
#	python get_average_colors.py \
	--images_dir /mnt/data/students/fairface/data/padding_0.25/train_prepared \
	--dest_csv ./fairface_data_ffhq_stylegan3r-model.csv \
	--gan_model /mnt/data/students/models/stylegan3-official/stylegan3-r-ffhq-1024x1024.pkl --resize 1024


ffhq_color_data_ffhq_stylegan3r_model.csv: get_average_colors.py
	python get_average_colors.py \
	--images_dir $(FFHQ_IMAGES_DIR) \
	--dest_csv ./ffhq_color_data_ffhq_stylegan3r_model.csv \
	--gan_model $(MODELS_DIR)/stylegan3-official/stylegan3-r-ffhq-1024x1024.pkl #--resize 1024


COLOR_DATA_OUT_DIR := generated_data/average_colors/stylegan3-official
generated_data/average_colors/stylegan3-official: get_average_colors.py
	mkdir -p $(COLOR_DATA_OUT_DIR)
	@for model_file in $(shell find  ${MODELS_DIR}/stylegan3-official -type f -printf "%f\n") ; do \
		echo Creating $(COLOR_DATA_OUT_DIR)/ffhq_test_$${model_file}.csv ; \
		python get_average_colors.py \
		--images_dir $(FFHQ_IMAGES_DIR) \
		--dest_csv $(COLOR_DATA_OUT_DIR)/ffhq_test_$${model_file}.csv \
		--gan_model $(MODELS_DIR)/stylegan3-official/$${model_file} ; \
		rmdir $(COLOR_DATA_OUT_DIR) ; done
	mkdir -p ../ganfaces/data/color_data/
	cp $(COLOR_DATA_OUT_DIR)/ffhq_test_*.csv ../gan_faces/data/color_data



ffhq_prepare_512:
	mkdir -p $(IMAGES_ROOT_DIR)/ffhq_processed_512x512
	python dataset_tool.py \
	--source $(IMAGES_ROOT_DIR)/ffhq_images_1024x1024 \
	--dest $(IMAGES_ROOT_DIR)/ffhq_processed_512x512 \
	--resolution=512x512

#fairface_clean:
#	rm -R $(MODELS_DIR)/fairface_stylegan3t/training-runs


fairface_train:	
	mkdir -p $(MODELS_DIR)/fairface_stylegan3t/training-runs
	python train.py --outdir=$(MODELS_DIR)/fairface_stylegan3/training-runs --cfg=stylegan3-t --data=$(FAIRFACE_IMAGES_DIR) --gpus=8 --batch=32 --gamma=8.2 --mirror=1 --snap=5

#make ffhq_clean:
#	rm -R $(MODELS_DIR)/ffhq_stylegan3t/training-runs
# Deleted this target because it's too dangerous.

ffhq_train_512:
	mkdir -p $(MODELS_DIR)/ffhq_stylegan3t/training-runs
	python train.py --outdir=$(MODELS_DIR)/ffhq_stylegan3t/training-runs --data=$(IMAGES_ROOT_DIR)/ffhq_processed_512x512 --gpus=8 --batch=32 --gamma=8.2 --mirror=1 --snap=5 --cfg=stylegan3-t

fairface_train_stylegan3t:	
	mkdir -p /mnt/data/students/models/fairface_stylegan3t/training-runs
	python train.py --outdir=/mnt/data/students/models/fairface_stylegan3/training-runs --cfg=stylegan3-t --data=/mnt/data/students/fairface/data/padding_0.25/train_prepared --gpus=8 --batch=32 --gamma=8.2 --mirror=1 --snap=5

fairface_train_stylegan3r:	
	mkdir -p /mnt/data/students/models/fairface_stylegan3t/training-runs
	python train.py --outdir=/mnt/data/students/models/fairface_stylegan3/training-runs --cfg=stylegan3-r --data=/mnt/data/students/fairface/data/padding_0.25/train_prepared --gpus=8 --batch=32 --gamma=8.2 --mirror=1 --snap=5


make ffhq_clean:
	rm -R /mnt/data/students/models/ffhq_stylegan3t/training-runs

ffhq_train_stylegan3t:
	mkdir -p /mnt/data/students/models/ffhq_stylegan3t/training-runs
	python train.py --outdir=/mnt/data/students/models/ffhq_stylegan3t/training-runs --data=/mnt/data/students/ffhq_processed_512x512 --gpus=8 --batch=32 --gamma=8.2 --mirror=1 --snap=5 --cfg=stylegan3-t


rsync_data:
	rsync -r /mnt/data/students /share/kuran/data/nurvus_sync
