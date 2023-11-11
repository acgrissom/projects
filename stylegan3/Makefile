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
FFHQ_IMAGES_DIR := $(IMAGES_ROOT_DIR)/ffhq_processed_512x512/
FFHQ_IMAGES_DIR_128 = $(IMAGES_ROOT_DIR)/ffhq_processed_128x128/
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

fairface_prepare_128:
	mkdir -p $(IMAGES_ROOT_DIR)/fairface/data/padding_0.25/train_prepared_128x128
	python dataset_tool.py \
	--source $(IMAGES_ROOT_DIR)/fairface/data/padding_0.25/train \
	--dest $(IMAGES_ROOT_DIR)/fairface/data/padding_0.25/train_prepared_128x128 \
	--resolution 128x128


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






COLOR_DATA_OUT_DIR := generated_data/average_colors
${COLOR_DATA_OUT_DIR}/ffhq_test: get_average_colors.py
	mkdir -p $(COLOR_DATA_OUT_DIR)/ffhq_test
	@for model_file in $(shell find  ${MODELS_DIR}/stylegan3-official -type f -printf "%f\n") ; do \
		echo Creating $(COLOR_DATA_OUT_DIR)/ffhq_test/ffhq_test_$${model_file}.csv ; \
		python get_average_colors.py \
		--images_dir $(FFHQ_IMAGES_DIR) \
		--dest_csv $(COLOR_DATA_OUT_DIR)/ffhq_test/ffhq_test_$${model_file}.csv \
		--gan_model $(MODELS_DIR)/stylegan3-official/$${model_file} ; done
	mkdir -p ../ganfaces/data/color_data/ffhq_test
	cp -R $(COLOR_DATA_OUT_DIR)/ffhq_test ../ganfaces/data/color_data






#FFHQ_STYLEGANR_FINAL_MODEL := $(MODELS_DIR)/ffhq_stylegan3t/training-runs/128x128/00007-stylegan3-r-ffhq_processed_128x128-gpus8-batch32-gamma8.2/
#${COLOR_DATA_OUT_DIR}/ffhq_test_128: snapshot_subdir=00005-stylegan3-r-ffhq_processed_128x128-gpus8-batch32-gamma8.2
${COLOR_DATA_OUT_DIR}/ffhq_test_128: snapshot_subdir=00007-stylegan3-r-ffhq_processed_128x128-gpus8-batch32-gamma8.2
#${COLOR_DATA_OUT_DIR}/ffhq_test_128: snapshot_subdir=00007-stylegan3-r-ffhq_processed_128x128-gpus8-batch32-gamma8.2
${COLOR_DATA_OUT_DIR}/ffhq_test_128: snapshots_path=${MODELS_DIR}/ffhq_stylegan3t/training-runs/128x128/${snapshot_subdir}
${COLOR_DATA_OUT_DIR}/ffhq_test_128: get_average_colors.py 
	mkdir -p $(COLOR_DATA_OUT_DIR)/ffhq_test_128
	@for model_file in $(shell find ${snapshots_path}/*000.pkl -type f -printf "%f\n") ; do \
		echo Reading $${model_file} ; \
		python get_average_colors.py \
		--images_dir $(FFHQ_IMAGES_DIR_128) \
		--dest_csv $(COLOR_DATA_OUT_DIR)/ffhq_test_128/ffhq_test_${snapshot_subdir}_$${model_file}.csv \
		--gan_model ${snapshots_path}/$${model_file} \
		--model_dimensions 128x128 ; done
	mkdir -p ../ganfaces/data/color_data/ffhq_test_128
	cp -R $(COLOR_DATA_OUT_DIR)/ffhq_test_128 ../ganfaces/data/color_data/




#FFHQ_STYLEGANR_FINAL_MODEL := $(MODELS_DIR)/ffhq_stylegan3t/training-runs/
#${COLOR_DATA_OUT_DIR}/ffhq_test_512: snapshot_subdir=00000-stylegan3-t-ffhq_processed_512x512-gpus8-batch32-gamma8.2
#${COLOR_DATA_OUT_DIR}/ffhq_test_512: snapshot_subdir=00133-stylegan3-r-ffhq_processed_512x512-gpus8-batch32-gamma8.2

${COLOR_DATA_OUT_DIR}/ffhq_test_512: snapshot_subdir=00135-stylegan3-r-ffhq_processed_512x512-gpus8-batch32-gamma8.2

${COLOR_DATA_OUT_DIR}/ffhq_test_512: snapshots_path=${MODELS_DIR}/ffhq_stylegan3t/training-runs/${snapshot_subdir}
${COLOR_DATA_OUT_DIR}/ffhq_test_512: get_average_colors.py 
	mkdir -p $(COLOR_DATA_OUT_DIR)/ffhq_test_512
	@for model_file in $(shell find ${snapshots_path}/*000.pkl -type f -printf "%f\n") ; do \
		echo Reading $${model_file} ; \
		python get_average_colors.py \
		--images_dir $(FFHQ_IMAGES_DIR) \
		--dest_csv $(COLOR_DATA_OUT_DIR)/ffhq_test_512/ffhq_test_${snapshot_subdir}_$${model_file}.csv \
		--gan_model ${snapshots_path}/$${model_file} \
		--model_dimensions 512x512 ; done
	mkdir -p ../ganfaces/data/color_data/ffhq_test_512
	cp -R $(COLOR_DATA_OUT_DIR)/ffhq_test_512 ../ganfaces/data/color_data/






${COLOR_DATA_OUT_DIR}/fairface_test: get_average_colors.py ${COLOR_DATA_OUT_DIR}/fairface_test/fairface_model
	mkdir -p $(COLOR_DATA_OUT_DIR)/fairface_test
	@for model_file in $(shell find  ${MODELS_DIR}/stylegan3-official -type f -printf "%f\n") ; do \
		echo Creating $(COLOR_DATA_OUT_DIR)/fairface_test/fairface_test_$${model_file}.csv ; \
		python get_average_colors.py \
		--images_dir $(FAIRFACE_IMAGES_DIR) \
		--dest_csv $(COLOR_DATA_OUT_DIR)/fairface_test/fairface_test_$${model_file}.csv \
		--gan_model $(MODELS_DIR)/stylegan3-official/$${model_file} ; done
	mkdir -p ../ganfaces/data/color_data/fairface_test
	cp -R $(COLOR_DATA_OUT_DIR)/fairface_test ../ganfaces/data/color_data/
	mv ../ganfaces/data/color_data/fairface_test/fairface_model/*.csv ../ganfaces/data/color_data/fairface_test/
	rmdir ../ganfaces/data/color_data/fairface_test/fairface_model

FAIRFACE_STYLEGANT_FINAL_MODEL := /mnt/data/students/models/fairface_stylegan3/training-runs/00007-stylegan3-t-train_prepared-gpus8-batch32-gamma8.2/network-snapshot-025000.pkl
FAIRFACE_STYLEGANR_FINAL_MODEL := $(MODELS_DIR)/fairface_stylegan3/training-runs/00012-stylegan3-r-train_prepared-gpus8-batch32-gamma8.2/network-snapshot-025000.pkl


${COLOR_DATA_OUT_DIR}/fairface_test/fairface_model: get_average_colors.py
	mkdir -p $(COLOR_DATA_OUT_DIR)/fairface_test/fairface_model
	python get_average_colors.py \
	--images_dir $(FAIRFACE_IMAGES_DIR) \
	--dest_csv $(COLOR_DATA_OUT_DIR)/fairface_test/fairface_model/fairface_test_fairface_stylegan3r_25000.csv \
	--gan_model ${FAIRFACE_STYLEGANR_FINAL_MODEL} \
	--model_dimensions 512x512
	mkdir -p ../ganfaces/data/color_data/fairface_test/fairface_model
	python get_average_colors.py \
	--images_dir $(FAIRFACE_IMAGES_DIR) \
	--dest_csv $(COLOR_DATA_OUT_DIR)/fairface_test/fairface_model/fairface_test_fairface_stylegan3t_25000.csv \
	--gan_model ${FAIRFACE_STYLEGANT_FINAL_MODEL}  \
	--model_dimensions 512x512

${COLOR_DATA_OUT_DIR}/ffhq_test/fairface_model: get_average_colors.py
	mkdir -p $(COLOR_DATA_OUT_DIR)/ffhq_test/fairface_model
	python get_average_colors.py \
	--images_dir $(FFHQ_IMAGES_DIR) \
	--dest_csv $(COLOR_DATA_OUT_DIR)/ffhq_test/fairface_model/ffhq_test_fairface_stylegan3r_25000.csv \
	--gan_model ${FAIRFACE_STYLEGANR_FINAL_MODEL} \
	--model_dimensions 512x512
	cp -R $(COLOR_DATA_OUT_DIR)/ffhq_test/fairface_model ../ganfaces/data/color_data/
	python get_average_colors.py \
	--images_dir $(FFHQ_IMAGES_DIR) \
	--dest_csv $(COLOR_DATA_OUT_DIR)/ffhq_test/fairface_model/ffhq_test_fairface_stylegan3t_25000.csv \
	--gan_model ${FAIRFACE_STYLEGANT_FINAL_MODEL}  \
	--model_dimensions 512x512
	cp -R $(COLOR_DATA_OUT_DIR)/ffhq_test/fairface_model ../ganfaces/data/color_data/
	mv ../ganfaces/data/color_data/ffhq_test/fairface_model/*.csv ../ganfaces/data/color_data/ffhq_test/
	rmdir ../ganfaces/data/color_data/ffhq_test/fairface_model





${COLOR_DATA_OUT_DIR}: ${COLOR_DATA_OUT_DIR}/fairface_test ${COLOR_DATA_OUT_DIR}/ffhq_test
	echo "Generating color CSVs with FFHQ and FairFace images."



ffhq_prepare_512:
	mkdir -p $(IMAGES_ROOT_DIR)/ffhq_processed_512x512
	python dataset_tool.py \
	--source $(IMAGES_ROOT_DIR)/ffhq_images_1024x1024 \
	--dest $(IMAGES_ROOT_DIR)/ffhq_processed_512x512 \
	--resolution=512x512

ffhq_prepare_128:
	mkdir -p $(IMAGES_ROOT_DIR)/ffhq_processed_128x128
	python dataset_tool.py \
	--source $(IMAGES_ROOT_DIR)/ffhq_images_1024x1024 \
	--dest $(IMAGES_ROOT_DIR)/ffhq_processed_128x128 \
	--resolution=128x128


#fairface_clean:
#	rm -R $(MODELS_DIR)/fairface_stylegan3t/training-runs


fairface_train:	
	mkdir -p $(MODELS_DIR)/fairface_stylegan3t/training-runs
	python train.py --outdir=$(MODELS_DIR)/fairface_stylegan3/training-runs --cfg=stylegan3-t --data=$(FAIRFACE_IMAGES_DIR) --gpus=8 --batch=32 --gamma=8.2 --mirror=1 --snap=5

#make ffhq_clean:
#	rm -R $(MODELS_DIR)/ffhq_stylegan3t/training-runs
# Deleted this target because it's too dangerous.

ffhq_train_512_stylegant:
	mkdir -p $(MODELS_DIR)/ffhq_stylegan3t/training-runs
	python train.py --outdir=$(MODELS_DIR)/ffhq_stylegan3t/training-runs --data=$(IMAGES_ROOT_DIR)/ffhq_processed_512x512 --gpus=8 --batch=32 --gamma=8.2 --mirror=1 --snap=5 --cfg=stylegan3-t

ffhq_train_512_stylegan3r:
	mkdir -p $(MODELS_DIR)/ffhq_stylegan3t/training-runs
	python train.py --outdir=$(MODELS_DIR)/ffhq_stylegan3t/training-runs --data=$(IMAGES_ROOT_DIR)/ffhq_processed_512x512 --gpus=8 --batch=32 --gamma=8.2 --mirror=1 --snap=5 --cfg=stylegan3-r --seed 2


ffhq_train_512_stylegan3r_resume_latest:
	mkdir -p $(MODELS_DIR)/ffhq_stylegan3t/training-runs
	python train.py --outdir=$(MODELS_DIR)/ffhq_stylegan3t/training-runs --data=$(IMAGES_ROOT_DIR)/ffhq_processed_512x512 --gpus=8 --batch=32 --gamma=8.2 --mirror=1 --snap=5 --cfg=stylegan3-r --resume=latest


ffhq_train_128_stylegan3t:
	mkdir -p $(MODELS_DIR)/ffhq_stylegan3t/training-runs/128x128
	python train.py --outdir=$(MODELS_DIR)/ffhq_stylegan3t/training-runs/128x128 --data=$(IMAGES_ROOT_DIR)/ffhq_processed_128x128 --gpus=8 --batch=32 --gamma=8.2 --mirror=1 --snap=5 --cfg=stylegan3-t


ffhq_train_128_stylegan3r:
	mkdir -p $(MODELS_DIR)/ffhq_stylegan3t/training-runs/128x128
	python train.py --outdir=$(MODELS_DIR)/ffhq_stylegan3t/training-runs/128x128 --data=$(IMAGES_ROOT_DIR)/ffhq_processed_128x128 --gpus=8 --batch=32 --gamma=8.2 --mirror=1 --snap=5 --cfg=stylegan3-r



fairface_train_stylegan3t:	
	mkdir -p /mnt/data/students/models/fairface_stylegan3t/training-runs
	python train.py --outdir=/mnt/data/students/models/fairface_stylegan3/training-runs --cfg=stylegan3-t --data=/mnt/data/students/fairface/data/padding_0.25/train_prepared --gpus=8 --batch=32 --gamma=8.2 --mirror=1 --snap=5

fairface_train_stylegan3r:	
	mkdir -p /mnt/data/students/models/fairface_stylegan3t/training-runs
	python train.py --outdir=/mnt/data/students/models/fairface_stylegan3/training-runs --cfg=stylegan3-r --data=/mnt/data/students/fairface/data/padding_0.25/train_prepared --gpus=8 --batch=32 --gamma=8.2 --mirror=1 --snap=5


fairface_train_stylegan3r_128:	
	mkdir -p /mnt/data/students/models/fairface_stylegan3t/training-runs
	python train.py --outdir=/mnt/data/students/models/fairface_stylegan3/training-runs --cfg=stylegan3-r --data=/mnt/data/students/fairface/data/padding_0.25/train_prepared_128x128 --gpus=8 --batch=32 --gamma=8.2 --mirror=1 --snap=5


make ffhq_clean:
	rm -R /mnt/data/students/models/ffhq_stylegan3t/training-runs

ffhq_train_stylegan3t:
	mkdir -p /mnt/data/students/models/ffhq_stylegan3t/training-runs
	python train.py --outdir=/mnt/data/students/models/ffhq_stylegan3t/training-runs --data=/mnt/data/students/ffhq_processed_512x512 --gpus=8 --batch=32 --gamma=8.2 --mirror=1 --snap=5 --cfg=stylegan3-t


rsync_data:
	rsync -rv /mnt/data/students /share/kuran/data/nurvus_sync
