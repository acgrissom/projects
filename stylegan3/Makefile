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



