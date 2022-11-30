anaconda.sh:
	curl https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh > anaconda.sh
	sh anaconda.sh
	rm anaconda.sh

conda_environment:
	conda create --name transformers

conda_dependencies:
	conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
	conda install -c conda-forge tensorflow -y
	conda install -c huggingface -c conda-forge datasets -y
	conda install -c conda-forge jax -y
	conda install -c conda-forge ftfy -y
	pip install --upgrade diffusers transformers scipy --upgrade
	pip install accelerate --upgrade
	pip install spacy --upgrade
	pip install ftfy --upgrade
	pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
	pip install flax --upgrade


astronaut_rides_horse.png:
	python stable_diffusion_local.py

network_astronaut_rides_horse.png:
	python stable_diffusion_network.py

clean:
	rm -f astronaut_rides_horse.png
	rm -f anaconda.sh
