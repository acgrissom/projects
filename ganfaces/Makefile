OUT_DIR = results/figures

mac_packages:
	brew install imagemagick

linux_packages:
	apt install imagemagick
	apt install r-base
	apt isntall caca-utils
r_packages:
	Rscript scripts/install_r_prereqs.R


results/figures/atypical_mens_face_dist.pdf:
	Rscript scripts/AtypicalFacePlotCode.R
	convert results/figures/atypical_mens_face_dist.svg atypical_mens_face_dist.pdf
	convert results/figures/atypical_mens_faces_raincloud.svg results/figures/atypical_mens_faces_raincloud.pdf
	convert results/figures/atypical_mens_face_dist.svg | lpr



