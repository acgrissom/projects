OUT_DIR = results/figures

mac_packages:
	brew install inkscape catimg imagemagick

linux_packages:
	apt install inkscape r-base caca-utils

r_packages:
	Rscript scripts/install_r_prereqs.R


results/figures/atypical_mens_face_dist.pdf:
	Rscript scripts/AtypicalFacePlotCode.R
	inkscape results/figures/atypical_mens_face_dist.svg --export-type=pdf --export-filename==atypical_mens_face_dist.pdf
	inkscape results/figures/atypical_mens_face_raincloud.svg --export-type=pdf --export-filename=results/figures/atypical_mens_face_raincloud.pdf
	convert results/figures/atypical_mens_face_dist.svg results/figures/atypical_mens_face_dist.jpg
	catimg results/figures/atypical_mens_face_dist.jpg



