OUT_DIR = results/figures

pip_packages:
	pip install colormap easydev pyplot-themes

mac_packages: pip_packages
	brew install inkscape catimg imagemagick libsm xquartz

linux_packages: pip_packages
	apt update
	apt install inkscape r-base caca-utils r-cran-lme4 r-cran-car r-cran-gmp r-cran-rcurl

r_packages: 
	Rscript scripts/install_r_prereqs.R

results/figures/training_color_histogram_logscale.pdf:
	mkdir -p results/figures
	python scripts/color_histogram.py
	inkscape results/figures/training_color_histogram_logscale.svg --export-type=pdf --export-filename=results/figures/training_color_histogram_logscale.pdf
	catimg results/figures/training_color_histogram_logscale.jpg


results/figures/training_color_scatter.png:
	mkdir -p results/figures
	python scripts/color_histogram.py
	catimg results/figures/training_color_scatter.png

results/figures/luminance_hdr.pdf: scripts/luminance_hdr.R
	mkdir -p results/figures
	Rscript scripts/luminance_hdr.R
	inkscape results/figures/luminance_hdr.svg --export-type=pdf --export-filename=results/figures/luminance_hdr.pdf
	catimg results/figures/luminance_hdr.jpg


results/figures/luminance_dist.pdf: scripts/luminance_hdr.pdf
	inkscape results/figures/luminance_dist.svg --export-type=pdf --export-filename=results/figures/luminance_dist.pdf
	catimg results/figures/luminance_dist.jpg


results/figures/atypical_mens_face_dist.pdf: scripts/AtypicalFacePlotCode.R
	mkdir -p results/figures
	Rscript scripts/AtypicalFacePlotCode.R
	inkscape results/figures/atypical_mens_face_dist.svg --export-type=pdf --export-filename=results/figures/atypical_mens_face_dist.pdf
	inkscape results/figures/atypical_mens_face_hist.svg --export-type=pdf --export-filename=results/figures/atypical_mens_face_hist.pdf
	inkscape results/figures/atypical_mens_face_raincloud.svg --export-type=pdf --export-filename=results/figures/atypical_mens_face_raincloud.pdf
	inkscape results/figures/atypical_mens_face_dist_aggregate.svg --export-type=pdf --export-filename=results/figures/atypical_mens_face_dist_aggregate.pdf
	inkscape results/figures/atypical_mens_face_race_dist_aggregate.svg --export-type=pdf --export-filename=results/figures/atypical_mens_face_race_dist_aggregate.pdf
	catimg results/figures/atypical_mens_face_dist.jpg


results/figures: results/figures/luminance_hdr.pdf results/figures/atypical_mens_face_dist.pdf results/figures/training_color_histogram_logscale.pdf results/figures/training_color_scatter.png scripts/color_histogram.py results/figures/luminance_hdr.pdf

git_update:
	git add scripts/*.R scripts/*.py
	git add results/figures/*.svg
	git commit
	git push
clean:
	rm -rf results/figures
