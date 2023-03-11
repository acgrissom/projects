OUT_DIR = results/figures

pip_packages:
	pip install colormap easydev pyplot-themes pandas pandasql

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

results/figures/red_skin_grid.pdf: scripts/2d_score_correlation.R
	mkdir -p results/figures
	Rscript scripts/2d_score_correlation.R
	inkscape results/figures/red_skin_grid.svg --export-type=pdf --export-filename=results/figures/red_skin_grid.pdf
	catimg results/figures/red_skin_grid.jpg



results/figures/luminance_dist.pdf: scripts/luminance_hdr.pdf
	inkscape results/figures/luminance_dist.svg --export-type=pdf --export-filename=results/figures/luminance_dist.pdf
	catimg results/figures/luminance_dist.jpg



results/figures/atypical_womens_face_dist.pdf: scripts/woman_atypical_dist_plot.R
	mkdir -p results/figures
	Rscript scripts/woman_atypical_dist_plot.R
	inkscape results/figures/atypical_womens_face_dist.svg --export-type=pdf --export-filename=results/figures/atypical_womens_face_dist.pdf
	inkscape results/figures/atypical_womens_face_hist.svg --export-type=pdf --export-filename=results/figures/atypical_womens_face_hist.pdf
	inkscape results/figures/atypical_womens_face_raincloud.svg --export-type=pdf --export-filename=results/figures/atypical_womens_face_raincloud.pdf
	inkscape results/figures/atypical_womens_face_dist_aggregate.svg --export-type=pdf --export-filename=results/figures/atypical_womens_face_dist_aggregate.pdf
	inkscape results/figures/atypical_womens_face_race_dist_aggregate.svg --export-type=pdf --export-filename=results/figures/atypical_womens_face_race_dist_aggregate.pdf
	catimg results/figures/atypical_mens_face_dist.jpg

results/figures/atypical_all_50face_unbalanced_dist.pdf: results/figures/atypical_all_50face_dist.pdf
	inkscape results/figures/atypical_all_50face_unbalanced_dist.svg --export-type=pdf --export-filename=results/figures/atypical_all_50face_unbalanced_dist.pdf
	catimg results/figures/atypical_all_50face_unbalanced_dist.jpg

results/figures/atypical_all_50face_dist.pdf: scripts/atypical_plot_cleaned.R
	mkdir -p results/figures
	Rscript scripts/atypical_plot_cleaned.R
	inkscape results/figures/atypical_all_50face_dist.svg --export-type=pdf --export-filename=results/figures/atypical_all_50face_dist.pdf
	catimg results/figures/atypical_all_50face_dist.jpg


results/figures/atypical_mens_face_dist.pdf: scripts/AtypicalFacePlotCode.R
	mkdir -p results/figures
	Rscript scripts/AtypicalFacePlotCode.R
	inkscape results/figures/atypical_mens_face_dist.svg --export-type=pdf --export-filename=results/figures/atypical_mens_face_dist.pdf
	inkscape results/figures/atypical_mens_face_hist.svg --export-type=pdf --export-filename=results/figures/atypical_mens_face_hist.pdf
	inkscape results/figures/atypical_mens_face_raincloud.svg --export-type=pdf --export-filename=results/figures/atypical_mens_face_raincloud.pdf
	inkscape results/figures/atypical_mens_face_dist_aggregate.svg --export-type=pdf --export-filename=results/figures/atypical_mens_face_dist_aggregate.pdf
	inkscape results/figures/atypical_mens_face_race_dist_aggregate.svg --export-type=pdf --export-filename=results/figures/atypical_mens_face_race_dist_aggregate.pdf
	catimg results/figures/atypical_mens_face_dist.jpg

results/figures/rgb_posteriors_hdi.pdf: /results/figures/rgb_posteriors.pdf
	inkscape results/figures/rgb_posteriors_hdi.svg --export-type=pdf --export-filename=results/figures/rgb_posteriors_hdi.pdf


results/figures/rgb_posteriors.pdf: scripts/rgb_linear_regression.py
	mkdir -p results/figures
	python scripts/rgb_linear_regression.py
	inkscape results/figures/rgb_posteriors.svg --export-type=pdf --export-filename=results/figures/rgb_posteriors.pdf

	catimg results/figures/rgb_posteriors.jpg

results/figures/race_hair_posteriors.pdf: scripts/race_hair_linear_regression.py
	mkdir -p results/figures
	python scripts/race_hair_linear_regression.py
	inkscape results/figures/race_hair_posteriors.svg --export-type=pdf --export-filename=results/figures/race_hair_posteriors.pdf
	catimg results/figures/race_hair_posteriors.jpg

results/figures/ordinal_posteriors.pdf: scripts/race_hair_linear_regression.py
	mkdir -p results/figures
	python scripts/ordinal_regression.py
	inkscape results/figures/ordinal_posteriors.svg --export-type=pdf --export-filename=results/figures/ordinal_posteriors.pdf
	catimg results/figures/ordinal_posteriors.jpg



results/figures: results/figures/luminance_hdr.pdf results/figures/training_color_histogram_logscale.pdf results/figures/training_color_scatter.png scripts/color_histogram.py results/figures/luminance_hdr.pdf results/figures/atypical_all_50face_dist.pdf results/figures/red_skin_grid.pdf

git_update:
	git add scripts/*.R scripts/*.py
	git add results/figures/*.svg
	git commit
	git push
clean:
	rm -rf results/figures
