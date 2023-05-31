OUT_DIR = results/figures

pip_packages:
	pip install colormap easydev pandas pandasql arviz seaborn

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

results/figures/fairface_color_histogram_logscale.pdf:
	mkdir -p results/figures
	python scripts/color_histogram.py --input_csv data/fairface_data.csv --output_filename results/figures/fairface --num_bins=10
	inkscape results/figures/fairface_color_histogram_logscale.svg --export-type=pdf --export-filename=results/figures/fairface_color_histogram_logscale.pdf
	catimg results/figures/fairface_color_histogram_logscale.jpg

results/figures/fairface_color_histogram_ffhq_stylegan3r_model_logscale.pdf:
	mkdir -p results/figures
	python scripts/color_histogram.py --input_csv data/fairface_data_ffhq_stylegan3r_model.csv --output_filename results/figures/fairface_color_ffhq_stylegan3r_model --num_bins=10
	inkscape results/figures/fairface_color_ffhq_stylegan3r_model_color_histogram_logscale.svg --export-type=pdf --export-filename=results/figures/fairface_color_ffhq_stylegan3r_model_color_histogram_logscale.pdf
	catimg results/figures/fairface_color_ffhq_stylegan3r_model_color_histogram_logscale.jpg



results/figures/ffhq_color_histogram_logscale.pdf:
	mkdir -p results/figures
	python scripts/color_histogram.py --input_csv data/ffhq_color_data.csv --output_filename results/figures/ffhq --num_bins=20
	inkscape results/figures/ffhq_color_histogram_logscale.svg --export-type=pdf --export-filename=results/figures/ffhq_color_histogram_logscale.pdf
	catimg results/figures/ffhq_color_histogram_logscale.jpg



results/figures/training_color_scatter.png:
	mkdir -p results/figures
	python scripts/color_histogram.py
	catimg results/figures/training_color_scatter.png

results/figures/ffhq_luminance_hdr.pdf: scripts/luminance_hdr.R
	mkdir -p results/figures
	Rscript scripts/luminance_hdr.R data/correct_LAB_format_images_data.csv results/figures/ffhq
	inkscape results/figures/ffhq_luminance_hdr.svg --export-type=pdf --export-filename=results/figures/ffhq_luminance_hdr.pdf
	catimg results/figures/ffhq_luminance_hdr.jpg


results/figures/fairface_luminance_hdr.pdf: scripts/luminance_hdr.R
	mkdir -p results/figures
	Rscript scripts/luminance_hdr.R data/fairface_data.csv results/figures/fairface
	inkscape results/figures/fairface_luminance_hdr.svg --export-type=pdf --export-filename=results/figures/fairface_luminance_hdr.pdf
	catimg results/figures/fairface_luminance_hdr.jpg


results/figures/red_skin_grid.pdf: scripts/2d_score_correlation.R
	mkdir -p results/figures
	Rscript scripts/2d_score_correlation.R
	inkscape results/figures/red_skin_grid.svg --export-type=pdf --export-filename=results/figures/red_skin_grid.pdf
	catimg results/figures/red_skin_grid.jpg



results/figures/ffhq_luminance_dist.pdf: scripts/ffhq_luminance_hdr.pdf
	catimg results/figures/ffhq_luminance_dist.jpg

results/figures/fairface_luminance_dist.pdf: scripts/fairface_luminance_hdr.pdf
	catimg results/figures/fairface_luminance_dist.jpg




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



results/figures:  results/figures/training_color_histogram_logscale.pdf results/figures/training_color_scatter.png scripts/color_histogram.py results/figures/ffhq_luminance_hdr.pdf results/figures/fairface_luminance_hdr.pdf results/figures/atypical_all_50face_dist.pdf results/figures/red_skin_grid.pdf results/figures/ffhq_color_histogram_logscale.pdf results/figures/fairface_color_histogram_logscale.pdf

git_update:
	git add scripts/*.R scripts/*.py
	git add results/figures/*.svg
	git commit
	git push
clean:
	rm -rf results/figures
