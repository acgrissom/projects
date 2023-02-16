OUT_DIR = results/figures

mac_packages:
	brew install inkscape catimg imagemagick libsm xquartz

linux_packages:
	apt install inkscape r-base caca-utils

r_packages:
	Rscript scripts/install_r_prereqs.R

results/figures/luminance_hdr.pdf: scripts/luminance_hdr.R
	Rscript scripts/luminance_hdr.R
	inkscape results/figures/luminance_hdr.svg --export-type=pdf --export-filename=results/figures/luminance_hdr.pdf
	catimg results/figures/luminance_hdr.jpg


results/figures/atypical_mens_face_dist.pdf: scripts/AtypicalFacePlotCode.R
	Rscript scripts/AtypicalFacePlotCode.R
	inkscape results/figures/atypical_mens_face_dist.svg --export-type=pdf --export-filename=results/figures/atypical_mens_face_dist.pdf
	inkscape results/figures/atypical_mens_face_hist.svg --export-type=pdf --export-filename=results/figures/atypical_mens_face_hist.pdf
	inkscape results/figures/atypical_mens_face_raincloud.svg --export-type=pdf --export-filename=results/figures/atypical_mens_face_raincloud.pdf
	inkscape results/figures/atypical_mens_face_dist_aggregate.svg --export-type=pdf --export-filename=results/figures/atypical_mens_face_dist_aggregate.pdf
	inkscape results/figures/atypical_mens_face_race_dist_aggregate.svg --export-type=pdf --export-filename=results/figures/atypical_mens_face_race_dist_aggregate.pdf
	catimg results/figures/atypical_mens_face_dist.jpg


results/figures: results/figures/luminance_hdr.pdf results/figures/atypical_mens_face_dist.pdf

clean:
	rm results/figures/*.*
