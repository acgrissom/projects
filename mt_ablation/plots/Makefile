install_packages:
	Rscript packages.R

new.csv:change_csv.py replace_long_names.sh
	python3 change_csv.py
	sh replace_long_names.sh

unshuffled.svg: plot.R
	touch plot.R
	Rscript plot.R

shuffled.svg: unshufled.svg
	touch plot.R

plots: unshuffled.svg
	touch plot.R

view:
	catimg unshuffled.svg.png 
