Use the Makefile to install Anaconda if it is not alrady installed.  Then run `make conda_environment` and then `conda init`. Exit your shell, and log in again.


Once in the shell, to enter the Anaconda sandbox, type `conda activate transformers` to enter the huggingface sandbox.

From there, use `make conda_dependencies` to install all of the necessary dependencies in the sandbox.  You should then be able to run `python stable_diffusion.py`.  The example code is taken from the following site.

https://huggingface.co/CompVis/stable-diffusion-v1-4