Use the Makefile to install Anaconda if it is not alrady installed.  Then run `make conda_environment` and then `conda init`. Exit your shell, and log in again.


Once in the shell, to enter the Anaconda sandbox, type `conda activate transformers` to enter the huggingface sandbox.

From there, use `make conda_dependencies` to install all of the necessary dependencies in the sandbox.  

You should then be able to run `make astronaut_rides_horse.png`.  This will use the local download on Kuran to do the rendering.  There's an alternate version that uses the network-hosted one, but this isn't recommended for doing many renders.

The example code is taken from the following site: https://huggingface.co/CompVis/stable-diffusion-v1-4

Also see the following for code explanations:

* Diffusers blog post: https://huggingface.co/docs/diffusers/conceptual/stable_diffusion
* Documentation: https://huggingface.co/blog/stable_diffusion
