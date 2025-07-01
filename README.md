### Folders

**bh_emsenble/** - Naive ensemble test on black hole model. Stores the 100 models and their training logs.

**datasets/** - EHT data for black hole imaging 

**inpainting/** - Test jax implementation on simple images

### Files (in root directory)

**bayes_rays_blackhole.ipynb** - 2D test implementation of bayes rays on black hole imaging problem. 

**bayes_rays_holdout.ipynb** - 2D test implementation of bayes rays on the image of the black hole, holding out half of the pixels to verify it captures the uncertainty correctly.

**black_hole_params.msgpack** - Saved model parameters for the black hole imaging problem (training takes a while, this is to save time when loading up a fresh notebook)

**neural_image_example.ipynb** - Example implementation from Brandon Zhao. I plot the naive ensemble results + std map and calibration plot here. 
