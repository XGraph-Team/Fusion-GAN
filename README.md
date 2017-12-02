# Fusion GAN
- Fusion GAN is an extension of vanillia GAN, it can be used to fuse patterns from different domains. 
- Current implementation is based on discretized sequence modeling, so can be directly applied for any problem of such kind, such as word sequence.
- Replacing the generator as you want, you can fuse any type of data. e.g., use CNN as generator, it's possible to fuse two different styles, say a fusion of Van Gogh and Picasso.

# Related papaer
Codes for the paper 
> Zhiqian Chen, Chih-Wei Wu, Cheng-Yen Lu, Alexander Lerch, Chang-Tien Lu, Learning to Fuse Music Genres with Generative Adversarial Dual Learning, International Conference on Data Mining(ICDM), New Orleans, USA, 2017

# Demo
Please refer to http://people.cs.vt.edu/czq/publication/fusiongan/

# Detailed Manual (under construction)
Download all files and run

```python 
python fusion_gan.py

```

*.pkl is pre-processed files of music, but they cannot be recovered into original music
