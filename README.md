# Fusion GAN
- Fusion GAN is an extension of vanillia GAN, it can be used to fuse patterns from different domains. 
- Current implementation is based on discretized sequence modeling, so can be directly applied for any problem of such kind, such as word sequence.
- Replacing the generator as you want, you can fuse any type of data. e.g., use CNN as generator, it's possible to fuse two different styles, say a fusion of Van Gogh and Picasso.

# Related papar
Codes for the paper 
> Zhiqian Chen, Chih-Wei Wu, Cheng-Yen Lu, Alexander Lerch, Chang-Tien Lu, Learning to Fuse Music Genres with Generative Adversarial Dual Learning, International Conference on Data Mining(ICDM), New Orleans, USA, 2017

# Demo
Please refer to https://imczq.com/publication/17_fusiongan_icdm/

# Detailed Manual (under construction)
required python package
```
numpy tensorflow magenta.music pandas midi music21 
```

Files description
- `discriminator.py`: GAN discriminator class
- `generator.py`: GAN generator class
- `dataloader.py`: data handler
- `midi_io.py`: music data process, such as MIDI to seq and seq to MIDI
- `rollout.py`: rollout for delayed update in reinforcement learning

Download all files and run

```python 
python fusion_gan.py

```

*.pkl is pre-processed files of music, but they cannot be recovered into original music

# Use your own training data
Please see `midi_io.py` in which there are functions for converting between MIDI and number sequences. 
Then, update the data path at `main` function of `fusion_gan.py`

# Citation
```
@article{fusiongan-icdm
  author    = {Zhiqian Chen and
               Chih{-}Wei Wu and
               Yen{-}Cheng Lu and
               Alexander Lerch and
               Chang{-}Tien Lu},
  title     = {Learning to Fuse Music Genres with Generative Adversarial Dual Learning},
  booktitle = {Proceedings of the The IEEE International Conference on Data Mining},
  year      = {2017},
}
```
