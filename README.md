Using pix2pix to map sketches into frogs.

DexiNed-TF2 performs edge detection, creating the dataset of sketches from images.

1. Download Images
2. DexiNed Edge Detection
    <code>python src/DexiNed-TF2/main.py --model_state="test" --test_dir="data/raw/data-224" --output_dir="data/interim" --checkpoint_dir="src/DexiNed-TF2" --checkpoint="DexiNed23_model.h5" --test_img_height=224 --test_img_width=224 --test_bs=32</code>
3. Canny Edge Detection - to further simplify sketches
4. Build Data Input Pipeline
5. Train Pix2Pix


Useful links:
https://ml4a.github.io/guides/Pix2Pix/
https://heartbeat.fritz.ai/training-a-tiny-pix2pix-gan-for-snapchat-c6062e86204c
https://arxiv.org/pdf/1504.06375.pdf - Holistically-Nested Edge Detection
https://affinelayer.com/pixsrv/ - pix2pix demo
https://github.com/jonshamir/frog-dataset


pix2pix demo with implementation code
https://zaidalyafeai.github.io/pix2pix/cats.html#
https://github.com/zaidalyafeai/zaidalyafeai.github.io/tree/master/pix2pix

GauGAN - SPADE
https://arxiv.org/pdf/1903.07291.pdf