Using pix2pix to map sketches into frogs.

DexiNed-TF2 performs edge detection, creating the dataset of sketches from images.

1. Download Images
2. DexiNed Edge Detection
    <code>python src/DexiNed-TF2/main.py --model_state="test" --test_dir="data/raw/data-224" --output_dir="data/interim" --checkpoint_dir="src/DexiNed-TF2" --checkpoint="DexiNed23_model.h5" --test_img_height=224 --test_img_width=224 --test_bs=32</code>
3. Canny Edge Detection - to further simplify sketches
4. Build Data Input Pipeline
5. Train Pix2Pix