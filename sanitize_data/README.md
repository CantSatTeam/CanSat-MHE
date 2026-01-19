To use Vaihingen:

1. Download and unzip the Vaihingen data from [this Drive link](https://drive.google.com/drive/folders/1dcYlPGGAikhtSF0aZVL2okA-_AZNPTUl?usp=drive_link).
2. Move the folders `raw_dsm/` and `raw_image/` from the Vaihingen data into the `sanitize_data/` folder in the project directory.
3. Run `crop_isprs.py` to generate `dsm_cropped/` and `image_cropped/`.
4. Run `gen_data_split.py` to generate `data_split_dirs/`
4. Run `rename_cropped.py` to rename the cropped images and DSMs.
5. Move `dsm_cropped/` and `image_cropped/` to `HTC-DC-Net/data_dir/`.
6. Rename `dsm_cropped/` to `ndsm/` and `image_cropped/` to `image/`.
7. Move `data_split_dirs/` to `HTC-DC-Net/`.
