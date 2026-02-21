import os

dirs = ["image_cropped", "dsm_cropped"]

### old, for htc-dc net
# suffix = ["IMG", "AGL"]
# for dir_name, suf in zip(dirs, suffix):
#     for filename in os.listdir(dir_name):
#         base, ext = os.path.splitext(filename)
#         new_name = f"{base.split('_')[4].zfill(4)}_{suf}{ext}"
#         os.rename(os.path.join(dir_name, filename), os.path.join(dir_name, new_name))

prefix = ["image", "dsm"]
for dir_name, pre in zip(dirs, prefix):
    for filename in os.listdir(dir_name):
        base, ext = os.path.splitext(filename)
        id = base.split('_')[4].zfill(4)
        new_name = f"{pre}{id}{ext}"
        os.rename(os.path.join(dir_name, filename), os.path.join(dir_name, new_name))
