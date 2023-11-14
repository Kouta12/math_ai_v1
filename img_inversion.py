import cv2
import os, glob


dir_path = "./images/"

files_dir = [
    f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))
]

# ベースの画像を読み込み白黒反転
for DIR in files_dir:
    DIR_PATH = dir_path + DIR + "/"
    files = glob.glob(DIR_PATH + "*.jpg")
    for read_file in files:
        img = cv2.imread(read_file, cv2.IMREAD_GRAYSCALE)
        img = 255 - img
        cv2.imwrite(read_file, img)


