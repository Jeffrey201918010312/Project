import math
import random

from tools.datatools import random_transfer
from tools.utils import mkdir, imread, imwrite, get_pics_list
import os
from multiprocessing import pool,cpu_count
from tqdm import tqdm

def gen_images(save_folder, img_path, num=100):
    '''
    Generate num sheets of expanded data based on image paths
    :param save_folder: Save path
    :param img_path: Input image path
    :param num: Number of generators
    :return:none
    '''
    img_dir = os.path.dirname(img_path)
    save_dir = mkdir(img_dir.replace('data', save_folder))
    img = imread(img_path)
    base_name = os.path.basename(img_path).split(".")[0] + ".jpg"
    if num == 0:
        save_path = os.path.join(save_dir, base_name)
        imwrite(save_path, img)
    i = 0
    while i < num:
        img = random_transfer(img)
        save_path = os.path.join(save_dir, str(i) + base_name)
        imwrite(save_path, img)
        i += 1

def enhance_images(root_dir, folder):
    '''
    Enter all images in the specified folder expansion
    :param root_dir: root
    :param folder:
    :return:
    '''
    img_list = get_pics_list(root_dir, folder)
    
    seed = 100
    random.seed(seed)
    random.shuffle(img_list)
    num = int(len(img_list) * 0.7)
    test_list = img_list[num:]  
    for img_path in tqdm(test_list):
        try:
            gen_images('datasets//test', img_path, 0)
        except:
            continue
    img_list = img_list[:num]
    if len(img_list) == 0:
        return
    total_num = 2000
    ech_num = total_num / len(img_list)
    for img_path in tqdm(img_list):
        try:
            gen_images('datasets//train', img_path, math.ceil(ech_num))
        except:
            continue


if __name__ == '__main__':
    folder_list = os.listdir("data")
    # p = pool.Pool(cpu_count())
    for folder in folder_list:
        # p.apply_async(enhance_images, ('data', folder,))
        enhance_images('data', folder)
    # p.close()
    # p.join()