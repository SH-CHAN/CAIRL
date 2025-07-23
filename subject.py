########## DATA EXAMPLE ##########

subject_icedrug = ['R3S10',...]

control_icedrug = ['R3C1', ...]

# subjects = subject_icedrug + control_icedrug

# ice drug tms pre and post (one-to-one correspondence)
tms_icedrug_pre = ['R3S10',...]
tms_icedrug_post = ['R3P1',...]


# ice drug notms pre and post (one-to-one correspondence)
notms_icedrug_pre = ['R3S14',...]

notms_icedrug_post = ['R3PC1',...]
########## DATA EXAMPLE ##########

prep_group = tms_icedrug_pre + notms_icedrug_pre
post_group = tms_icedrug_post + notms_icedrug_post
subjects = list(set(prep_group + post_group))

img_position =[[ 397.75606431,   38.13005359,  400.,          400.,        ],
                  [1122.24393569,   38.13005359,  400.,          400.,        ],
                  [1122.24393569,  641.86994641,  400.,          400.,        ],
                  [ 397.75606431,  641.86994641,  400.,          400.,        ]]
screen_size = [1920.0 , 1080.0 ]

import pandas as pd
import numpy as np
image_heroin_control = np.array(pd.read_csv('/root/autodl-tmp/ImageClassification/image_list/heroin_control_img.csv').iloc[:,0])
image_heroin_cue     = np.array(pd.read_csv('/root/autodl-tmp/ImageClassification/image_list/heroin_cue_img.csv').iloc[:,0])
image_ice_drug_control = np.array(pd.read_csv('/root/autodl-tmp/ImageClassification/image_list/ice_drug_control_img.csv').iloc[:,0])
image_ice_drug_cue     = np.array(pd.read_csv('/root/autodl-tmp/ImageClassification/image_list/ice_drug_cue_img.csv').iloc[:,0])

import torch
# from utils import
def precompute_IOR_templates(grid_H=27, grid_W=48, sigma=1):
    y_idx = torch.arange(grid_H).view(-1, 1).expand(grid_H, grid_W)  # row
    x_idx = torch.arange(grid_W).view(1, -1).expand(grid_H, grid_W)  # col

    templates = {}
    for i in range(grid_H):
        for j in range(grid_W):
            dist_sq = (x_idx - j) ** 2 + (y_idx - i) ** 2
            template = torch.exp(-dist_sq / (2 * sigma**2))  # shape: (27, 48)
            templates[(i, j)] = template
    return templates
templates = precompute_IOR_templates()

def compute_grid_mask_index(grid_size=(27, 48), screen_size=(1920, 1080), img_boxes=img_position):
    rows, cols = grid_size
    screen_width, screen_height = screen_size
    grid_w = screen_width / cols
    grid_h = screen_height / rows

    # -1 表示不属于任何框，类型为 int64
    mask = torch.full((rows * cols,), -1, dtype=torch.int64)

    for idx in range(rows * cols):
        row = idx // cols
        col = idx % cols

        # 将 row 从左上角编号转换为从左下角编号
        flipped_row = rows - 1 - row

        # 网格中心点坐标（左下角为原点）
        x_center = col * grid_w + grid_w / 2
        y_center = flipped_row * grid_h + grid_h / 2

        for i, (x, y, w, h) in enumerate(img_boxes):
            if x <= x_center <= x + w and y <= y_center <= y + h:
                mask[idx] = i  # 设置为框的编号（0~3）
                break
    return mask
screen_mask_index = compute_grid_mask_index()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

