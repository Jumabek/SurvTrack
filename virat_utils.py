from loguru import logger
import motmetrics as mm
from os.path import join
import pandas as pd
from collections import OrderedDict
from glob import glob
from tqdm import tqdm
import os 
import ntpath 

datafolder  = '/mnt/data/survTrack/assets/customdata/virat'
annotation_folder = join(datafolder,'annotations')


def read_virat_gt(gt_tracks_path):
    sep = ' '
    min_confidence = -1
    gt = pd.read_csv(
    gt_tracks_path,
    sep=sep,
    skipinitialspace=True,
    header=None,
    names=['Id','obj_duration','FrameId', 'X', 'Y', 'Width', 'Height', 'ClassId'],
    engine='python'
    )

    # Account for matlab convention.
    gt[['X', 'Y']] -= (1, 1)
    gt = gt.drop(columns=['obj_duration'],axis=0)

    gt.set_index(['FrameId', 'Id'], inplace=True)

    gt.insert(4, 'Confidence', 1)
    gt.insert(5, 'Visibility', -1)
    gt.sort_index(inplace=True)
    gt['ClassId'] = gt['ClassId'].replace(1, 0)
    gt = gt[~gt['ClassId'].isin([4, 5])]

    return gt



def load_gts():
    vidnames = [ntpath.basename(f)[:-4] for f in glob(join(datafolder,'videos', '*.mp4'))]

    gts = OrderedDict([
        (
            vidname
            ,read_virat_gt(join(annotation_folder, vidname + '.viratdata.objects.txt'))
        ) for vidname in tqdm(vidnames)
    ])

    return gts



def compute_metrics_for_video(gt_data, pred_data):
    gt_df = gt_data.reset_index().rename(columns={'FrameId': 'FrameId', 'Id': 'ObjId', 'X': 'X', 'Y': 'Y', 'Width': 'Width', 'Height': 'Height'})
    pred_df = pred_data.reset_index().rename(columns={'FrameId': 'FrameId', 'Id': 'ObjId', 'X': 'X', 'Y': 'Y', 'Width': 'Width', 'Height': 'Height'})

    acc = mm.MOTAccumulator(auto_id=True)

    for frame_id in gt_df['FrameId'].unique():
        gt_frame = gt_df[gt_df['FrameId'] == frame_id]
        pred_frame = pred_df[pred_df['FrameId'] == frame_id]
        
        gt_boxes = gt_frame[['X', 'Y', 'Width', 'Height']].to_numpy()
        pred_boxes = pred_frame[['X', 'Y', 'Width', 'Height']].to_numpy()
        
        iou_matrix = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=1.0)
        
        acc.update(gt_frame['ObjId'], pred_frame['ObjId'], iou_matrix)

    mh = mm.metrics.create()

    # Compute metrics
    summary = mh.compute(acc, metrics=['motp', 'mota', 'idf1'], name='Metrics')
    motp = summary.loc['Metrics', 'motp']*100
    mota = summary.loc['Metrics', 'mota']*100
    idf1 = summary.loc['Metrics', 'idf1']*100

    

    return {'motp': motp, 'mota': mota, 'idf1': idf1}


import matplotlib.pyplot as plt
def matplotlib_setup(fontsize=35):
    font = {'size'   : fontsize}
    plt.rc('font', **font)
    plt.rcParams["axes.linewidth"]  = 2.5
    plt.grid(linewidth=3,axis='y', color='grey')

    CB91_Blue = '#2CBDFE'
    CB91_Green = '#47DBCD'
    CB91_Pink = '#F3A0F2'
    CB91_Purple = '#9D2EC5'
    CB91_Violet = '#661D98'
    CB91_Amber = '#F5B14C'
    color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
              CB91_Purple, CB91_Violet]
    
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)    