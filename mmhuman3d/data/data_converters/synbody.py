import glob
import os
from typing import List

import cdflib
import h5py
import numpy as np
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS
import mmcv


@DATA_CONVERTERS.register_module()
class SynbodyConverter(BaseModeConverter):
    """Synbody dataset
    """
    ACCEPTED_MODES = ['train']

    def __init__(self, modes: List = [], merged_path='data/preprocessed_datasets', do_npz_merge=False, do_split=False) -> None:
        super(SynbodyConverter, self).__init__(modes)
        self.do_npz_merge = do_npz_merge
        # merged_path is the folder (will) contain merged npz
        self.merged_path = merged_path
        
        
    def _get_imgname(v):
        root_folder_id = v.split('/').index('synbody')
        imglist = '/'.join(v.split('/')[root_folder_id:])
        im = []
        for i in range(1, 61):
            imglist_tmp = os.path.join(imglist, 'rgb_resized')
            imglist_tmp = os.path.join(imglist_tmp, f'{i:04d}.jpeg')
            im.append(imglist_tmp)
        return im
    
    def _merge_npz(self, root_path):
        # root_path is where the npz files stored. Should ends with 'synbody'
        if not root_path.endswith('synbody'):
            root_path = os.path.join(root_path, 'synbody')
        ple = glob.glob(os.path.join(root_path, 'Synbody_0805*/'))
                            
        merged = {}
        merged['image_path'] = []
        merged['keypoints2d'] = []
        merged['keypoints3d'] = []
        merged['smpl'] = {}
        merged['smpl']['transl'] = []
        merged['smpl']['global_orient'] = []
        merged['smpl']['betas'] = []
        merged['smpl']['body_pose'] = []
        print(ple)
        for pl in tqdm(ple, desc='PLACE'):
            print(f'There are {len(ple)} places')
            for v in tqdm(glob.glob(pl + '/*'), desc='Video'):
                for p in tqdm(glob.glob(v + '/smpl_with_joints/*.npz'), desc='person'):
                    npfile_tmp = np.load(p, allow_pickle=True)
                    imgname = self._get_imgname(v)
                    merged['image_path'] += imgname
                    merged['smpl']['transl'].append(npfile_tmp['smpl'].item()['transl'][1:61])
                    merged['smpl']['global_orient'].append(npfile_tmp['smpl'].item()['global_orient'][1:61])
                    betas = npfile_tmp['smpl'].item()['betas']
                    betas = np.repeat(betas, 60, axis=0)
                    merged['smpl']['betas'].append(betas)
                    merged['smpl']['body_pose'].append(npfile_tmp['smpl'].item()['body_pose'][1:61])
                    merged['keypoints3d'].append(npfile_tmp['keypoints3d'][1:61])
                    merged['keypoints2d'].append(npfile_tmp['keypoints2d'][1:61])
                    
        for k in merged['smpl'].keys():
            merged['smpl'][k] = np.vstack(merged['smpl'][k])
        merged['keypoints3d'] = np.vstack(merged['keypoints3d'])
        merged['keypoints2d'] = np.vstack(merged['keypoints2d'])
        
        outpath = os.path.join(self.merged_path, 'synbody_train_merged.npz')
        os.makedirs(outpath, exist_ok=True)
        np.savez(outpath, **merged)
        return outpath
    
    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str) -> dict:
        """
        Args:
            dataset_path (str): Path to directory where raw images and
            annotations are stored.
            out_path (str): Path to directory to save preprocessed npz file
            mode (str): Mode in accepted modes

        Returns:
            dict:
                A dict containing keys image_path, bbox_xywh, keypoints2d,
                keypoints2d_mask, keypoints3d, keypoints3d_mask, cam_param
                stored in HumanData() format
        """
        # use HumanData to store all data
        human_data = HumanData()

        # structs we use
        if self.do_npz_merge:
            npfile = np.load(self._merge_npz(dataset_path), allow_pickle=True)
        else:
            npfile = np.load(os.path.join(self.merged_path, 'synbody_one_view_adult_54_trainset.npz'), allow_pickle=True)
        
        bbox_ = []
        for kp in npfile['keypoints2d']:
            # since the 2d keypoints are not strictly corrcet, a large scale factor is used
            bbox = self._keypoints_to_scaled_bbox(kp, 1.5)
            xmin, ymin, xmax, ymax = bbox
            bbox = np.array([max(0, xmin), max(0, ymin), min(1280, xmax), min(720, ymax)])
            bbox_xywh = self._xyxy2xywh(bbox)
            bbox_.append(bbox_xywh)
            
        bbox_ = np.array(bbox_).reshape((-1, 4))
        bbox_ = np.hstack([bbox_, np.ones([bbox_.shape[0], 1])])
        
        image_path_ = []
        for imp in npfile['image_path']:
            imp = imp.split('/')
            imp.remove('smpl')
            image_path_.append('/'.join(imp))
            
        human_data['image_path'] = image_path_
        human_data['bbox_xywh'] = bbox_
        keypoints2d_, mask = convert_kps(np.concatenate((npfile['keypoints2d'], np.ones_like(npfile['keypoints2d'][..., 0:1])), axis=2), 'smpl_45', 'human_data')
        keypoints3d_, mask = convert_kps(np.concatenate((npfile['keypoints3d'], np.ones_like(npfile['keypoints3d'][..., 0:1])), axis=2), 'smpl_54', 'human_data')
        human_data['keypoints2d'] = keypoints2d_
        human_data['keypoints3d'] = keypoints3d_
        human_data['keypoints2d_mask'] = mask
        human_data['keypoints3d_mask'] = mask
        
        smpl = {}
        for k in npfile['smpl'].item().keys():
            smpl[k] = npfile['smpl'].item()[k]
        human_data['smpl'] = smpl
        human_data['config'] = 'synbody'

        human_data.compress_keypoints_by_mask()
        # store the data struct
        os.makedirs(out_path,exist_ok=True)
        out_file = os.path.join(out_path, f'synbody_one_view_adult_54_trainset_humandata.npz')
        human_data.dump(out_file)
