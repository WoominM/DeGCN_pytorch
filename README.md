<h1 align="center"> DeGCN: Deformable Graph Convolutional Networks for Skeleton-Based Action Recognition </h1>
<p align="center">
<a href="https://ieeexplore.ieee.org/document/10478824"><img src="https://img.shields.io/badge/IEEE-Paper-blue"></a>
<a href="https://paperswithcode.com/sota/skeleton-based-action-recognition-on-ntu-rgbd-1?p=degcn-deformable-graph-convolutional-networks"><img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/degcn-deformable-graph-convolutional-networks/skeleton-based-action-recognition-on-ntu-rgbd-1"></a>
</p>

Official PyTorch implementation of "DeGCN : Deformable Graph Convolutional Networks for Skeleton-Based Action Recognition"

# Prerequisites

- Python >= 3.6
- PyTorch >= 1.1.0
- PyYAML, tqdm, tensorboardX, h5py, sklearn, matplotlib, thop
- Run `pip install -e torchpack`
- Run `pip install -e torchlight` 

# Data Preparation

### Download datasets.

#### There are 2 datasets to download:

- NTU RGB+D 60 Skeleton
- NTU RGB+D 120 Skeleton

1. Request dataset here: https://rose1.ntu.edu.sg/dataset/actionRecognition
2. Download the skeleton-only datasets:
   1. `nturgbd_skeletons_s001_to_s017.zip` (NTU RGB+D 60)
   2. `nturgbd_skeletons_s018_to_s032.zip` (NTU RGB+D 120)
   3. Extract above files to `./data/nturgbd_raw`

### Data Processing

#### Directory Structure

Put downloaded data into the following directory structure:

```
- data/
  - ntu/
  - ntu120/
  - nturgbd_raw/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      ...
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
      ...
```

#### Generating Data

- Generate NTU RGB+D 60 or NTU RGB+D 120 dataset:

```
 cd ./data/ntu # or cd ./data/ntu120
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
```


# Training & Testing

### Training

- Change the config file depending on what you want.

```
# Example: training DeGCN on NTU RGB+D 120 cross subject with GPU 0
python main.py --config config/nturgbd120-cross-subject/default.yaml --work-dir work_dir/ntu120/csub/degcn --device 0
```

- To train model on NTU RGB+D 60/120 with bone or motion modalities, setting `bone` or `vel` arguments in the config file `default.yaml` or in the command line.

```
# Example: training DeGCN on NTU RGB+D 120 cross subject under bone modality
python main.py --config config/nturgbd120-cross-subject/default.yaml --train-feeder-args bone=True --test-feeder-args bone=True --work-dir work_dir/ntu120/csub/degcn_bone --device 0
```

- To train model the JBF stream, setting `model` arguments in the config file `default.yaml` or in the command line.

```
# Example: training DeGCN with the JBF stream on NTU RGB+D 120 cross subject
python main.py --config config/nturgbd120-cross-subject/default.yaml --model model.jbf.Model --work-dir work_dir/ntu120/csub/degcn_bone --device 0
```

- To train your own model, put model file `your_model.py` under `./model` and run:

```
# Example: training your own model on NTU RGB+D 120 cross subject
python main.py --config config/nturgbd120-cross-subject/default.yaml --model model.your_model.Model --work-dir work_dir/ntu120/csub/your_model --device 0
```

### Testing

- To test the trained models saved in <work_dir>, run the following command:

```
python main.py --config <work_dir>/config.yaml --work-dir <work_dir> --phase test --save-score True --weights <work_dir>/xxx.pt --device 0
```

- To ensemble the results of different modalities, run 
```
# Example: ensemble four modalities of DeGCN on NTU RGB+D 120 cross subject
python ensemble.py --datasets ntu120/xsub --joint-dir work_dir/ntu120/xsub/degcn --bone-dir work_dir/ntu120/xsub/degcn_bone --joint-motion-dir work_dir/ntu120/xsub/degcn_motion
```

<!-- ### Pretrained Models

- Download pretrained models for producing the final results on NTU RGB+D 60&120 cross subject .
- Put files to <work_dir> and run **Testing** command to produce the final result. -->


## Acknowledgements

This repo is based on [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN). The data processing is borrowed from [SGN](https://github.com/microsoft/SGN) and [HCN](https://github.com/huguyuehuhu/HCN-pytorch).

Thanks to the original authors for their work!


# Citation

Please cite this work if you find it useful:.

      @inproceedings{,
        title={DeGCN: Deformable Graph Convolutional Networks for Skeleton-Based Action Recognition},
        author={Woomin Myung, Nan Su, Jing-Hao Xue, Guijin Wang},
        journal={IEEE transactions on image processing (TIP)}
        year={2024}
      }
