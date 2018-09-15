# Reinforced Video Captioning with Entailment Rewards (EMNLP 2017)
This repository contains the re-implementation along with improved results of the paper: [Reinforced Video Captioning with Entailment Rewards](https://arxiv.org/abs/1708.02300) (EMNLP 2017)

This code is tested on python 3.6 and pytorch 0.3.

## Setup:
Install all the required packages from requirements.txt file.

#### Datasets:
Download the ResNet-152 frame-level features + ResNeXt-101 motion features for [MSR-VTT](https://drive.google.com/file/d/1bZJ0noxJ9EwXV161d4w6p6PqaszhM8t4/view?usp=sharing) videos.

Download the captions and vocabulary data for [MSR-VTT](https://drive.google.com/drive/folders/1HhF8Tl3ZXQzaILlg6vCXST9A_6BjPU5r?usp=sharing), and place the downloaded data in 'data' folder. 

### Evaluation:
Clone the code from *[here](https://github.com/ramakanth-pasunuru/video_caption_eval_python3)* to setup the evaluation metrics, and place it the parent directory on this repository. Note that this is also required during training since tensorboard logs the validation scores.


## Run Code:
To train Baseline-XE model:
```
python main.py --model_name "model_name"
```
To train CIDEr-RL model:
```
python main.py --model_name "model_name" --load_path "path_to_baseline_model_folder" --lr 0.00001 --reward_type CIDEr --max_epoch 40 --loss_function xe+rl
```
To train CIDEnt-RL model:
```
python main.py --model_name "model_name" --load_path "path_to_baseline_model_folder" --load_entailment_path "path_to_entailment_model_end_with_*pth" --lr 0.00001 --reward_type CIDEnt --max_epoch 40 --loss_function xe+rl
```
For testing:
```
python main.py --mode test --load_path "path_to_model_folder" --beam_size 5 
```

## Pretrained Models

Download the pretrained models for Baseline, CIDEr-RL, and CIDEnt-RL from [here](https://drive.google.com/open?id=1Zl5jDAo6to1bRoNi_HtzQumEG27U62RD).

For running the pretrained models:
```
python main.py --mode test --load_path "path_to_model_ending_with_*.pth" --beam_size 5 
```

## MSR-VTT Results

On running the above given pretrained models you should achieve the following results:

| Models             | CIDEr | BLEU-4 | METEOR | ROUGE |
| ------------------ | ----- | ------ | ------ | ----- |
| Baseline-XE        | 48.2  | 40.8   | 28.1   | 60.7  |
| CIDEr-RL           | 52.5  | 41.8   | 28.0   | 62.2  |
| CIDEnt-RL          | 53.0  | 42.2   | 28.2   | 62.3  |

Note that first our CIDEr-RL model achieves stat. signif. improvements over Baseline-XE (on CIDEr, BLEU, and ROUGE) and then our CIDEnt-RL model achieves stat. signif. improvements over CIDEr-RL (on CIDEr, BLEU, and METEOR).

**Note that this released baseline model has several improvements (and significantly better results) w.r.t. our EMNLP17 paper results, due to the following enhancements:**

- Better visual features: (1) ResNet-152 frame-level features (2) ResNeXt-101 motion features
- Used [SCST](https://arxiv.org/pdf/1612.00563.pdf) approach instead of [MIXER](https://arxiv.org/pdf/1511.06732.pdf) for reinforcement leanring
- Used better entailment classifier


## References
If you find this code helpful, please consider citing the following papers:

    @inproceedings{pasunuru2017reinforced,
        title={Reinforced Video Captioning with Entailment Rewards},
        author={Pasunuru, Ramakanth and Bansal, Mohit},
        booktitle={EMNLP},
        year={2017}
    }
    
    @inproceedings{pasunuru2017multi,
        title={Multi-Task Video Captioning with Video and Entailment Generation},
        author={Pasunuru, Ramakanth and Bansal, Mohit},
        booktitle={ACL},
        year={2017}
    }

