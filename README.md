# Promoting Accurate Image Reconstruction via Synthetic Noise for Unsupervised Screw Anomaly Detection and Location (ScrewNet) <h1>

## Enviroment
* Ubuntu 18.04.6 LTS
* Python3.7.7
```bash
pip install --upgrade -r requirements.txt
````
## Architecture
```bib
├── README.md
├── checkpoints/
│   ├── Z_M6x20-200/
│   │   └── latest_lambda10.ckpt
│   ├── Z_M6x34.2-1000_F2000/
│   │   └── latest_lambda10.ckpt
│   └── Z_M8x32.4/
│       └── latest_lambda10.ckpt
├── models/
│   ├── AE_CycleGAN_Z_M6x20-200.h5
│   ├── AE_CycleGAN_Z_M6x34.2-1000_F2000.h5
│   └── AE_CycleGAN_Z_M8x32.4.h5
├── arch/
├── Z_M6x34.2-1000_F2000/
│   ├── train_A (real undamaged screws)
│   ├── val_A (real undamaged screws)
│   ├── test_A (real damaged screws)
│   └── test_B (real undamaged screws)
├── main.py
├── dataset.py
├── ImageProcessing.py
├── Metrics.py
├── model.py
├── run.py
└── util.py

````


##  Dataset
You can download the screw datasets used in the paper from the following links. Make sure to place the dataset in the same directory as main.py.
  * M6x20: https://drive.google.com/file/d/19x0n98L8RVDFGE2HaYT18SDifkfTcnNY/view?usp=sharing
  * M6x34.2:https://drive.google.com/file/d/116wIEJp_gq67lPfN_EIZ1YjfBISYOnVI/view?usp=sharing
  * M8x32.4:https://drive.google.com/file/d/1vlLNF4V4dvnBTq5u3ihxtwyJziKcqggN/view?usp=sharing


## Pre-train models
You can download the pre-trained noise generation module and anomaly detection networks from Google Drive. Make sure to place the pre-trained models in the correct directories:
  * Pre-trained Noise Generation Module:[https://drive.google.com/file/d/1G5d12SFKlz5-G3xiibXZrABajjHU6rT4/view?usp=sharing](https://drive.google.com/file/d/1YsHSFclkMg0ArDihn0kWwkHvchOyuEbP/view?usp=sharing)
 * Pre-trained Anomaly Detection Networks:https://drive.google.com/file/d/1z7O5oMgW7e6EjptKVJEHjJrQruMraLY-/view?usp=sharing


## Training
If you want to re-train the anomaly detection network, you can use the following commands:
```bib
python main.py --training true --gpu_id 0 --Noise_Type CycleGAN --Screw_Type Z_M6x20-200 --epochs 80 --lr 0.05
python main.py --training true --gpu_id 0 --Noise_Type CycleGAN --Screw_Type Z_M6x34.2-1000_F2000 --epochs 80 --lr 0.05
python main.py --training true --gpu_id 0 --Noise_Type CycleGAN --Screw_Type Z_M8x32.4 --epochs 80 --lr 0.05
````

## Testing
To test the model, use the following commands:
```bib
python main.py --training false --gpu_id 0  --Noise_Type CycleGAN --Screw_Type Z_M6x20-200  --test_folder testA
python main.py --training false --gpu_id 0  --Noise_Type CycleGAN --Screw_Type Z_M6x34.2-1000_F2000 --test_folder testA
python main.py --training false --gpu_id 0  --Noise_Type CycleGAN --Screw_Type Z_M8x32.4 --test_folder testA 
````
## Acknowledgments
Our project is developed based on the CycleGAN architecture. We extend our sincere thanks to the creators of CycleGAN(https://github.com/jzsherlock4869/cyclegan-pytorch) for their excellent work. Additionally, we would like to express our gratitude to the anonymous reviewers for their insightful feedback, which has been instrumental in improving this work.
