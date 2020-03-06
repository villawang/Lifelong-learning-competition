# Lifelong-learning-competition
This is our solution submitted to the IROS2019 competition [Lifelong Object Recognition Challege](https://lifelong-robotic-vision.github.io/competition/Object-Recognition.html)
```
@article{she2019openloris,
  title={OpenLORIS-Object: A Dataset and Benchmark towards Lifelong Object Recognition},
  author={She, Qi and Feng, Fan and Hao, Xinyue and Yang, Qihan and Lan, Chuanlin and Lomonaco, Vincenzo and Shi, Xuesong and Wang, Zhengwei and Guo, Yao and Zhang, Yimin and others},
  journal={arXiv preprint arXiv:1911.06487},
  year={2019}
}
```
## Requirements
Python3 \
torch == 1.2 \
torchvision == 0.4.0 \
scipy \
numpy \
sklearn \
tqdm \
torchsummary \
pillow==6.2.1 \
matplotlib \
pandas
## Usage
Pretrained backbone MobilenetV2 is in model/ directory. Pretrained model is in the output/ directory. Use the train.py to train the lifelong leanring model from the scratch.
