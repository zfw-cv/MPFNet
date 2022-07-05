# Self-supervised multi-scale pyramid fusion networks for realistic bokeh effect rendering

**| [Paper](https://www.sciencedirect.com/science/article/abs/pii/S1047320322001110) |**

If you are interested in my work, you can contact me by **<u>email</u>(zhifengwang686@gmail.com)**, I will check the email regularly. thanks!

#### Example:

![example](../../source/images/readme/example-1656995214622.jpg)

#### 1、Dataset

You could get the EBB! dataset by registering [here](https://competitions.codalab.org/competitions/24716).and put it in the train folder.

Train split: data/train.csv

Test split (val294 set): data/test.csv

#### 2、Installation

```
git clone https://github.com/zfw-cv/MPFNet.git
cd MPFNet
pip install -r requirements.txt
```

#### 3、Train

You can download the pre-training model by **[here](https://drive.google.com/drive/folders/1-f_HBaC6nqjVemcyWOtPdv_EMK0AAGdr?usp=sharing)** .And put it in the checkpoints folder.

```python
python train.py
```

#### 4、Test

```python
python test.py
```

#### 5、Val

To test our effects more easily, you can directly use the results obtained from our pre-trained weights file.

```
python val.py
```

#### 6、Citation

If you find our work useful in your research, please cite our paper.

```
@article{wang2022self,
  title={Self-supervised multi-scale pyramid fusion networks for realistic bokeh effect rendering},
  author={Wang, Zhifeng and Jiang, Aiwen and Zhang, Chunjie and Li, Hanxi and Liu, Bo},
  journal={Journal of Visual Communication and Image Representation},
  pages={103580},
  year={2022},
  publisher={Elsevier}
}
```