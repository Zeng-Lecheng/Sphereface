# An implementation of SphereFace in PyTorch

This is an implementation of [SphereFace](https://github.com/wy1iu/sphereface) with the most recent (2023) version of PyTorch. It implements the 4-layer CNN mentioned in "Table 2" in [the paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Liu_SphereFace_Deep_Hypersphere_CVPR_2017_paper.pdf). It refers a lot to [this repo](https://github.com/clcarwin/sphereface_pytorch) while it's documented a bit and deprecated code removed.

## Setup

1. Follow https://docs.python.org/3/library/venv.html#creating-virtual-environments and set a virtual environment up, or it's up to you to use any env.
2. Follow https://docs.python.org/3/library/venv.html#how-venvs-work to activate the virtual environment
3. Install dependencies: `pip3 install -r requirements.txt`

## Data

This repo uses [LFW dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz) and assumes this structure of data path:

```
root
|---data
    |---lfw
        |---Aaron_Eckhart
            |---Aaron_Eckhart_0001.jpg
        |---Aaron_Guiel
            |---Aaron_Guiel_0001.jpg
        ... (names)
    |---pairsDevTest.txt
    |---pairsDevTrain.txt
|---saved_models
|---main.py
|---train.py
|---model.py
|---utils.py
|---(other files)
```

Links to the two txts:
- http://vis-www.cs.umass.edu/lfw/pairsDevTrain.txt
- http://vis-www.cs.umass.edu/lfw/pairsDevTest.txt

Detailed information about LFW and the two txts are on [their website](https://vis-www.cs.umass.edu/lfw/index.html).

## Train

`main.py` trains a CNN-based model and evaluate it. Available command line arguments are listed:

```bash
python3 main.py [-h] [--epochs number of epochs] [--lr learning rate] [--bs batch size] [--d device]
```

The PyTorch build in `requirements.txt` is CPU-only, if you have CUDA version installed, use `--d cuda` to enable it. Otherwise, omit it or use `--d cpu`.

It then prints out loss at each epoch and accuracy in determine whether two images are of the same person.
