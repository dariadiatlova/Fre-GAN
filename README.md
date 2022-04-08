# Test-task for VK-research internship 2022 :dove:
## Implementation of [Fre-GAN: Adversarial Frequency-consistent Audio Synthesis](https://arxiv.org/pdf/2106.02297.pdf)

### 1. Dataset

In this project we use [Mozilla Common Voice Corpus 3.0](https://commonvoice.mozilla.org/ru/datasets) for russian language.

The dataset consists of 31 hours of recorded speech in `.mp3` format with a sample rate `48000`. To download the dataset, please, open [Common Voice](https://commonvoice.mozilla.org/ru/datasets) webpage with Russian Specch datasets. Select `Common Voice Version 3.0`, make sure the language is `Russian`. After that enter your email, click right mouse bottun
and `copy url adress`. Then run [`download.sh`](download.sh) script from the root of cloned repository with an argumet (`copied url adress`). The following script will create a directory in the root of repository with `.wav` files imported and converted with the same sample rate from the original dataset. 

The are 2 reasons for conducting audio-format conversion: 
- [`hifi-gan`](https://github.com/jik876/hifi-gan/blob/master/) implementation is used as a beseline model for the experiment, and the authors use `.wav` format of audio in theirs implementation;
- at least with librosa loadig `.wav` files is a bit faster than loading `mp3` files.

After running [`download.sh`](download.sh) script folders should be placed as following:


      Fre-GAN
          |- data
              |- audio
              | __init__.py
              | train.tsv
              | test.tsv

### 2. Training setup

#### Step 1: Adjust training parameters in [`config.yaml`](src/config.yaml)

#### Step 2: To train model in `Docker` please, run from the root of this repository: 

      docker build --network=host -t fre-gan:train .
      
#### Step 3: After build is complit, to run using `GPU`:

      docker run --gpus 1 -ti fre-gan:train
      
For `CPU`-only:

      docker run -ti fre-gan:train
      
#### Step 4: From the repository root run:

      python3 -m src.train
      
      
If you are not using Docker just skip steps 2 & 3 :)
