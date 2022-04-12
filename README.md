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


Step 1: Adjust training parameters in [`config.yaml`](src/config.yaml)



Step 2: To train model in `Docker` please, run from the root of this repository: 

      docker build --network=host -t fre-gan:train .
      
      
Step 3: After build is complit, to run using `GPU`:

      docker run --gpus 1 -ti fre-gan:train
      
For `CPU`-only:

      docker run -ti fre-gan:train
      
      
Step 4: From the repository root run:

      python3 -m src.train
      
      
If you are not using Docker just skip steps 2 & 3 :)


### 3. Evaluation setup

Step 1 Download model wigths: [Dummy Weights](https://fre-gan.s3.eu-west-1.amazonaws.com/epoch%3D1-step%3D1060.ckpt) – 2nd training epoch, the words are hardly understandable and speech sounds robotic.

Step 2: from the repository root run:

      python3 -m src.inference -w <model_weights_path> -p <reference_wav_path>
      
      
The following script will generate output wav-file in `data/generated_samples` directory. You can add a custom path with `-o` flag and
obeserve other flags in `inference.py`(src/inference.py) script

To run evaluation in Docker complete Step3 from training block before running the script.


### 4. Baseline

To train HiFi-GAN vocoder on `Common Voice Version 3.0` dataset:

- follow the instructions to dataset loading in the first block;

- clone [forked HiFi-GAN repository](https://github.com/dariadiatlova/hifi-gan/tree/common-voice-training);

- run `train.py` from common-voice-branch.

Instructions for evaluation are the same as in original repository. [Dummy Weights](https://hifi-gan.s3.eu-west-1.amazonaws.com/g_00000500) for inference.


### 5. [Report](https://ionian-dogsled-238.notion.site/d28198092c6b4be3b7a6f866356e2ed7), [Wandb](https://wandb.ai/daryoou_sh/Mel-Fixed-Fre-GAN/reports/Fre-GAN--VmlldzoxODI3ODc2?accessToken=980x5j43uddkmorraspr3svkrss4ezz28fvbhein7ms4std13hupvt1w4m2ujfap) :nerd_face:
