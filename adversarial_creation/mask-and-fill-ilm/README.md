# mask-and-fill approach
This code is based on the repo [ilm] (https://github.com/chrisdonahue/ilm) for the paper Infilling by Language Modeling (ILM).

Requirements:
tranformers==3.5.0
pytorch-ignite==0.3.0

## Installation

From ILM repo - We recommend installing this package using `virtualenv`. After activating the virtual environment, run the following commands:

1. `git clone git@github.com:chrisdonahue/ilm.git`
1. `cd ilm`
1. `pip install -r requirements.txt`
1. `python -c "import nltk; nltk.download('punkt')"`
1. `pip install -e .`
1. `pip install lm-scorer`

## Data
The raw input data for ilm model for dialogue is a text file with each line containing dialogue utterances separated by tabs. A sample file test_combined.txt is uploaded in data/dailydialog folder.

## Commands 
To convert dialogue data to masked versions which are then saved as pickle files, run
``` python create_ilm_examples.py test data/char_masks/dailydialog --seed 0 --data_name dailydialog --data_split $SPLIT --mask_cls ilm.mask.hierarchical_dailydialog.MaskHierarchical --data_dir data/dailydialog 
```
Here $SPLIT would be train, valid and test

To preview masked spans created in the file above which will be used for training the ilm model, run
``` python preview_ilm_examples_dailydialog.py data/char_masks/dailydialog/test.pkl
```

To train the model
```
python dd_train_ilm.py experiment_dd train-lm data/char_masks/dailydialog/ --seed 0 --train_examples_tag train --eval_examples_tag valid --eval_max_num_examples 512 --mask_cls ilm.mask.hierarchical_dailydialog.MaskHierarchica
```

To generate adversarial data
```
python create_adv_ilm.py --experiment_name test1
```
