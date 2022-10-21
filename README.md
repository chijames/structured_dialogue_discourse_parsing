# Structured Dialogue Discourse Parsing
Pytorch implementation of the paper [Structured Dialogue Discourse Parsing](https://aclanthology.org/2022.sigdial-1.32/).

## Installation
Run the following commands to recreate the environment we used for the development of this project:
```sh
conda create -n sddp python=3.9.6 -y
conda activate sddp
conda install pytorch==1.12.1 cudatoolkit=11.3 -c pytorch -y
pip install -r requirements.txt
pip install git+https://github.com/timvieira/arsenal.git@b7a5f93b04533236bbe297b4079256ea15812b88
pip install -e git+https://github.com/rycolab/spanningtrees.git@68c68e6f37d0951a524b901d9e84c5c1c498420e#egg=spanningtrees
```

## Data Preparation
Please pass --link_only to all the shell scripts if you want to train and test using the link_only option. The default is link and relation.
```sh
cd data/
bash parse.sh (--link_only)
cd ../
```

## Train/save on one domain
```sh
bash train_stac.sh (--link_only)
bash train_molweni.sh (--link_only)
```

## Test on one domain
```sh
bash test_stac.sh (--link_only)
bash test_molweni.sh (--link_only)
```

## Train on one domain and test/save on another domain
```sh
bash train_stac_on_molweni.sh (--link_only)
bash train_molweni_on_stac.sh (--link_only)
```

## Test on another domain
```sh
bash test_stac_on_molweni.sh (--link_only)
bash test_molweni_on_stac.sh (--link_only)
```
