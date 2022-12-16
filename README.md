# Graph Representation Learning
## Candidate 1068137

### Installation

Run the following command to create a virtual environment and download the requisite packages.

```zsh
python3 -m venv venv
source ./venv/bin/activate
pip3 install -Ur requirements.txt
```


### Experiments

Run the following commands to replicate the results. 

```zsh
python3 main.py -d Cora --epochs 100 --runs 5 --max_K 16
python3 main.py -d CiteSeer --epochs 100 --runs 5 --max_K 16
python3 main.py -d Cora --epochs 100 --runs 5 --max_K 16 --v2 True
python3 main.py -d Cora --epochs 100 --runs 5 --max_K 16 --v2 True
```

Run the following command for the complete list of arguments that can be supplied. 

```
python3 main.py --help
```