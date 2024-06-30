# FedAvg FedProx Ditto pFedMe APFL FedRep FedBABU FedAMP FedPHP FedFomo APPLE FedALA
ls="5"

python -u main.py -lbs 64 -ls $ls -lr 0.01 -nc 20 -jr 0.2 -nb 50 -data omniglot -m cnn -gr 200 -did $1 -algo FedProx -mu 0.02
python -u main.py -lbs 64 -ls $ls -lr 0.01 -nc 20 -jr 0.2 -nb 50 -data omniglot -m cnn -gr 200 -did $1 -algo Ditto -pls 5 -mu 0.5
python -u main.py -lbs 64 -ls $ls -lr 0.01 -nc 20 -jr 0.2 -nb 50 -data omniglot -m cnn -gr 200 -did $1 -algo FedAvg
python -u main.py -lbs 64 -ls $ls -lr 0.01 -nc 20 -jr 0.2 -nb 50 -data omniglot -m cnn -gr 200 -did $1 -algo pFedMe -lam 15 -bt 1 -lrp 0.01 -K 5
python -u main.py -lbs 64 -ls $ls -lr 0.01 -nc 20 -jr 0.2 -nb 50 -data omniglot -m cnn -gr 200 -did $1 -algo APFL -al 1.0
python -u main.py -lbs 64 -ls $ls -lr 0.01 -nc 20 -jr 0.2 -nb 50 -data omniglot -m cnn -gr 200 -did $1 -algo FedAMP -lam 5e-7 -alk 5e-3 -sg 0.1
python -u main.py -lbs 64 -ls $ls -lr 0.01 -nc 20 -jr 0.2 -nb 50 -data omniglot -m cnn -gr 200 -did $1 -algo FedPHP -lam 0.1 -mu 0.1
python -u main.py -lbs 64 -ls $ls -lr 0.01 -nc 20 -jr 0.2 -nb 50 -data omniglot -m cnn -gr 200 -did $1 -algo FedFomo -M 5
python -u main.py -lbs 64 -ls $ls -lr 0.01 -nc 20 -jr 0.2 -nb 50 -data omniglot -m cnn -gr 200 -did $1 -algo FedRep -pls $ls
python -u main.py -lbs 64 -ls $ls -lr 0.01 -nc 20 -jr 0.2 -nb 50 -data omniglot -m cnn -gr 200 -did $1 -algo FedBABU
python -u main.py -lbs 64 -ls $ls -lr 0.01 -nc 20 -jr 0.2 -nb 50 -data omniglot -m cnn -gr 200 -did $1 -algo FedDRL