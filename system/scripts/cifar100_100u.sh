# FedAvg FedProx Ditto pFedMe APFL FedRep FedBABU FedAMP FedPHP FedFomo APPLE FedALA
ls="5"

# python -u main.py -lbs 64 -ls $ls -lr 0.01 -nc 100 -jr $3 -nb 100 -data Cifar100_u100_$1 -m cnn -gr 500 -did $2 -algo FedProx -mu 0.02
# python -u main.py -lbs 64 -ls $ls -lr 0.01 -nc 100 -jr $3 -nb 100 -data Cifar100_u100_$1 -m cnn -gr 500 -did $2 -algo Ditto -pls 5 -mu 0.5
# python -u main.py -lbs 64 -ls $ls -lr 0.01 -nc 100 -jr $3 -nb 100 -data Cifar100_u100_$1 -m cnn -gr 500 -did $2 -algo pFedMe -lam 15 -bt 1 -lrp 0.01 -K 5
# python -u main.py -lbs 64 -ls $ls -lr 0.01 -nc 100 -jr $3 -nb 100 -data Cifar100_u100_$1 -m cnn -gr 500 -did $2 -algo FedAvg
# python -u main.py -lbs 64 -ls $ls -lr 0.01 -nc 100 -jr $3 -nb 100 -data Cifar100_u100_$1 -m cnn -gr 500 -did $2 -algo APFL -al 1.0
python -u main.py -lbs 64 -ls $ls -lr 0.01 -nc 100 -jr $3 -nb 100 -data Cifar100_u100_$1 -m cnn -gr 500 -did $2 -algo FedAMP -lam 5e-7 -alk 5e-3 -sg 0.1
# python -u main.py -lbs 64 -ls $ls -lr 0.01 -nc 100 -jr $3 -nb 100 -data Cifar100_u100_$1 -m cnn -gr 500 -did $2 -algo FedPHP -lam 0.1 -mu 0.1
# python -u main.py -lbs 64 -ls $ls -lr 0.01 -nc 100 -jr $3 -nb 100 -data Cifar100_u100_$1 -m cnn -gr 500 -did $2 -algo FedFomo -M 5
# python -u main.py -lbs 64 -ls $ls -lr 0.01 -nc 100 -jr $3 -nb 100 -data Cifar100_u100_$1 -m cnn -gr 500 -did $2 -algo FedRep -pls $ls
python -u main.py -lbs 64 -ls $ls -lr 0.01 -nc 100 -jr $3 -nb 100 -data Cifar100_u100_$1 -m cnn -gr 500 -did $2 -algo FedBABU
# python -u main.py -lbs 64 -ls $ls -lr 0.01 -nc 100 -jr $3 -nb 100 -data Cifar100_u100_$1 -m cnn -gr 500 -did $2 -algo FedDRL
