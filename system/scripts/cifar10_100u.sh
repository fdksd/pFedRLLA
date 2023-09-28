# FedAvg FedProx Ditto pFedMe APFL FedRep FedBABU FedAMP FedPHP FedFomo APPLE FedALA
ls="5"

# python -u main.py -lbs 64 -ls $ls -lr 0.01 -nc 100 -jr $3 -nb 10 -data Cifar10_u100_$1 -m cnn -gr 500 -did $2 -algo FedProx -mu 0.02
# python -u main.py -lbs 64 -ls $ls -lr 0.01 -nc 100 -jr $3 -nb 10 -data Cifar10_u100_$1 -m cnn -gr 500 -did $2 -algo Ditto -pls 5 -mu 0.5
# python -u main.py -lbs 64 -ls $ls -lr 0.01 -nc 100 -jr $3 -nb 10 -data Cifar10_u100_$1 -m cnn -gr 500 -did $2 -algo FedAvg
# python -u main.py -lbs 64 -ls $ls -lr 0.01 -nc 100 -jr $3 -nb 10 -data Cifar10_u100_$1 -m cnn -gr 500 -did $2 -algo pFedMe -lam 15 -bt 1 -lrp 0.01 -K 5
# python -u main.py -lbs 64 -ls $ls -lr 0.01 -nc 100 -jr $3 -nb 10 -data Cifar10_u100_$1 -m cnn -gr 500 -did $2 -algo APFL -al 1.0
python -u main.py -lbs 64 -ls $ls -lr 0.01 -nc 100 -jr $3 -nb 10 -data Cifar10_u100_$1 -m cnn -gr 500 -did $2 -algo FedAMP -lam 5e-7 -alk 5e-3 -sg 0.1
# python -u main.py -lbs 64 -ls $ls -lr 0.01 -nc 100 -jr $3 -nb 10 -data Cifar10_u100_$1 -m cnn -gr 500 -did $2 -algo FedPHP -lam 0.1 -mu 0.1
# python -u main.py -lbs 64 -ls $ls -lr 0.01 -nc 100 -jr $3 -nb 10 -data Cifar10_u100_$1 -m cnn -gr 500 -did $2 -algo FedFomo -M 5
# python -u main.py -lbs 64 -ls $ls -lr 0.01 -nc 100 -jr $3 -nb 10 -data Cifar10_u100_$1 -m cnn -gr 500 -did $2 -algo FedRep -pls $ls
python -u main.py -lbs 64 -ls $ls -lr 0.01 -nc 100 -jr $3 -nb 10 -data Cifar10_u100_$1 -m cnn -gr 500 -did $2 -algo FedBABU
# python -u main.py -lbs 64 -ls $ls -lr 0.01 -nc 100 -jr $3 -nb 10 -data Cifar10_u100_$1 -m cnn -gr 500 -did $2 -algo FedDRL

# python -u main.py -lbs 64 -ls $ls -lr 0.1 -nc 100 -jr 0.1 -nb 10 -data Cifar10_u100_s5 -m cnn -gr 100 -did 1 -algo FedAvg
# python -u main.py -lbs 64 -ls $ls -lr 0.1 -nc 100 -jr 0.1 -nb 10 -data Cifar10_u100_s5 -m cnn -gr 100 -did 1 -algo FedDRL
# python -u main.py -lbs 64 -ls $ls -lr 0.1 -nc 100 -jr 0.1 -nb 10 -data Cifar10_u100_s5 -m cnn -gr 100 -did 1 -algo FedProx -mu 0.02
# python -u main.py -lbs 64 -ls $ls -lr 0.1 -nc 100 -jr 0.1 -nb 10 -data Cifar10_u100_s5 -m cnn -gr 100 -did 1 -algo Ditto -pls 5 -mu 0.5
# python -u main.py -lbs 64 -ls $ls -lr 0.1 -nc 100 -jr 0.1 -nb 10 -data Cifar10_u100_s5 -m cnn -gr 100 -did 1 -algo pFedMe -lam 15 -bt 1 -lrp 0.01 -K 5
# python -u main.py -lbs 64 -ls $ls -lr 0.1 -nc 100 -jr 0.1 -nb 10 -data Cifar10_u100_s5 -m cnn -gr 100 -did 1 -algo APFL -al 1.0 -gr 100 -did 1
# python -u main.py -lbs 64 -ls $ls -lr 0.1 -nc 100 -jr 0.1 -nb 10 -data Cifar10_u100_s5 -m cnn -gr 100 -did 1 -algo FedAMP -lam 5e-7 -alk 5e-3 -sg 0.1
# python -u main.py -lbs 64 -ls $ls -lr 0.1 -nc 100 -jr 0.1 -nb 10 -data Cifar10_u100_s5 -m cnn -gr 100 -did 1 -algo FedPHP -lam 0.1 -mu 0.1
# python -u main.py -lbs 64 -ls $ls -lr 0.1 -nc 100 -jr 0.1 -nb 10 -data Cifar10_u100_s5 -m cnn -gr 100 -did 1 -algo FedFomo -M 5
# python -u main.py -lbs 64 -ls $ls -lr 0.1 -nc 100 -jr 0.1 -nb 10 -data Cifar10_u100_s5 -m cnn -gr 100 -did 1 -algo FedRep
# python -u main.py -lbs 64 -ls $ls -lr 0.1 -nc 100 -jr 0.1 -nb 10 -data Cifar10_u100_s5 -m cnn -gr 100 -did 1 -algo FedBABU


