@echo off

set GPU=0
rem pnet
echo "Generate Data for pNet"
python3 jfda/prepare.py --net p --wdier --celeba --worker 8
echo "Train pNet"
python jfda/train.py --net p --gpu %GPU% --size 128 --lr 0.05 --lrw 0.1 --lrp 5 --wd 0.0001 --epoch 25
rem rnet
echo "Generate Data for rNet"
python jfda/prepare.py --net r --gpu %GPU% --detect --celeba --wider --worker 4
echo "Train rNet"
python jfda/train.py --net r --gpu %GPU% --size 128 --lr 0.05 --lrw 0.1 --lrp 5 --wd 0.0001 --epoch 25

rem onet
echo "Generate Data for oNet"
python jfda/prepare.py --net o --gpu %GPU% --detect --celeba --wider --worker 4
echo "Train oNet"
python jfda/train.py --net o --gpu %GPU% --size 64 --lr 0.05 --lrw 0.1 --lrp 7 --wd 0.0001 --epoch 35

rem lnet
echo "Generate Data for lNet"
python jfda/lnet.py --prepare --worker 8
echo "Train lNet"
python jfda/lnet.py --train --gpu %GPU% --lr 0.1 --lrw 0.1 --lrp 2 --epoch 10
pause