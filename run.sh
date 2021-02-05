#!/bin/bash

python train_model.py '1'
python make_config.py
python convert.py

git add .
git commit -m 'first subject'
git push 

python client.py 'runAcc.py'
python client.py 'runAcc-Copy1.py'
python client.py 'runAcc-Copy2.py'

python train_model.py '7'
python make_config.py
python convert.py

git add .
git commit -m 'second subject'
git push 

python client.py 'runAcc.py'
python client.py 'runAcc-Copy1.py'
python client.py 'runAcc-Copy2.py'

python train_model.py '12'
python make_config.py
python convert.py

git add .
git commit -m 'third subject'
git push 

python client.py 'runAcc.py'
python client.py 'runAcc-Copy1.py'
python client.py 'runAcc-Copy2.py'
python train_model.py '15'
python make_config.py
python convert.py

git add .
git commit -m 'forth subject'
git push 

python client.py 'runAcc.py'
python client.py 'runAcc-Copy1.py'
python client.py 'runAcc-Copy2.py'

python train_model.py '31'
python make_config.py
python convert.py

git add .
git commit -m 'fifth subject'
git push 

python client.py 'runAcc.py'
python client.py 'runAcc-Copy1.py'
python client.py 'runAcc-Copy2.py'