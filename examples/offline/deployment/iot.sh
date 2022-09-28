#!/usr/bin/env sh


## Raspberry PI
# upload kjl_deploy to the login machine (tigerteam.io)
#scp -p ~/Downloads/kjl_deploy.zip  ky2440@tigerteam.io:/home/ky2440/
scp -p ~/PycharmProjects/kjl_deploy.zip  ky2440@tigerteam.io:/home/ky2440/
# upload the zip to another login machine (iotlab) from tigerteam.io
ssh ky2440@tigerteam.io
scp -p kjl_deploy.zip iotlab.cs.uchicago.edu:/home/ky2440/
# upload the zip to RSPI from iotlab. RSPI: dc:a6:32:ed:6c:63
ssh iotlab.cs.uchicago.edu
scp -p kjl_deploy.zip pi@192.168.143.242:/home/pi/ky2440/
ssh pi@192.168.143.242
tmux new -s kjl
#scp -p kjl_deploy.zip pi@192.168.143.163:/home/pi/ky2440/ # (old ip address)
# decompress the zip and execute the script
cd ky2440
unzip -q kjl_deploy.zip
cd kjl
PYTHONPATH=. PYTHONUNBUFFERED=1 python3.7 examples/offline/deployment/deploy_evaluate_model.py > deploy.txt 2>&1 &




