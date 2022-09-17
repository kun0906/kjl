# Compress and download codes and results from 'Neon'
# Python 3.7.3 with ’–enable-optimizations’ option for RSPI (32bit: Python3.7.3) and NANO (64bit: Python3.7.3)
# Numpy 1.18.2 (install from the numpy source codes that searches for BLAS and LAPACK dynamic link libraries at
# build time as influenced by the system environment variable

```sh
# ~/PycharmProjects
# compress 'kjl', but exclude 'offline/out' and 'legacy'
zip -r kjl_deploy.zip kjl -x kjl/examples/offline/out/\* kjl/legacy/\* kjl/examples/online/\* kjl/examples/offline/report/\* kjl/examples/offline/deployment/out/\* kjl/examples/offline/legacy/\* kjl/\.git/\*
# download kjl_deploy.zip from the remote server (Neon) to local 
#scp -p ky2440@neon.cs.uchicago.edu:~/kjl_deploy.zip ~/Downloads/
```

# Upload data and results to different devices

## Raspberry PI
```shell
# upload kjl_deploy to the login machine (tigerteam.io)
#scp -p ~/Downloads/kjl_deploy.zip  ky2440@tigerteam.io:/home/ky2440/
scp -p ~/PycharmProjects/kjl_deploy.zip  ky2440@tigerteam.io:/home/ky2440/
# upload the zip to another login machine (iotlab) from tigerteam.io
ssh ky2440@tigerteam.io
scp -p kjl_deploy.zip iotlab.cs.uchicago.edu:/home/ky2440/
# upload the zip to RSPI from iotlab. RSPI: dc:a6:32:ed:6c:63
ssh iotlab.cs.uchicago.edu 
scp -p kjl_deploy.zip pi@192.168.143.242:/home/pi/ky2440/
#ssh pi@192.168.143.242 
#scp -p kjl_deploy.zip pi@192.168.143.163:/home/pi/ky2440/ # (old ip address)
# decompress the zip and execute the script
ssh pi@192.168.143.242
#tmux new -s kjl 
tmux attach -t kjl 
cd ky2440
unzip -q kjl_deploy.zip
vim kjl/examples/offline/deployment/deploy_evaluate_model.py 
cd kjl
PYTHONPATH=. PYTHONUNBUFFERED=1 python3.7 examples/offline/deployment/deploy_evaluate_model.py > deploy.txt 2>&1 &

# compress and download the resutls
cd ..
#zip -r kjl_deploy.zip kjl -x kjl/examples/offline/out/\* kjl/legacy/\*
zip -r pi_out.zip kjl/examples/offline/deployment/out -x kjl/legacy/\*
exit
scp -p pi@192.168.143.242:/home/pi/ky2440/pi_out.zip pi_out.zip
exit
scp -p iotlab.cs.uchicago.edu:/home/ky2440/pi_out.zip pi_out.zip
exit
scp -p ky2440@tigerteam.io:/home/ky2440/pi_out.zip ~/PycharmProjects/kjl/examples/offline/report/out/src_dst/results/
```

## NANO
```shell
# upload kjl_deploy to the login machine (tigerteam.io)
scp -p ~/Downloads/kjl_deploy.zip  ky2440@tigerteam.io:/home/ky2440/
# upload the zip to another login machine (iotlab) from tigerteam.io
ssh ky2440@tigerteam.io
scp -p kjl_deploy.zip iotlab.cs.uchicago.edu:/home/ky2440/
# upload the zip to Nano from iotlab. Nano: c4:41:1e:5b:18:a5
ssh iotlab.cs.uchicago.edu 
#scp -p kjl_deploy.zip nano@192.168.143.251:/home/nano/ky2440/ # (old ip address)
scp -p kjl_deploy.zip nano@192.168.143.162:/home/nano/ky2440/
# decompress the zip and execute the script
ssh nano@192.168.143.162
#tmux new -s kjl 
tmux attach -t kjl 
cd ky2440
unzip -q kjl_deploy.zip
vim kjl/examples/offline/deployment/deploy_evaluate_model.py 
cd kjl
PYTHONPATH=. PYTHONUNBUFFERED=1 python3.7 examples/offline/deployment/deploy_evaluate_model.py > deploy.txt 2>&1 &

# compress and download the resutls
cd ..
zip -r nano_out.zip kjl/examples/offline/deployment/out -x kjl/legacy/\*
#zip -r nano_out.zip kjl/examples/offline/deployment/out/src_dst -x kjl/legacy/\*
exit
scp -p nano@192.168.143.162:/home/nano/ky2440/nano_out.zip nano_out.zip
exit
scp -p iotlab.cs.uchicago.edu:/home/ky2440/nano_out.zip nano_out.zip
exit
scp -p ky2440@tigerteam.io:/home/ky2440/nano_out.zip ~/PycharmProjects/kjl/examples/offline/report/out/src_dst/results/
scp -p ky2440@tigerteam.io:/home/ky2440/pi_out.zip ~/PycharmProjects/kjl/examples/offline/report/out/src_dst/results/
```

# useful commands
```shell
arp -n 
nmap 
```

# get all configuration and work for python3.7.3
import numpy.distutils
np_config_vars = numpy.distutils.unixccompiler.sysconfig.get_config_vars()
import pprint
pprint.pprint(np_config_vars)
