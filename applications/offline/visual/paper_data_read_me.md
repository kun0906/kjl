

1. build models on NEON
    kjl_neon_idv.sh
    (main_kjl_idv.py)

2. only save the model params to disk on NEON
    main_model_params_files.py
    speedup/data/models/

3. copy "speedup/data/models" to "kjl_reload" project
    zip -rq models.zip models
    cp -rp speedup/data/models.zip ~/kjl_reload/examples/speedup/data/
    scp -rp ky2440@neon.cs.uchicago.edu:/home/ky2440/kjl/examples/speedup/data/models.zip ~/PycharmProjects/kjl_reload/examples/speedup/data/

    cd ~/kjl_reload/examples/speedup/data/
    rm -rf models
    unzip -q models.zip

4. copy "kjl_reload" to different devices and run "main_reload.sh"
    cd ~
    zip -rq kjl_reload

    # download to local
    scp -rp ky2440@neon.cs.uchicago.edu:/home/ky2440/kjl_reload.zip ~/PycharmProjects/
    # copy zip from local to remote
    scp -rp  ~/PycharmProjects/kjl_reload.zip ky2440@tigerteam.io:/home/ky2440/
    ssh ky2440@tigerteam.io
    scp -rp kjl_reload.zip iotlab.cs.uchicago.edu:/home/ky2440/
    ssh iotlab.cs.uchicago.edu
    scp -rp kjl_reload.zip pi@192.168.143.163:/home/pi/ky2440/
    ssh pi@192.168.143.163
    tmux attach -t kjl
    cd ky2440
    rm -rf kjl_reload
    unzip -q kjl_reload.zip
    cd kjl_reload/examples/
    ./speedup/main_kjl.sh

    scp -rp kjl_reload.zip nano@192.168.143.162:/home/nano/ky2440/
    ssh nano@192.168.143.162
    tmux attach -t kjl
    cd ky2440
    rm -rf kjl_reload

5. copy out from remote
    zip -rq speedup/nano_out.zip speedup/out/
    scp -rp nano@192.168.143.162:/home/nano/ky2440/kjl_reload/examples/speedup/nano_out.zip  /home/ky2440/
    exit
    scp -rq  iotlab.cs.uchicago.edu:/home/ky2440/nano_out.zip  /home/ky2440/
    exit
    scp -rq ky2440@tigerteam.io:/home/ky2440/nano_out.zip  ~/PycharmProjects/kjl/examples/speedup/paper_data/
    scp -rq ky2440@tigerteam.io:/home/ky2440/pi_out.zip  ~/PycharmProjects/kjl/examples/speedup/paper_data/
    scp -rq ky2440@neon.cs.uchicago.edu:/home/ky2440/kjl_reload/examples/speedup/neon_out.zip ~/PycharmProjects/kjl/examples/speedup/paper_data/
    scp -rq ky2440@neon.cs.uchicago.edu:/home/ky2440/kjl/examples/speedup/neon_train_out.zip ~/PycharmProjects/kjl/examples/speedup/paper_data/



