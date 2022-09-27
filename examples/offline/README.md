Recreate new models from saved paramters stored on the disk and evalute them on test sets.

#################################################################################################################
1. Instructions to run experiments and obtain results
    # python3.7.9
    PYTHONPATH=. PYTHONUNBUFFERED=1 python3.7 examples/offline.py
    PYTHONPATH=. PYTHONUNBUFFERED=1 python3.7 examples/offline-parallel.py
    or 
    main_reload.sh: a shell script to run all experiments.

    $cd kjl/examples/
    $chmod 755 ./offline/main_reload.sh
    $./offline/main_reload.sh

    Note: the main_reload.sh uses python3.7.
          If Python version is not 3.7, please modify the main_reload.sh.

#################################################################################################################
2. Project structures
|-kjl: (root)
  |--examples
     |--offline
         |--data: (all models and datasets)
         |--main_reload.sh (bash shell sript)
         |--main_reload_idv.py
         |--out: (results)

  |--kjl: (core codes)
     |--model: (includes all models)
  |--utils: (tools)


#################################################################################################################
3. Requried libraries
    Python==3.7.3

    # Can be installed by pip3
    pandas==0.25.1
    numpy==1.19.2
    scikit-learn==0.22.1
    func_timeout==4.3.5

    joblib==1.0.1
    Cython==0.29.14    # for quickshift++
    matplotlib==3.1.1
    memory-profiler==0.57.0 #
    openpyxl==3.0.2
    scapy==2.4.4        # parse pcap
    scipy==1.4.1
    seaborn==0.11.1
    xlrd==1.2.0
    XlsxWriter==1.2.8


    # install instructions for the following libraries
    1) odet==0.0.1         # extract features
        i) cd "kjl/kjl/dataset/odet-master"
        ii) pip3.7 install .

    2) QuickshiftPP==1.0        # seek modes
        i) cd "kjl/kjl/model/quickshift/"
        iii) python3.7 setup.py build
        iv) python3.7 setup.py install
