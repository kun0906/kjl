## KJL
A python library is created for efficient anomaly detection, which mainly includes two submodules: kernel projection ('projection') and novelty detection models ('models'). 

##Architecture
- docs/: 
    includes all documents (such as APIs)
- applications/: 
    includes applications and toy examples and datasets for you to play with it 
- kjl/: 
    source codes: includes two main sublibraries (projects and models)
    - projects/: 
        includes KJL and Nystrom (such as OCSVM)
    - models/: 
        includes OCSVM and GMM
    - utils/:
        includes common functions (such as load and dump data)
    - visul/: 
        includes visualization functions
- thrid_party/: 
    others (such as xxx.sh, make) 
- tools/: 
    includes useful tools (e.g., upload files)

- LICENSE.txt
- README.md
- requirements.txt
- setup.py
- version.txt

## Install
    pip3 install . 
    (pip3 will call setup.py to install the library automatically)

## Usage

    """ 1.1 Parse data and extract features

	"""
	lg.info(f'\n--- 1.1 Load data')
	feat_file = f'{OUT_DIR}/DEMO_IAT+SIZE.dat'
	X, y = load(feat_file)

	""" 1.2 Split train and test set

	"""
	lg.info(f'\n--- 1.2 Split train and test set')
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
	                                                    shuffle=True, random_state=RANDOM_STATE)
	lg.debug(f'X_train:{X_train.shape}, y_train: {Counter(y_train)}')
	lg.debug(f'X_test:{X_test.shape}, y_test: {Counter(y_test)}')

	""" 1.3 preprocessing
		projection
	"""
	lg.info(f'\n--- 1.3 Preprocessing')
	proj = Projection(name='KJL')
	proj.fit(X_train)
	X_train = proj.transform(X_train)

	""" 2.1 Build the model

	"""
	lg.info(f'\n--- 2.1 Build the model')
	model_name = 'OCSVM'
	model = Model(name=model_name, q=proj.q, overwrite=OVERWRITE, random_state=RANDOM_STATE)
	model.fit(X_train, y_train)

	""" 2.2 Evaluate the model

	"""
	lg.info(f'\n--- 2.2 Evaluate the model')
	X_test = proj.transform(X_test)
	res = model.eval(X_test, y_test)

	""" 3. Dump the result to disk

	"""
	lg.info(f'\n--- 3. Save the result')
	res_file = os.path.join(OUT_DIR, f'DEMO-KJL-{model_name}-results.dat')
	check_path(os.path.dirname(res_file))
	dump(res, out_file=res_file)
	lg.info(f'res_file: {res_file}')

	return res

For more examples, please check the 'examples' directory

## TODO

- Complete the online application
- Further evaluate and optimize the library continually.
- Add 'test' cases
- Add LICENSE.txt
- Generated docs from docs-string automatically

Welcome to make any comments to make it more robust and easier to use!

## Contact
- Email: kun.bj@outlook.com