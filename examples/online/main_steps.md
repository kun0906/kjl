20201111-algorithm

=======================
Configuration:
	Fridge: idle, open_shut, and browse
	Ratio = 100:0 (one example)
		i) init_set: open_shut(500)
		ii) arrival_set: browse(500)
		iii) test_set: normal (open_shut(100) + browse(100)) + abnormal (idle(200))

Main steps (Best params): 
	1. Build models with different params (i.e., n_components and q_kjl) and get the best model on test_set.
		(n_components=[1, 5, 10, 15, 20, 25, 30, 35, 40, 45], q_kjl=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
		i) With std:
			Get std on init_set
			Standardize init_set and test_set to obtain std_init_set and std_test_set 
		ii) With kjl:
			Project init_set (i.e, std_init_set) and test_set (i.e., std_test_set)
		iii) Build the model on the projected init_set 
		iv) Evaluate the model on the projected test_set (prj_test)
			Choose the best one (Model) 


	Split arrive_set into 10 batches and each batch (batch_data_i, i is from 1 to 10) has 50 datapoints
	2. Retrain/update the best model  
		a) For Batch GMM: 
			i) With std: 
				Use the old std (obtained from the init_set) to standardize batch_data_i
			ii) Add standardized batch_data_i into std_init_set
				std_init_set = [std_init_set, batch_data_i]
			iii) With kjl: 
				1) Recompute sigma (the new sigma) on the std_init_set
				2) Reselect n (n=100) datapoints from std_init_set
				3) Recompute KJL with the new sigma
				4) Project std_init_set to obtain the projected data (prj_train).
			iv) Retrain the model on prj_train
			
		b) For Online GMM:
			i) With std: 
				Use the old std (obtained from the init_set) to standardize batch_data_i
			ii) Add standardized batch_data_i into std_init_set
				std_init_set = [std_init_set, batch_data_i]
			ii) With kjl:
				1) Random select m rows (m_rows) from batch_data_i	(m= batch_size_i / std_init_set_size * n, here n=100)
				2) Replace the m random rows in Xrow with the m_rows 
				3) Recompute KJL with the old sigma (obtained from the init_set)
				4) Project std_init_set to obtain the projected_data (prj_train)
			iii) Update the model with prj_train
			
	3. Evaluate the updated model on test_set.
		I) With std:
			Use the old std (obtained from the init_set) to standardize test_set
		ii) With kjl:
			Project test_set with the new/updated KJL in step 2 to obtain the projected test data (prj_test)
		iii) Get AUC on the prj_test
 
	Repeat 2 and 3 steps (10 times)






=======================
Configuration:
	Fridge: idle, open_shut, and browse
	Ratio = 100:0 (one example)
		i) init_set: open_shut(500)
		ii) arrival_set: browse(500)
		iii) test_set: normal (open_shut(100) + browse(100)) + abnormal (idle(200))

Main steps (default params): 
	1. Build models with the default params (i.e., n_components (obtained by meanshift) and q_kjl=0.9) and get the model on test_set.
		i) With std:
			Get std on init_set
			Standardize init_set and test_set to obtain std_init_set and std_test_set 
		ii) With kjl:
			Project init_set (i.e, std_init_set) and test_set (i.e., std_test_set)
		iii) Build the model on the projected init_set and evaluate it on the projected test_set 
			Choose the best one (Model)
	
	
	Split arrive_set into 10 batches and each batch (batch_data_i, i is from 1 to 10) has 50 datapoints
	2. Retrain/update the best model  
		a) For Batch GMM: 
			i) With std: 
				Use the old std (obtained from the init_set) to standardize batch_data_i
			ii) Add standardized batch_data_i into std_init_set
				std_init_set = [std_init_set, batch_data_i]
			iii) With kjl: 
				1) Recompute sigma (the new sigma) on the std_init_set
				2) Reselect n (n=100) datapoints from std_init_set
				3) Recompute KJL with the new sigma
				4) Project std_init_set to obtain the projected data (prj_train).
			iv) use meanshift on prj_train to get the n_component
			v) Retrain the model on prj_train
			
		b) For Online GMM:
			i) With std: 
				Use the old std (obtained from the init_set) to standardize batch_data_i
			ii) Add standardized batch_data_i into std_init_set
				std_init_set = [std_init_set, batch_data_i]
			iii) With kjl:
				1) Random select m rows (m_rows) from batch_data_i (m= batch_size_i / std_init_set_size * n, here n=100)
				2) Replace the m random rows in Xrow with the m_rows 
				3) Recompute KJL with the old sigma (obtained from the init_set)
				4) Project std_init_set to obtain the projected_data (prj_train)
			iv) use meanshift on prj_train to get the n_component
			v) Update the model with prj_train
			
	3. Evaluate the updated model on test_set.
		I) With std:
			Use the old std (obtained from the init_set) to standardize test_set
		ii) With kjl:
			Project test_set with the new/updated KJL in step 2 to obtain the projected test data (prj_test)
		iii) Get AUC on the prj_test
 
	Repeat 2 and 3 steps (10 times)










Try different n_kjl, d_kjl









































Configuration:
	Fridge: idle, open_shut, and browse
	Ratio = 100:0 (one example)
		i) init_set: open_shut(500)
		ii) arrival_set: browse(500)
		iii) test_set: normal (open_shut(100) + browse(100)) + abnormal (idle(200))

Main steps (Best params): 
	1. Build models with different params (i.e., n_components and q_kjl) and get the best model on test_set.
		i) With std:
			Get std on init_set
			Standardize init_set and test_set to obtain std_init_set and std_test_set 
		ii) With kjl:
			Project init_set (i.e, std_init_set) and test_set (i.e., std_test_set)
		iii) Build the model on the projected init_set and evaluate it on the projected test_set 
			Choose the best one (Model)
	
	
	Split arrive_set into 10 batches and each batch (batch_data_i, i is from 1 to 10) has 50 datapoints
	2. Retrain/update the best model  
		a) For Batch GMM: 
			i) With std: 
				Use the old std (obtained from the init_set) to standardize batch_data_i
			ii) Add standardized batch_data_i into std_init_set
				std_init_set = [std_init_set, batch_data_i]
			iii) With kjl: 
				Reselect n (n=100) datapoints from std_init_set, recompute KJL, and project std_init_set to obtain the projected data (prj_train).
			iv) Retrain the model on prj_train
			
		b) For Online GMM:
			i) With std: 
				Use the old std (obtained from the init_set) to standardize batch_data_i
			ii) Add standardized batch_data_i into std_init_set
				std_init_set = [std_init_set, batch_data_i]
			ii) With kjl:
				Update the previous KJL with batch_data_i
				Project std_init_set to obtain the projected_data (prj_train)
			iii) Update the model with prj_data	
			
	3. Evaluate the updated model on test_set.
		I) With std:
			Use the old std (obtained from the init_set) to standardize test_set
		ii) With kjl:
			Project test_set with the new/updated KJL in step 2 to obtain the projected test data (prj_test)
		iii) Get AUC on the prj_test
 
	Repeat 2 and 3 steps (10 times)


=================================================

Configuration:
	ratio = 100:0
	i) init_train: normal1 (500) -activity1  
	ii) arrival: normal2 (500) - activity2
	iii) test: normal (normal1(100) + normal2(100)) + abnormal(abnormal1(100) + abnormal2(100)) 

	E.g., 
		Fridge: idle, open_shut, and browse
		ratio = 100:0
		init_train: open_shut(500)
		arrival: browse(500)
		test_set: normal (open_shut(100) + browse(100)) + abnormal (idle(200))

Main steps (Best params): 
	1. build models with different params (n_comps and q_kjl) and get the best model on test_set
		i) with std:
			get std on init_set and standardize the init_set and test_set 
		ii) with kjl:
			project init_set and test_set 
		iii) build the model on the projected init_set and evaluate it on the projected test_set 
			choose the best one

	2. retrain/update the best model  
		For Batch GMM:
			I) add new data (select 50 datapoints from arrival data) into init_train (new_train)
			ii) with std: 
				get the old std (remain), and standardize the new_train				
			iii) with kjl: 
				reselect n datapoints, recompute KJL, and project new_train to obtain prj_train.
			iv) build the model on prj_train
			
		For Online GMM:
			I) with std: 
				update std with the new data (50)
				standardize the new data  
			ii) with kjl:
				update kjl with the new data
				get the prj_data1 on the new_data
			iii) standardize the init_acculumated_set and project it to get the prj_data2
			iv) combine the prj_data1 and prj_data2 into prj_data1
			v) update the model with prj_data	
			vI) init_acculumated_set = [init_acculumated_set, new_data(50)]

	3. Evaluate the updated model on the test_set (projected test_set).
		get AUC

	Repeat 2 and 3 steps (10 times)


 




￼












Attempts
1. different n and d for KJL
2. different init_size 
3. meanshift


			Ratio = 100:0
 : normal_0, abnormal_0, normal_1, abnormal_1

init_size: 
   train: normal_0 (500)
   test: normal_0 (50) + abnormal_0 (50) + normal_1(50) +abnormal_1(50)

arrive:  normal_1 (500)

test_set:  normal_0 (100) + abnormal_0(100) + normal_1(100) +abnormal_1(100)	Ratio = 100:0
Fridge :open_shut, idle, browse,  idle1

init_size: 
   train: open_shut (500)
   test: open_shut (50) + idle(50) +browse(50)+idle1 (50) 

arrive:  browse (500)

test_set: open_shut (100) + idle(100) +browse(100)+ idle1 (100)	Ratio = 100:0
Fridge :  browse, idle1, open_shut, idle

init_size: 
   train:  browse (500)
   test:browse(50) + idle1 (50) + open_shut (50) + idle(50) 

arrive:  open_shut (500)

test_set: browse(100) + idle1 (100) + open_shut (100) +  idle (100)	Ratio = 100:0
Fridge : idle, open_shut, idle1, browse

init_size: 
   train: idle (500)
   test: idle (50) + open_shut (50) + idle1(50) +browse(50)

arrive:  idle2 (500)

test_set:  idle (100) + open_shut (100) + idle1(100) +browse(100)



Fridge: normal: idle , abnormal: browse

Dishwasher: normal: idle, abnormal: open


init_size: 
   train: fridge_idle (3000)
   test: fridge_idle (50) + fridge_browse (50) + dishwasher_idle(50) +dishwasher_open (50)

arrive:  dishwasher_idle (3000)

test_set:  fridge_idle (100) + fridge_browse (100) + dishwasher_idle(100) +dishwasher_open(100)	Fridge: normal: idle , abnormal: open

Dishwasher: normal: idle, abnormal: open


init_size: 
   train: fridge_idle (3000)
   test: fridge_idle (50) + fridge_open (50) + dishwasher_idle(50) +dishwasher_open (50)

arrive:  dishwasher_idle (3000)

test_set:  fridge_idle (100) + fridge_open (100) + dishwasher_idle(100) +dishwasher_open(100)
normal: fridge idle , abnormal: open (500)
normal: fridge idle1 , abnormal: browse (500)

init_size: 
   train: idle (3000)
   test idle (50) + open (50) + idle1(50) +browse (50)

arrive:  idle1 (3000)

test_set:  idle (100) + open (100) + idle1(100) +browse (100)



