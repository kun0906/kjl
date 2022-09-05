#!/bin/sh
### note: run main_reload.sh under the 'applications' directory, i.e. ./offline/main_reload.sh

######################################################################################################################
echo "Starting experiments..."
start_time=$(date +'%Y-%m-%d %H:%M:%S')
echo $start_time

echo 'PWD:' $PWD
PATH=$PATH:$PWD # add current directory to PATH
#echo 'PATH:' $PATH

######################################################################################################################
mkdir "speedup/out/"
for dataset in UNB345_3 CTU1 MAWI1_2020 MACCDC1 SFRIG1_2020 AECHO1_2020 DWSHR_WSHR_2020; do
	for model in "OCSVM(rbf)" "KJL-GMM(full)" "Nystrom-GMM(full)"  "KJL-QS-init_GMM(full)" "Nystrom-QS-init_GMM(full)"; do
		cmd="PYTHONPATH=../:./ python3.7 -u speedup/main_reload_idv.py -d' ${dataset}' -m '${model}' > speedup/out/main_kjl_'${dataset}'_'${model}'.txt 2>&1"
#		PYTHONPATH=../:./ python3.7 -u offline/main_reload_idv.py -d ${datasets} -m ${models}> offline/out/main_kjl_${datasets}_${models}.txt 2>&1
		echo $cmd
		eval $cmd
	done
done

######################################################################################################################
echo "Finish experiments."
end_time=$(date +'%Y-%m-%d %H:%M:%S')
echo $start_time, $end_time
echo "$(($(date -d "$end_time" '+%s') - $(date -d "$start_time" '+%s'))) seconds"

# End of script
