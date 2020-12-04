""" Save output to file by subprocess
https://janakiev.com/blog/python-shell-commands/
https://stackoverflow.com/questions/18421757/live-output-from-subprocess-command
https://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
"""

import subprocess


def run(in_file, out_file):
    # cmd = f"python3.7 {in_file} > {out_file} 2>&1"
    cmd = f"python3.7 {in_file}"
    print(cmd)
    with open(out_file, 'wb') as f:
        # buffsize: 0 = unbuffered (default value); 1 = line buffered; N = approximate buffer size, ...
        # p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        #                      bufsize=0, universal_newlines=True)
        p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             bufsize=1, universal_newlines=True)
        for line in p.stdout:
            # sys.stdout.write(line) # output to console
            f.write(line.encode())  # save to file
            print(line, end='')  # output to console

    return 0


in_file = '/online/demo.py'
in_file = '/main_gmm.py'
out_file = f'{in_file}-log.txt'
run(in_file, out_file)
