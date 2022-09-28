"""
	python3 ignore.py

"""
import os


def search(in_dir = None):
	ignored_files = []
	add_files = []
	for f in os.listdir(in_dir):
		if f == '.git': continue
		f = os.path.join(in_dir, f)
		if os.path.isfile(f):
			size = os.path.getsize(f)
			if size > 2 * 1000 * 1000: # 2 MB
				ignored_files.append((f, size))
			else:
				add_files.append(f)
		else:
			v1, v2 = search(in_dir = f)
			add_files.extend(v1)
			ignored_files.extend(v2)

	return add_files, ignored_files

def write(ignore_files, ignore_path = '.gitignore'):
	with open(ignore_path, 'w') as f:
		ss = ['./legacy/', './**/*.DS_Store', '*.zip', '*.dat', '.idea/', '.git/']
		ss = '\n'.join(ss)
		f.write(ss + '\n\n')

		for i, (file, size) in enumerate(ignore_files):
			if i %100 == 0: print(f'{file}, {int(size)/(1000*1000):.2f}MB')
			f.write(file + '\n')

def add(add_files):
	for i, file in enumerate(add_files):
		cmd = f"git add -f \"{file}\""
		if i%100 == 0: print(file)
		os.system(cmd)

if __name__ == '__main__':
	add_files, ignored_files = search(in_dir='.')
	print(len(add_files), len(ignored_files))
	print(ignored_files[:10])
	# add(add_files)
	write(ignored_files, ignore_path='.gitignore')
	print(len(ignored_files))