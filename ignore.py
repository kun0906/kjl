"""
	python3 ignore.py

"""
import os


def search(in_dir = None):
	ignored_files = []
	for f in os.listdir(in_dir):
		f = os.path.join(in_dir, f)
		if os.path.isfile(f):
			size = os.path.getsize(f)
			if size > 20 * 1000 * 1000: # 20 MB
				ignored_files.append((f, size))
		else:
			ignored_files.extend(search(in_dir = f))

	return ignored_files

def write(ignore_files, ignore_path = '.gitignore'):
	with open(ignore_path, 'w') as f:
		for file, size in ignore_files:
			print(f'{file}, {int(size)/(1000*1000):.2f}MB')
			f.write(file + '\n')


if __name__ == '__main__':
	ignored_files = search(in_dir='.')
	print(len(ignored_files))
	print(ignored_files[:10])
	write(ignored_files, ignore_path='.gitignore_new')
	print(len(ignored_files))