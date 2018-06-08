import os

for i in os.listdir(os.getcwd()):
	try:
		number = int(i.split('.')[0].split('_')[-1])
		os.rename(i, 'test_%04d.jpg' % number)
	except:
		pass