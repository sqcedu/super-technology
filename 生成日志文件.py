import time, os

new_time = time.strftime("%Y-%m-%d")
disk_status = os.popen('df -h').readlines()  #readlines
f = open(new_time+'.log', 'w')
f.write('%s\n' % disk_status)
f.flush()
f.close()
f.close()