import subprocess

child1 = subprocess.Popen(['python3','TG.py'])
child1.wait()

#child2 = subprocess.Popen(['python3','PMI.py'])
#child2.wait()

print("\n*** Result ***")
child3 = subprocess.Popen(['python3','BOTH.py'])
child3.wait()