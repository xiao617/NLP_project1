import subprocess

# GET : train_TG.txt & test_TG.txt
child1 = subprocess.Popen(['python3','TG.py'])
child1.wait()

# GET : train_PMI.txt & test_PMI.txt
child2 = subprocess.Popen(['python3','PMI.py'])
child2.wait()

# Use file got above to do LinearRegression()
print("\n*** Result ***")
child3 = subprocess.Popen(['python3','BOTH.py'])
child3.wait()