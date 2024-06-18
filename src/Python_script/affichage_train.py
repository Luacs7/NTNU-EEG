import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Texte à analyser
text = """
Train = MCR = 0.178+/-0.382 (sig. at p=0.000; N=411, batch data)
Test = MCR = 0.170+/-0.376 (sig. at p=0.000; N=47, batch data)

Train = MCR = 0.174+/-0.379 (sig. at p=0.000; N=368, batch data)
Test = MCR = 0.189+/-0.391 (sig. at p=0.000; N=90, batch data)

Train = MCR = 0.182+/-0.386 (sig. at p=0.000; N=318, batch data)
Test = MCR = 0.171+/-0.377 (sig. at p=0.000; N=140, batch data)

Train = MCR = 0.167+/-0.373 (sig. at p=0.000; N=275, batch data)
Test = MCR = 0.186+/-0.389 (sig. at p=0.000; N=183, batch data)

Train = MCR = 0.163+/-0.369 (sig. at p=0.000; N=227, batch data)
Test = MCR = 0.190+/-0.393 (sig. at p=0.000; N=231, batch data)

train = MCR = 0.188+/-0.391 (sig. at p=0.000; N=181, batch data)
Test = MCR = 0.199+/-0.399 (sig. at p=0.000; N=277, batch data)

Train = MCR = 0.173+/-0.378 (sig. at p=0.000; N=139, batch data)
Test = MCR = 0.207+/-0.405 (sig. at p=0.000; N=319, batch data)

Train = MCR = 0.115+/-0.319 (sig. at p=0.000; N=113, batch data)
Test = MCR = 0.255+/-0.436 (sig. at p=0.000; N=345, batch data)

Train = MCR = 0.097+/-0.296 (sig. at p=0.000; N=93, batch data)
Test = MCR = 0.238+/-0.426 (sig. at p=0.000; N=365, batch data)

Train = MCR = 0.130+/-0.337 (sig. at p=0.000; N=69, batch data)
Test = MCR = 0.254+/-0.436 (sig. at p=0.000; N=389, batch data)

Train = MCR = 0.133+/-0.340 (sig. at p=0.000; N=45, batch data)
Test = MCR = 0.298+/-0.457 (sig. at p=0.000; N=413, batch data)

Train = MCR = 0.000+/-0.000 (sig. at p=0.000; N=24, batch data)
Test = MCR = 0.279+/-0.448 (sig. at p=0.000; N=434, batch data)

Train = MCR = 0.062+/-0.242 (sig. at p=0.000; N=32, batch data)
Test = MCR = 0.333+/-0.471 (sig. at p=0.000; N=426, batch data)

Train = MCR = 0.000+/-0.000 (sig. at p=0.000; N=14, batch data)
Test = MCR = 0.311+/-0.463 (sig. at p=0.000; N=444, batch data)

Train = MCR = 0.000+/-0.000 (sig. at p=0.000; N=5, batch data)
Test = MCR = 0.521+/-0.463 (sig. at p=0.000; N=453, batch data)


"""

# Expression régulière pour extraire les valeurs
pattern = re.compile(
    r"(Train|train) = MCR = (?P<train_mcr>\d+\.\d+)\+/\-(?P<train_std>\d+\.\d+) \(sig\. at p=0\.000; N=(?P<train_n>\d+), batch data\)\n"
    r"Test = MCR = (?P<test_mcr>\d+\.\d+)\+/\-(?P<test_std>\d+\.\d+) \(sig\. at p=0\.000; N=(?P<test_n>\d+), batch data\)"
)

# Liste pour stocker les valeurs extraites
data = []

# Recherche et extraction des valeurs
for match in pattern.finditer(text):
    data.append(match.groupdict())


# Création du DataFrame
df = pd.DataFrame(data)

N= 458

Accuracy_train = []
Accuracy_test = []
N_TEST=[]
for elem in df["train_mcr"]:
    Accuracy_train +=[1- float(elem)]
for elem in df["test_mcr"]:
    print(elem)
    Accuracy_test +=[1- float(elem)]
for elem in df["test_n"]:
    N_TEST +=[float(elem)]

N_TEST = np.array(N_TEST)

Ratio = N-N_TEST
plt.grid()
# plt.xlim(0,1)
plt.ylim(0,1)
plt.scatter(Ratio,Accuracy_train)
plt.scatter(Ratio,Accuracy_test)
plt.show()



























