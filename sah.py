import tensorflow as tf #biblioteka za rad sa neuralnim mrezama
import numpy as np


file = open('baza.data', 'r') #citanje iz baze podataka


def coder(lista): #funkcija kodiranja ulaza binarno
	ret=np.zeros(66)# 6*8=48 i 18 za izlaze je 66
	niz_brojeva=['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','draw']
	for i in range(0,7):#prolazimo kroz svaki string
		if(i%2==0):# kodiramo ili slovo ili rezultat
			if(i<6): # kodiramo broj
				num=ord(lista[i])-ord('a')#kodiramo polja oznacena tipom char
				ret[num+i*8]=1
			else:	#kodiramo rezultat
				num=niz_brojeva.index(lista[i])#kodiramo izlaz koji je dat kao string
				ret[num+i*8]=1

		else:
			ret[int(lista[i])- 1 +i*8]=1#kodiramo broj sa ulaza (-1 jer je indeksiranje od 0)
	return ret


txt=[]
for line in file:
    txt.append(coder(line.strip().split(',') ))#prosledjujemo niz od 7 stringova

np.random.shuffle(txt)#ovo zbog toga da lepo mozemo da podelimo delove za trening, test i validaciju

file.close()

#izdvajamo delove iz baze podataka za trening, validaciju i test
Z_train=txt[0:16834]
Z_validation=txt[16834:22445]
Z_test=txt[22445:28056]

#delimo Z_train,Z_validation i Z_test na ulaze i izlaze
X_train=[]
X_validation=[]
X_test=[]
Y_train=[]
Y_validation=[]
Y_test=[]

for i in range (0, len(Z_train)):
	X_train.append(Z_train[i][0:48])
	Y_train.append(Z_train[i][48:66])

for i in range (0, len(Z_validation)):
	X_validation.append(Z_validation[i][0:48])
	Y_validation.append(Z_validation[i][48:66])
	
	X_test.append(Z_test[i][0:48])
	Y_test.append(Z_test[i][48:66])






# alociramo memoriju, jer tako mora u tf, shape(2,48) je jer guramo dva po dva ulaza
# mogli smo uzimati i vise ulaza od dva, onda se brze prolazi jedna epoha, ali je potreban veci broj epoha za dostizanje odredjene tacnosti
X = tf.placeholder(shape=(2,48) ,dtype=tf.float64, name='X')
y = tf.placeholder( dtype=tf.float64, name='y')

#nesto sto je neophodno da program radi, treba pogledati cemu tacno sluzi
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#gotova neuralna mreza
hidden1 = tf.layers.dense(inputs=X, units=70, activation=tf.nn.relu)#prvi skriveni sloj neurona
hidden2 = tf.layers.dense(inputs=hidden1, units=80, activation=tf.nn.relu)#drugi skriveni sloj neurona
hidden3 = tf.layers.dense(inputs=hidden2, units=70, activation=tf.nn.relu)#treci skriveni sloj neurona
y_est = tf.layers.dense(inputs=hidden3, units=18)



# Definisanje funkcije gubitka
loss = tf.losses.softmax_cross_entropy(y, y_est) #softmax_cross_entropy je ta funkcija koja racuna tacnost na osnovu tacnih i procenjenih izlaza, a ima i drugih koje mozemo da biramo

# Definisanje trening operatora koji sluzi za minimizaciju gubitka
optimizer = tf.train.AdamOptimizer()

#minimizacija funkcije gubitka
train = optimizer.minimize(loss)

correct=tf.equal(tf.argmax(y_est,1),tf.argmax(y,1)) #da li su na istim mestima jedinice true/false
accuracy=tf.reduce_sum(tf.cast(correct,'float')) #pretvaramo niz booleana u float i trazimo aritmeticku sredinu



init = tf.global_variables_initializer() #inicijalizujemo promenljive
sess = tf.Session() #pravljenje sesije koja sluzi za racunanje (sve zivo), jer do sada imamo samo strukturu programa
sess.run(init) # skuplja sve gore deklarisano da ih ima i da raspolzae sa njima na jednom mestu


saver = tf.train.Saver() #za cuvanje podataka
# saver.restore(sess,'./mreza') fora je da nastavi od prethodnog sto je radjeno, uzima podatke iz fajla

counter=0
lvalPR=0

for i in range(35):
	ltr=0 #funkcija gubitka za trening skup
	lval=0 #funkcija gubitka na skupu za validaciju
	acc=0 #tacnost na trening skupu
	acc_val=0 #tacnost na skupu za vaidaciju
	for j in range(8417): #uzimamo dva po dva pa je ovo polovina od celog trening skupa
		inpt={X: X_train[2*j:2*j+2], y: Y_train[2*j:2*j+2]}
		sess.run(train, feed_dict=inpt) # u prethodno napravljene placeholdere stavljamo dva po dva podatka OVDE SE DOGADJA PROMENA NEURALNE MREZE
		ltr+=loss.eval(inpt,session=sess) # za svaka dva ulaza mi racunamo funkciju gubitka
		acc+=accuracy.eval(inpt,session=sess) #racuna tacnost dva po dva pa mnozimo sa 2 da bi dobili ukupan borj poklapanja
	ltr=ltr/16834
	acc=acc/16834

	for j in range(2805): #nakon svake epohe prolazimo kroz skup za validaciju
		lval+=loss.eval({X: X_validation[2*j:2*j+2], y: Y_validation[2*j:2*j+2]},session=sess)
		acc_val+=accuracy.eval({X: X_validation[2*j:2*j+2], y: Y_validation[2*j:2*j+2]},session=sess)
	lval=lval/5610
	acc_val=acc_val/5610
	
	print('----------------------------------------------------------')
	print('|'+(str(i+1)).center(10) + '|' + 'Trening'.center(22) + '|' + 'Validacija'.center(22)+'|')
	print('----------------------------------------------------------')
	print('|'+'Greska'.center(10) + '|' + (str(ltr)).center(22) + '|' + (str(lval)).center(22) + '|')
	print('----------------------------------------------------------')
	print('|' + "{:^10}".format('Tacnost') + '|' + "{:^22,.3f}".format(float(acc)) + '|' + "{:^22,.3f}".format(float(acc_val))+'|')
	print('----------------------------------------------------------')
	print("\n")

	if(lvalPR>lval):
		save_path = saver.save(sess, './mreza')
		counter=0
	else:
		counter+=1
	if(counter>=4):
		break

	lvalPR=lval
	
	
ltest=0
acctest=0
for i in range(2805):
    ltest+=loss.eval({X : X_test[2*i:2*i+2], y: Y_test[2*i:2*i+2]},session=sess)
    acctest+=accuracy.eval({X : X_test[2*i:2*i+2], y: Y_test[2*i:2*i+2]},session=sess)

ltest=ltest/5610
acctest=acctest/5610
print("Test".center(25) + '\n')
print("Greska: " + str(ltest))
print("Tacnost: " + "{:.3f}".format(float(acctest)))


sess.close()
