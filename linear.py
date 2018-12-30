#Libraries
import numpy
import matplotlib.pyplot as plt

#Global Data
n=2
m=50
L_rate=1e-8

#Values to determine
theta=numpy.zeros(shape=(n,1) ,dtype=float)
file=open("dataset3.txt")
file.seek(0)
jtheta=[]
iteration=[]
thetaa=[]
no_iteration=1000

#Hyphothesis Function
def h(X):
	global theta
	return numpy.matmul(numpy.transpose(theta),X)

#Calculation Formula
def cal():
	global n,m,L_rate,theta
	tmp=numpy.zeros(shape=(n,1) ,dtype=float)
	for x in range(no_iteration):
			extra=0
			for i in range(m):
				data_set=file.readline()
				data_set=data_set.split()
				X=[float(a) for a in data_set[0:-1]]
				X=numpy.asarray(X)
				y=float(data_set[-1])
				diff=h(numpy.transpose(X))-y
				extra+=diff*diff/(2*m)
				#Using the Stochastic Gradient Descent(More superior and Fast)
				for j in range(n):
					tmp[j][0]=(L_rate*diff*X[j])
				theta=theta-tmp	
			file.seek(0)
			jtheta.append(extra)
			iteration.append(x+1)
			thetaa.append(theta[1][0])

#Ploting Graph only 2-D using theta0 value(As it difficult to understand the multi-dimensional Graph)
def Graph():
	global no_iteration,iteration,jtheta,m,thetaa;
	file.seek(0)
	yp = []	
	XX = []
	YY = []	
	Jtheta=numpy.zeros(shape=(n,1),dtype=float)
	for j in range(m):
			data_set=file.readline()
			data_set=data_set.split()
			X=[ float(a) for a in data_set[0:-1]]
			X=numpy.asarray(X)
			XX.append(X[1])
			YY.append(float(data_set[-1]))
			tmp=h(numpy.transpose(X))
			yp.append(float(tmp))
	
	#Graph of the Prediction vs Correction
	plt.ylabel('Price')
	plt.xlabel('Size')
	plt.plot(XX,yp,label="predict",color="blue")
	plt.scatter(XX,YY,label="correct",color="red")
	plt.title('Correct vs Prediction')
	plt.legend()
	plt.show()
	
	#Graph of the cost and iterations
	plt.xlabel('Iterations')
	plt.ylabel('cost')
	plt.title('Iterations affect on Cost')
	plt.plot(iteration,jtheta)
	plt.show()
	
	#Graph of the mean Square Formula
	plt.xlabel('theta')
	plt.ylabel('cost')
	plt.title('Mean Square Formula')
	plt.plot(thetaa,jtheta)
	plt.show()
	
#Function Calls	
cal()
Graph()