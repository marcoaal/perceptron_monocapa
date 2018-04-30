'''
	Perceptron Monocapa
	- Aguilar Licona Marco Antonio
	- Cortéz Abraham
	Aprendizaje
	2018-2
'''

from numpy import dot,array,isscalar,array_equal,multiply,outer
from csv import reader

def cargarCsv(archivo):
	datos_csv = []
	with open(archivo, 'r') as archivo_csv:
		csv = reader(archivo_csv)
		cont_linea = 0
		for linea in csv:
			if not linea:
				continue
			datos_csv.append(())
			if "." in linea[len(linea) - 1]:
				objetivos = linea[len(linea) - 1].split(".")
				datos_csv[cont_linea] += (array(([int(i) for i in objetivos])),)
			else:
				datos_csv[cont_linea] += (((int(linea[len(linea) - 1]))),)
			linea.pop()			
			datos_csv[cont_linea] += (array(([int(i) for i in linea])),)
			cont_linea = cont_linea + 1
	return datos_csv

def hardlim(x):
	if isscalar(x):
		if x<0:
			return 0
		else:
			return 1
	else:
		hardlim_elementos = []
		for elemento in x:
			if elemento<0:
				hardlim_elementos.append(0)
			else:
				hardlim_elementos.append(1)
		return array(hardlim_elementos)

num_ejercicio = 0
while (int(num_ejercicio)<1 or int(num_ejercicio)>4):
	print("----- Perceptrón multicapa -----")
	print("\nSeleccione el ejercicio para el cálculo de pesos y bias")
	print(" 1) Ejercicio 1\n 2) Ejercicio 2\n 3) Ejercicio 3\n 4) Ejercicio 4")
	num_ejercicio = input("\nNúmero de ejercicio: ")

	while not num_ejercicio.isdigit():
		num_ejercicio = input("\nLa entrada debe ser un número: ")

	if int(num_ejercicio) == 1:
		print("\n----- Ejercicio 1 -----\n")
		nombre_archivo = 'ejercicio_1.csv'
		w = array([-7,-5])
		bias = 4
		'''
		nombre_archivo = 'ejercicio_prueba.csv'
		w = array([0,0])
		bias = 0
		'''		
	elif int(num_ejercicio) == 2:
		print("\n----- Ejercicio 2 -----\n")
		nombre_archivo = 'ejercicio_2.csv'
		w = array([-7,-5])
		bias = 4
	elif int(num_ejercicio) == 3:
		print("\n----- Ejercicio 3 -----\n")
		nombre_archivo = 'ejercicio_3.csv'
		w = array([[4,0],[0,4]])
		bias = array([4,4])
	elif int(num_ejercicio) == 4:
		print("\n----- Ejercicio 4 -----\n")
		nombre_archivo = 'ejercicio_4.csv'
		w = array([[3,-3],[-3,3]])
		bias = array([-3,3])

datos_entrenamiento = cargarCsv(nombre_archivo)
lista_errores = []

iteraciones_max = 100
numero_aciertos = 0
iteraciones = 0
pos = 0

while (numero_aciertos < len(datos_entrenamiento) and iteraciones < iteraciones_max):
	if pos > len(datos_entrenamiento) - 1:
		pos = 0
	objetivo,p = datos_entrenamiento[pos]
	pos = pos + 1

	a_resultante = hardlim(dot(w, p) + bias)

	if isscalar(a_resultante):
		if a_resultante != objetivo:
			error = objetivo - a_resultante
			lista_errores.append(error)
			w = w + (error * p)
			bias = bias + error
			numero_aciertos = 0
		else:
			numero_aciertos = numero_aciertos + 1
	else:
		if not array_equal(a_resultante,objetivo):
			error = objetivo - a_resultante
			lista_errores.append(error)
			w = w + outer(error.T,p)
			bias = bias + error
			numero_aciertos = 0
		else:
			numero_aciertos = numero_aciertos + 1
	iteraciones = iteraciones + 1

print("----- Pesos Finales -----")
print(w)
print("----- Bias Final -----")
print(bias)
print("----- No. Total De Iteraciones -----")
print(iteraciones)