'''
	Perceptron Monocapa
	- Aguilar Licona Marco Antonio
	- Cortéz Abraham
	Aprendizaje
	2018-2
'''

from numpy import dot,array
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
			datos_csv[cont_linea] += (((int(linea[len(linea) - 1]))),)
			linea.pop()			
			datos_csv[cont_linea] += (array(([int(i) for i in linea])),)
			cont_linea = cont_linea + 1
	return datos_csv

hardlim = lambda x: 0 if x < 0 else 1

nombre_archivo = 'ejercicio_clase_t1.csv'
datos_entrenamiento = cargarCsv(nombre_archivo)

w = array([0,0])
bias = 0
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
	if a_resultante != objetivo:
		error = objetivo - a_resultante
		lista_errores.append(error)
		w = w + (error * p.transpose())
		bias = bias + error
		numero_aciertos = 0
	else:
		numero_aciertos = numero_aciertos + 1
	iteraciones = iteraciones + 1

print("Pesos finales")
print(w)
print("Bias final")
print(bias)
print("No. total de iteraciones")
print(iteraciones)