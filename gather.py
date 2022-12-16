import numpy as np
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

numero_procesos = size

def multiplicacion_matrices_paralela(params):
    numero_proceso = params["section"]
    print(numero_proceso)
    matriz1 = params["matriz1"]
    matriz2 = params["matriz2"]
    columnas_a = len(matriz1[0])
    columnas_b = len(matriz2[0])
    filas_por_proceso = int(len(matriz1) / numero_procesos)
    multiplicacion = np.zeros(dtype=float, shape=(
        filas_por_proceso, len(matriz1)))
    for i in range(numero_proceso * filas_por_proceso, (numero_proceso + 1) * filas_por_proceso):
        for j in range(columnas_b):
            suma = 0
            for k in range(columnas_a):
                suma += matriz1[i][k] * matriz2[k][j]
            multiplicacion[i - (numero_proceso * filas_por_proceso)][j] = suma
    return multiplicacion

resultado_proceso = 0
if rank==0:
  size_ = (200, 200)
  matriz1 = np.random.randint(10, size=size_).astype("float") / 100
  matriz2 = np.random.randint(10, size=size_).astype("float") / 100
  resultado_proceso = multiplicacion_matrices_paralela({"section": 0, "matriz1": matriz1, "matriz2": matriz2})
  comm.send({"section": 1, "matriz1": matriz1, "matriz2": matriz2}, dest=1)
  comm.send({"section": 2, "matriz1": matriz1, "matriz2": matriz2}, dest=2)
  comm.send({"section": 3, "matriz1": matriz1, "matriz2": matriz2}, dest=3)
    

if rank==1:
    data=comm.recv(source=0)
    resultado_proceso = multiplicacion_matrices_paralela(data)

if rank==2:
    data=comm.recv(source=0)
    resultado_proceso = multiplicacion_matrices_paralela(data)

if rank==3:
    data=comm.recv(source=0)
    resultado_proceso = multiplicacion_matrices_paralela(data)
   
matriz_multiplicacion_paralela_comp = comm.gather(resultado_proceso, root=0)
if rank==0:
    print(matriz_multiplicacion_paralela_comp)
    paralela_res = np.reshape(
    matriz_multiplicacion_paralela_comp, size_)
    print("Mis resultados obtenidos en paralelo")
    for i in range(4):
        print("%f" % (paralela_res[i][i]))
    print('Resultados de numpy')
    producto_res = np.dot(matriz1, matriz2)
    #producto_res = producto_res.T
    for i in range(4):
        print("%f" % (producto_res[i][i]))

    print(np.array(matriz_multiplicacion_paralela_comp).shape)

    print('¿Los resultados son correctos?',
        np.allclose(producto_res, paralela_res))