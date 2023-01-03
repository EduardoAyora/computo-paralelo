from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
tamanio = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    size_ = (200, 200)
    matriz1 = np.random.randint(10, size=size_).astype("float") / 100
    matriz2 = np.random.randint(10, size=size_).astype("float") / 100
    producto_res = np.dot(matriz1, matriz2)
    variable_to_share = [matriz1, matriz2]
else:
    variable_to_share = None

variable_to_share = comm.bcast(variable_to_share, root = 0)
for i in range(1, tamanio - 1):
    if rank == i:
        filas = np.vsplit(variable_to_share[0], tamanio - 2)
        columnas = variable_to_share[1]
        fila = filas[rank - 1]
        filas_a = len(fila)
        filas_b = len(columnas)
        columnas_a = len(fila[0])
        columnas_b = len(columnas[0])
        size_ = (filas_a, columnas_a)
        multiplicacion = np.zeros(dtype=float, shape=size_)
        for i in range(filas_a):
            for j in range(columnas_b):
                suma = 0
                for k in range(columnas_a):
                    suma += fila[i][k] * columnas[k][j]
                multiplicacion[i][j] = suma
        comm.send(multiplicacion, dest = 0)

if rank == 0:
    producto_res = np.dot(variable_to_share[0], variable_to_share[1])
    multiplicacion_paralela = list()
    for i in range(1, tamanio - 1):
        multiplicacion_paralela.append(comm.recv(source=i))
        paralela_res = np.concatenate(multiplicacion_paralela, axis=0)

    print("Mis resultados obtenidos en paralelo")
    for i in range(4):
        print("%f" % (paralela_res[i][i]))
    print('Resultados de numpy')
    producto_res = np.dot(matriz1, matriz2)
    for i in range(4):
        print("%f" % (producto_res[i][i]))
    
    print('¿Los resultados son correctos?',
          np.allclose(paralela_res, producto_res))
