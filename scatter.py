from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

size = comm.Get_size()
numero_procesos = size

if rank == 0:
    size_ = (200, 200)
    matriz1 = np.random.randint(10, size=size_).astype("float") / 100
    matriz2 = np.random.randint(10, size=size_).astype("float") / 100
    producto_res = np.dot(matriz1, matriz2)
    filas = np.vsplit(matriz1, 8)
    array_to_share = [
        [filas[0], matriz2],
        [filas[1], matriz2],
        [filas[2], matriz2],
        [filas[3], matriz2],
        [filas[4], matriz2],
        [filas[5], matriz2],
        [filas[6], matriz2],
        [filas[7], matriz2]
    ]
else:
    array_to_share = None

recvbuf = comm.scatter(array_to_share, root=0)

for i in range(8):
    if rank == i:
        filas = recvbuf[0]
        columnas = recvbuf[1]
        filas_a = len(filas)
        filas_b = len(columnas)
        columnas_a = len(filas[0])
        columnas_b = len(columnas[0])
        size_ = (filas_a, columnas_a)
        multiplicacion = np.zeros(dtype=float, shape=size_)
        for i in range(filas_a):
            for j in range(columnas_b):
                suma = 0
                for k in range(columnas_a):
                    suma += filas[i][k] * columnas[k][j]
                multiplicacion[i][j] = suma
        comm.send(multiplicacion, dest=0)

if rank == 0:
    multiplicacion_paralela = list([0, 0, 0, 0, 0, 0, 0, 0])

    for i in range(8):
        multiplicacion_paralela[i] = comm.recv(source=i)
    paralela_res = np.concatenate(
        (
            multiplicacion_paralela[0],
            multiplicacion_paralela[1],
            multiplicacion_paralela[2],
            multiplicacion_paralela[3],
            multiplicacion_paralela[4],
            multiplicacion_paralela[5],
            multiplicacion_paralela[6],
            multiplicacion_paralela[7]
        ),
        axis=0)

    print("Mis resultados obtenidos en paralelo")
    for i in range(4):
        print("%f" % (paralela_res[i][i]))
    print('Resultados de numpy')
    producto_res = np.dot(matriz1, matriz2)
    for i in range(4):
        print("%f" % (producto_res[i][i]))

    print('¿Los resultados son correctos?',
          np.allclose(paralela_res, producto_res))
