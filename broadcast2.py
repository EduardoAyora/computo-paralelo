from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
tamanio=comm.Get_size()
rank = comm.Get_rank()



if rank == 0:
    tm = int(input("Ingrese el tamaño de la matriz: "))
    size_ = (tm, tm)
    matriz1 = np.random.randint(10, size=size_).astype("float") / 100
    matriz2 = np.random.randint(10, size=size_).astype("float") / 100
    producto_res = np.dot(matriz1, matriz2)
   
    variable_to_share = [matriz1,matriz2]
    # comm.send(producto_res, dest=9)


           
else:
   variable_to_share = 100

variable_to_share = comm.bcast(variable_to_share, root=0)
# data3 = list([0, 0, 0, 0, 0, 0, 0, 0])
for kk in range(1, tamanio-1):
    if rank == kk:
        filas = np.vsplit(variable_to_share[0], tamanio-2)
        columnas = variable_to_share[1]
        fila = filas[rank-1]
        filas_a = len(fila)
        filas_b = len(columnas)
        columnas_a = len(fila[0])
        columnas_b = len(columnas[0])
        # print(filas_a, " ", columnas_a, " ", columnas_a, " ", columnas_b) 
        size_ = (filas_a, columnas_a)
        calculo = np.zeros(dtype=float, shape=size_)
        for i in range(filas_a):
            for j in range(columnas_b):
                suma = 0
                for k in range(columnas_a):
                    suma += fila[i][k] * columnas[k][j]
                calculo[i][j] = suma
        envia1 = calculo
        # data2= calculo
        # print(kk)
        comm.send(envia1, dest=tamanio-1)

# print(data2) 
# print("         ")

if rank == tamanio-1:
    producto_res = np.dot(variable_to_share[0], variable_to_share[1])
    data2 = list()
    for i in range(1, tamanio-1):
        data2.append(comm.recv(source=i))
        ListaUnida = np.concatenate(data2, axis=0)
    # ListaUnida = np.concatenate((data2[0], data2[1], data2[2], data2[3], data2[4], data2[5], data2[6], data2[7]), axis=0)
    print('¿Los resultados son correctos?', np.allclose(ListaUnida, producto_res))

