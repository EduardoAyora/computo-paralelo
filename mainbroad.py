from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def multiplicacion_matrices_paralela(params):
    numero_proceso = params["section"]
    matriz1 = params["matriz1"]
    matriz2 = params["matriz2"]
    columnas_a = len(matriz1[0])
    columnas_b = len(matriz2[0])
    filas_por_proceso = int(len(matriz1) / 4)
    multiplicacion = np.zeros(dtype=float, shape=(
        filas_por_proceso, len(matriz1)))
    for i in range(numero_proceso * filas_por_proceso, (numero_proceso + 1) * filas_por_proceso):
        for j in range(columnas_b):
            suma = 0
            for k in range(columnas_a):
                suma += matriz1[i][k] * matriz2[k][j]
            multiplicacion[i - (numero_proceso * filas_por_proceso)][j] = suma
    return multiplicacion

if rank == 2:
   variable_to_share = 100
           
else:
   variable_to_share = 0

variable_to_share = comm.bcast(variable_to_share, root=2)
print("process = %d" %rank + " variable shared  = %d " %variable_to_share)
