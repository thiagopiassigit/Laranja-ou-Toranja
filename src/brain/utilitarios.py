from keras.layers import Dense


def criar_neuronios(camadas_ocultas, neuronios, func_ativacao):
    camadas = list()

    for i in range(camadas_ocultas): camadas.append(Dense(neuronios, activation=func_ativacao))

    return camadas