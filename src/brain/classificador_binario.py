from keras import Sequential
from keras.layers import Dense
from brain.utilitarios import *


def configurar_e_treinar(entradas,
                         saidas,
                         camadas_ocultas,
                         neuronios_oculstos,
                         func_ativacao,
                         epocas=500,
                         otimizador="adam",
                         func_perda="binary_crossentropy",
                         metricas=['binary_accuracy'],
                         neuronios_entrada=None):

    neuronios_entrada = neuronios_oculstos if neuronios_entrada is None else neuronios_entrada

    camadas = criar_neuronios(camadas_ocultas, neuronios_oculstos, func_ativacao)

    camadas.insert(0, Dense(neuronios_entrada, activation=func_ativacao, input_shape=(entradas.shape[1],)))
    camadas.append(Dense(1, activation='sigmoid'))

    modelo = Sequential(camadas)

    modelo.compile(optimizer=otimizador, loss=func_perda, metrics=metricas)

    modelo.fit(entradas, saidas, epochs=epocas)

    return modelo