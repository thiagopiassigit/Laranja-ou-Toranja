from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from brain.classificador_binario import configurar_e_treinar

transformador = StandardScaler()

def criar_classes_binarias(dados):
    dados['name'].replace("orange", 0, inplace=True)
    dados['name'].replace("grapefruit", 1, inplace=True)


def repartir_dados(dados):
    entradas = dados.iloc[:, 1:]
    resultados = dados.iloc[:, :1]
    return entradas, resultados


def preparar_dados(dados):
    criar_classes_binarias(dados)
    entradas, resultados = repartir_dados(dados)
    entradas = transformador.fit_transform(entradas)
    return train_test_split(entradas, resultados, test_size=0.3)


if __name__ == '__main__':

    dados = read_csv('dados.csv')
    entradas_treino, entradas_teste, resultados_treino, resultados_teste = preparar_dados(dados)

    modelo = configurar_e_treinar(
        entradas_treino,
        resultados_treino,
        camadas_ocultas=2,
        neuronios_oculstos=20,
        epocas=300,
        otimizador="adam",
        neuronios_entrada=len(entradas_treino)
    )

    score = (modelo.evaluate(entradas_teste, resultados_teste)[1] * 100)

    print(f'{"%.2f" % score}%')