import numpy as np

class NeuronioDeleta:
    def __init__(self, num_entradas, taxa_aprendizado=0.1, epocas=1000):
        self.pesos = np.random.rand(num_entradas)
        self.bias = np.random.rand()
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def prever(self, entradas):
        u = np.dot(entradas, self.pesos) + self.bias
        return self.sigmoid(u)

    def treinar(self, entradas_treino, saidas_desejadas):
        for epoca in range(self.epocas):
            erro_total = 0
            for entradas, saida_desejada in zip(entradas_treino, saidas_desejadas):
                saida_atual = self.prever(entradas)
                erro = saida_desejada - saida_atual
                erro_total += erro**2

                delta = erro * saida_atual * (1 - saida_atual)
                self.pesos += self.taxa_aprendizado * delta * entradas
                self.bias += self.taxa_aprendizado * delta

            if epoca % 100 == 0:
                print(f"Época {epoca}, Erro total: {erro_total:.4f}")

            if erro_total < 0.001:
                print(f"Treinamento concluído na época {epoca}")
                break

X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])

print("Treinando para AND:")
neuronio_and = NeuronioDeleta(num_entradas=2)
neuronio_and.treinar(X_and, y_and)

print("\nTreinando para OR:")
neuronio_or = NeuronioDeleta(num_entradas=2)
neuronio_or.treinar(X_or, y_or)

def testar_neuronio(neuronio, X, operacao):
    print(f"\nResultados para {operacao}:")
    for entradas in X:
        previsao = neuronio.prever(entradas)
        print(f"Entradas: {entradas}, Saída: {previsao:.4f}")

testar_neuronio(neuronio_and, X_and, "AND")
testar_neuronio(neuronio_or, X_or, "OR")
