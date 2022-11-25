from numpy import exp, array, random


class percepton():
    def __init__ (self, obs_entr, prediccion):
        self.obs_entr = obs_entr
        self.prediccion = prediccion
        self.limiteMin = -1
        self.limiteMax = 1
        self.sesgo = 1
        self.wb = 0
        self.txAprendizaje = 0.1
        self.epochs = 300000


    def pes_inicial (self):
        w11 = (self.limiteMax-self.limiteMin) * random.random() + self.limiteMin
        w21 = (self.limiteMax-self.limiteMin) * random.random() + self.limiteMin
        w31 = (self.limiteMax-self.limiteMin) * random.random() + self.limiteMin
        peso = [w11, w21, w31, self.wb]
        return  peso

    def suma_ponderada(X1,W11,X2,W21,B,WB):
        return (B*WB+( X1*W11 + X2*W21))

    def funcion_activacion_sigmoide(valor_suma_ponderada):
        return (1 / (1 + exp(-valor_suma_ponderada)))

    def funcion_activacion_relu(valor_suma_ponderada):
        return (max(0,valor_suma_ponderada))

    def error_lineal(valor_esperado, valor_predicho):
        return (valor_esperado-valor_predicho)

    def calculo_gradiente(valor_entrada,prediccion,error):
        return (-1 * error * prediccion * (1-prediccion) * valor_entrada)

    def calculo_valor_ajuste(valor_gradiente, tasa_aprendizaje):
        return (valor_gradiente*tasa_aprendizaje)

    def calculo_nuevo_peso (valor_peso, valor_ajuste):
        return (valor_peso - valor_ajuste)

    def calculo_MSE(predicciones_realizadas, predicciones_esperadas):
        i=0;
        suma=0;
        for prediccion in predicciones_esperadas:
            diferencia = predicciones_esperadas[i] - predicciones_realizadas[i]
            cuadradoDiferencia = diferencia * diferencia
            suma = suma + cuadradoDiferencia
        media_cuadratica = 1 / (len(predicciones_esperadas)) * suma
        return media_cuadratica

