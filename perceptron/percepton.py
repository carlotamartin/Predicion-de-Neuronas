from numpy import exp, array, random
import matplotlib.pyplot as plt


class percepton():
    def __init__ (self, obs_entr, prediccion):
        self.observaciones_entrada = obs_entr
        self.prediccion = prediccion
        self.limiteMin = -1
        self.limiteMax = 1
        self.txAprendizaje = 0.1
        self.epochs = 3000
        self.Grafica_MSE = []


    def pes_inicial (self):
        w11 = (self.limiteMax-self.limiteMin) * random.random() + self.limiteMin
        w21 = (self.limiteMax-self.limiteMin) * random.random() + self.limiteMin
        w31 = (self.limiteMax-self.limiteMin) * random.random() + self.limiteMin
        wb = 0
        peso = [w11, w21, w31, wb]
        return  peso

    def suma_ponderada(self, X1,W11,X2,W21,B,WB):
        return (B*WB+( X1*W11 + X2*W21))

    def funcion_activacion_sigmoide(self, valor_suma_ponderada):
        return (1 / (1 + exp(-valor_suma_ponderada)))

    def funcion_activacion_relu(self, valor_suma_ponderada):
        return (max(0,valor_suma_ponderada))

    def error_lineal(self, valor_esperado, valor_predicho):
        return (valor_esperado-valor_predicho)

    def calculo_gradiente(self,valor_entrada,prediccion,error):
        return (-1 * error * prediccion * (1-prediccion) * valor_entrada)

    def calculo_valor_ajuste(self,valor_gradiente, tasa_aprendizaje):
        return (valor_gradiente*tasa_aprendizaje)

    def calculo_nuevo_peso (self,valor_peso, valor_ajuste):
        return (valor_peso - valor_ajuste)

    def calculo_MSE(self, predicciones_realizadas, predicciones_esperadas):
        i=0;
        suma=0;
        for prediccion in predicciones_esperadas:
            diferencia = predicciones_esperadas[i] - predicciones_realizadas[i]
            cuadradoDiferencia = diferencia * diferencia
            suma = suma + cuadradoDiferencia
        media_cuadratica = 1 / (len(predicciones_esperadas)) * suma
        return media_cuadratica

    def aprendizaje (self):
        sesgo = 1
        for epoch in range(0,self.epochs):
            print("EPOCH ("+str(epoch)+"/"+str(self.epochs)+")")
            predicciones_realizadas_durante_epoch = [];
            predicciones_esperadas = [];
            numObservacion = 0
            for observacion in self.observaciones_entrada:

                #Carga de la capa de entrada
                x1 = observacion[0];
                x2 = observacion[1];

                #Valor de predicción esperado
                valor_esperado = self.prediccion[numObservacion][0]

                #Etapa 1: Cálculo de la suma ponderada
                valor_suma_ponderada = percepton.suma_ponderada(x1,w11,x2,w21,sesgo,wb)


                #Etapa 2: Aplicación de la función de activación
                valor_predicho = percepton.funcion_activacion_sigmoide(valor_suma_ponderada)


                #Etapa 3: Cálculo del error
                valor_error = percepton.error_lineal(valor_esperado,valor_predicho)


                #Actualización del peso 1
                #Cálculo ddel gradiente del valor de ajuste y del peso nuevo
                gradiente_W11 = percepton.calculo_gradiente(x1,valor_predicho,valor_error)
                valor_ajuste_W11 = percepton.calculo_valor_ajuste(gradiente_W11,self.txAprendizaje)
                w11 = percepton.calculo_nuevo_peso(w11,valor_ajuste_W11)

                # Actualización del peso 2
                gradiente_W21 = percepton.calculo_gradiente(x2, valor_predicho, valor_error)
                valor_ajuste_W21 = percepton.calculo_valor_ajuste(gradiente_W21, self.txAprendizaje)
                w21 = percepton.calculo_nuevo_peso(w21, valor_ajuste_W21)


                # Actualización del peso del sesgo
                gradiente_Wb = percepton.calculo_gradiente(sesgo, valor_predicho, valor_error)
                valor_ajuste_Wb = percepton.calculo_valor_ajuste(gradiente_Wb, self.txAprendizaje)
                wb = percepton.calculo_nuevo_peso(wb, valor_ajuste_Wb)

                print("     EPOCH (" + str(epoch) + "/" + str(self.epochs) + ") -  Observación: " + str(numObservacion+1) + "/" + str(len(observaciones_entradas)))

                #Almacenamiento de la predicción realizada:
                predicciones_realizadas_durante_epoch.append(valor_predicho)
                predicciones_esperadas.append(self.prediccion[numObservacion][0])

                #Paso a la observación siguiente
                numObservacion = numObservacion+1

            MSE = percepton.calculo_MSE(predicciones_realizadas_durante_epoch, self.prediccion)
            self.Grafica_MSE.append(MSE[0])
            print("MSE: "+str(MSE))

        array = [w11, w21, wb]
        return array

    def prediccion(self, x1, w11, x2, w21, wb):
        sesgo = 1
        #Etapa 1: Cálculo de la suma ponderada
        valor_suma_ponderada = percepton.suma_ponderada(x1,w11,x2,w21,wb)
        valor_predicho = percepton.funcion_activacion_sigmoide(valor_suma_ponderada)

        print("Predicción del [" + str(x1) + "," + str(x2)  + "]")
        print("Predicción = " + str(valor_predicho))

    def plot (self):
        plt.plot(self.Grafica_MSE)
        plt.ylabel('MSE')
        plt.show()


def main():
    percept = percepton(([[1, 0], [1, 1], [0, 1], [0, 0]]), ([[0],[1], [0],[0]]))
    peso = percept.pes_inicial()
    print()
    print()
    print ("¡Aprendizaje terminado!")
    print ("Pesos iniciales: " )
    print ("W11 = "+str(peso[0]))
    print ("W21 = "+str(peso[1]))
    print ("Wb = "+str(peso[3]))

    array = percept.aprendizaje()
    print ("Pesos finales: " )
    print ("W11 = "+str(array[0]))
    print ("W21 = "+str(array[1]))
    print ("Wb = "+str(array[2]))

    print()
    print("--------------------------")
    print ("PREDICCIÓN ")
    print("--------------------------")
    x1 = 1
    x2 = 1
    percept.prediccion(x1, array[0], x2, array[1], array[2])


if __name__ == "__main__":
    main()

