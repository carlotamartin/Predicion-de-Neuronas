from numpy import exp, array, random
import matplotlib.pyplot as plt


class percepton():
    def __init__ (self, obs_entr, prediccion, epochs):
        self.observaciones_entrada = obs_entr
        self.prediccion = prediccion
        self.epochs = epochs



    def pes_inicial (self):
        limiteMin = -1
        limiteMax = 1
        w11 = (limiteMax-limiteMin) * random.random() + limiteMin
        w21 = (limiteMax-limiteMin) * random.random() + limiteMin
        w31 = (limiteMax-limiteMin) * random.random() + limiteMin
        wb = 0
        peso = [w11, w21, w31, wb]
        print()
        print()
        print ("¬°Aprendizaje terminado!")
        print ("Pesos iniciales: " )
        print ("W11 = "+str(peso[0]))
        print ("W21 = "+str(peso[1]))
        print ("Wb = "+str(peso[3]))
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

    def aprendizaje (self, w11, w21, wb):
        Grafica_MSE = []
        sesgo = 1
        txAprendizaje = 0.1
        for epoch in range(0,self.epochs):
            print("EPOCH ("+str(epoch)+"/"+str(self.epochs)+")")
            predicciones_realizadas_durante_epoch = [];
            predicciones_esperadas = [];
            numObservacion = 0
            for observacion in self.observaciones_entrada:

                #Carga de la capa de entrada
                x1 = observacion[0];
                x2 = observacion[1];

                #Valor de predicci√≥n esperado
                valor_esperado = self.prediccion[numObservacion][0]

                #Etapa 1: C√°lculo de la suma ponderada
                valor_suma_ponderada = percepton.suma_ponderada(self, x1,w11,x2,w21,sesgo,wb)


                #Etapa 2: Aplicaci√≥n de la funci√≥n de activaci√≥n
                valor_predicho = percepton.funcion_activacion_sigmoide(self, valor_suma_ponderada)


                #Etapa 3: C√°lculo del error
                valor_error = percepton.error_lineal(self,valor_esperado,valor_predicho)


                #Actualizaci√≥n del peso 1
                #C√°lculo ddel gradiente del valor de ajuste y del peso nuevo
                gradiente_W11 = percepton.calculo_gradiente(self,x1,valor_predicho,valor_error)
                valor_ajuste_W11 = percepton.calculo_valor_ajuste(self,gradiente_W11,txAprendizaje)
                w11 = percepton.calculo_nuevo_peso(self,w11,valor_ajuste_W11)

                # Actualizaci√≥n del peso 2
                gradiente_W21 = percepton.calculo_gradiente(self,x2, valor_predicho, valor_error)
                valor_ajuste_W21 = percepton.calculo_valor_ajuste(self,gradiente_W21, txAprendizaje)
                w21 = percepton.calculo_nuevo_peso(self,w21, valor_ajuste_W21)


                # Actualizaci√≥n del peso del sesgo
                gradiente_Wb = percepton.calculo_gradiente(self,sesgo, valor_predicho, valor_error)
                valor_ajuste_Wb = percepton.calculo_valor_ajuste(self,gradiente_Wb, txAprendizaje)
                wb = percepton.calculo_nuevo_peso(self,wb, valor_ajuste_Wb)

                print("     EPOCH (" + str(epoch) + "/" + str(self.epochs) + ") -  Observaci√≥n: " + str(numObservacion+1) + "/" + str(len(self.observaciones_entrada)))

                #Almacenamiento de la predicci√≥n realizada:
                predicciones_realizadas_durante_epoch.append(valor_predicho)
                predicciones_esperadas.append(self.prediccion[numObservacion][0])

                #Paso a la observaci√≥n siguiente
                numObservacion = numObservacion+1

            MSE = percepton.calculo_MSE(self,predicciones_realizadas_durante_epoch, self.prediccion)
            Grafica_MSE.append(MSE[0])
            print("MSE: "+str(MSE))

        array = [w11, w21, wb]
        print ("Pesos finales: " )
        print ("W11 = "+str(w11))
        print ("W21 = "+str(w21))
        print ("Wb = "+str(wb))

        print()
        print("--------------------------")
        print ("PREDICCI√ďN ")
        print("--------------------------")
        return array, Grafica_MSE

    def prediccion(self, x1, w11, x2, w21, wb):
        sesgo = 1
        #Etapa 1: C√°lculo de la suma ponderada
        valor_suma_ponderada = percepton.suma_ponderada(self,x1,w11,x2,w21,sesgo,wb)
        valor_predicho = percepton.funcion_activacion_sigmoide(self,valor_suma_ponderada)

        print("Predicci√≥n del [" + str(x1) + "," + str(x2)  + "]")
        print("Predicci√≥n = " + str(valor_predicho))

    def plot (self, Grafica_MSE):
        plt.plot(Grafica_MSE)
        plt.ylabel('MSE')
        plt.show()


def main():
    percept = percepton(([[1, 0], [1, 1], [0, 1], [0, 0]]), ([[0],[1], [0],[0]]), 3000)
    peso = percept.pes_inicial()
    Grafica_MSE = percept.aprendizaje(peso[0], peso[1], peso[3])
    x1 = 1
    x2 = 1
    #percept.prediccion( x1, peso[0], x2, peso[1], peso[3])
    #percept.plot(Grafica_MSE)


if __name__ == "__main__":
    main()

