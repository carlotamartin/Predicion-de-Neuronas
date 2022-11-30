import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt

class percepton():
    def __init__ (self, obs_entr, prediccion):
        self.valores_entradas_X = obs_entr
        self.valores_a_predecir_Y = prediccion
        self.tf_neuronas_entradas_X = tf.placeholder(tf.float32, [None, 2])
        self.tf_valores_reales_Y = tf.placeholder(tf.float32, [None, 1])
        self.peso = tf.Variable(tf.random_normal([2, 1]), tf.float32)
        self.sesgo = tf.Variable(tf.zeros([1, 1]), tf.float32)

    def sumaponderada(self):
        sumaponderada = tf.matmul(self.tf_neuronas_entradas_X,self.peso)
        sumaponderada = tf.add(sumaponderada,self.sesgo)
        prediccion = tf.sigmoid(sumaponderada)
        funcion_error = tf.reduce_sum(tf.pow(self.tf_valores_reales_Y-prediccion,2))
        return prediccion, funcion_error


    def optimizar(self, funcion_error):
        optimizador = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(funcion_error)
        return optimizador

    def aprendizaje(self, epochs, funcion_error, optimizador, prediccion):
        init = tf.global_variables_initializer()
        #Inicio de una sesión de aprendizaje
        sesion = tf.Session()
        sesion.run(init)

        #Para la realización de la gráfica para la MSE
        Grafica_MSE=[]

        for i in range(epochs):

            #Realización del aprendizaje con actualzación de los pesos
            sesion.run(optimizador, feed_dict = {self.tf_neuronas_entradas_X: self.valores_entradas_X, self.tf_valores_reales_Y:self.valores_a_predecir_Y})

            #Calcular el error
            MSE = sesion.run(funcion_error, feed_dict = {self.tf_neuronas_entradas_X: self.valores_entradas_X, self.tf_valores_reales_Y:self.valores_a_predecir_Y})

            #Visualización de la información
            Grafica_MSE.append(MSE)
            print("EPOCH (" + str(i) + "/" + str(epochs) + ") -  MSE: "+ str(MSE))


        print("--- VERIFICACIONES ----")
        for i in range(0,4):
            print("Observación:"+str(self.valores_entradas_X[i])+ " - Esperado: "+str(self.valores_a_predecir_Y[i])+" - Predicción: "+str(sesion.run(prediccion, feed_dict={self.tf_neuronas_entradas_X: [self.valores_entradas_X[i]]})))

        return Grafica_MSE

    def plot(self, grafica):
        plt.plot(grafica)
        plt.ylabel('MSE')
        plt.show()



def main():
    percept = percepton([[1., 0.], [1., 1.], [0., 1.], [0., 0.]],  [[0.], [1.], [0.], [0.]])
    prediccion, funcion_error = percept.sumaponderada()
    optimizador = percept.optimizar(funcion_error)
    graf = percept.aprendizaje(3000, funcion_error, optimizador, prediccion)
    percept.plot(graf)


if __name__ == "__main__":
    main()
