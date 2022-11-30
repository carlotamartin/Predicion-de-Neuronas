from perceptron import percepton, percepton_TF

def mostrar_menu(opciones):
    print('Seleccione una opción:')
    for clave in sorted(opciones):
        print(f' {clave}) {opciones[clave][0]}')


def leer_opcion(opciones):
    while (a := input('Opción: ')) not in opciones:
        print('Opción incorrecta, vuelva a intentarlo.')
    return a


def ejecutar_opcion(opcion, opciones):
    opciones[opcion][1]()


def generar_menu(opciones, opcion_salida):
    opcion = None
    while opcion != opcion_salida:
        mostrar_menu(opciones)
        opcion = leer_opcion(opciones)
        ejecutar_opcion(opcion, opciones)
        print()


def menu_principal():
    print('¿Cuántas capas quieres que tenga la red neuronal?')
    opciones = {
        '1': ('Percepton', accion1),
        '2': ('Percepton con Ténsor Flow', accion2),
        '6': ('Salir', salir)
    }
    generar_menu(opciones, '5')


def accion1():
    print('Has elegido la opción de ejecutar Percepton')
    percepton.main()


def accion2():
    print('Has elegido la opción de ejecutar Percepton con Ténsor Flow')
    percepton_TF.main()


def salir():
    print('Saliendo')


if __name__ == '__main__':
    menu_principal()