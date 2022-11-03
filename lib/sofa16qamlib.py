import numpy as np

class Sofa:
    def __init__(self):
        """ Constantes """
        # Posiciones en el diagrama de constelación
        self.A_MC = np.array([-3 - 3j, -3 - 1j, -3 + 3j, -3 + 1j,
                              -1 - 3j, -1 - 1j, -1 + 3j, -1 + 1j,
                               3 - 3j,  3 - 1j,  3 + 3j,  3 + 1j,
                               1 - 3j,  1 - 1j,  1 + 3j,  1 + 1j])

        # Diccionario para traducir índices decimales a símbolos
        self.DIC_INDEX_TO_SYMS = {index : sym for index, sym in enumerate(self.DIC_INDEX_TO_SYMS)}

        # Diccionario para traducir símbolos a índices decimales
        self.DIC_SYMS_TO_INDEX = dict(zip(self.DIC_INDEX_TO_SYMS.values(), self.DIC_INDEX_TO_SYMS.keys()))

        def modulate(self, inputs : list[int]) -> np.ndarray:
            """
            Realiza la modulación de los índices.

            :param inputs: Índices a modular (decimales).
            :returns: Índices modulados en formato 16QAM.
            """
            modulated = []
            for index in inputs:
                modulated.append(self.DIC_INDEX_TO_SYMS[index])

            return np.array(modulated)

        def demodulate(self, syms_rx : np.ndarray) -> list[int]:
            """
            Realiza la demodulación tradicional de los símbolos.

            :param syms_rx: Símbolos a demodular (complejos).
            :returns: Símbolos demodulados en formato 16QAM.
            """
            demodulated = []
            for symbol in syms_rx:
                # Distancia a cada centroide
                dist = np.abs(self.A_MC - symbol)
                # Índice del valor mínimo de distancia
                index = list(dist).index(np.min(dist))
                # Centroide más cercano al símbolo
                nearest_sym = self.A_MC[index]
                # Demodulación
                demodulated.append(self.DIC_SYMS_TO_INDEX[nearest_sym])

            return demodulated

        def linear_noise(self, syms_tx, snr):
            """
            Aplica ruido lineal simulando el canal físico.

            param syms_tx: Señal a distorsionar.
            param snr: Relación señal a ruido (0dB, ~40dB).
            returns: Señal distorsionada por ruido lineal.
            """
            # Cantidad de simbolos
            num_sym = len(syms_tx)
            # Copia del arreglo original
            sig_rx = syms_tx.copy()

            # Se calcula la potencia de la señal ideal
            signal_power = np.var(sig_rx)
            # Relación señal-a-ruido en decibeles
            Es_No = np.power(10, snr / 10.0)
            # Potencia del ruido
            N0 = signal_power / Es_No  
            # Distribución gaussiana siguiendo la potencia del ruido
            sigma = np.sqrt(N0 / 2)

            # Ruido aleatorio para la componente In-Phase
            ni = np.random.normal(scale = sigma, size = num_sym)
            # Ruido aleatorio para la componente Quadrature
            nq = np.random.normal(scale = sigma, size = num_sym) 

            # Ruido total
            n = ni + 1j * nq

            # Aplicación del ruido lineal a los símbolos originales
            sig_rx += n

            return sig_rx

        def non_linear_noise(self, syms_tx, phi, rot):
            """
            Aplica ruido no lineal simulando el canal físico.

            param syms_tx: Señales a distorsionar.
            param phi: Ruido de fase (0dB, ~10dB).
            param rot: Rotación a aplicar a la constelación en grados (0, 360).
            returns: Señal distorsionada por ruido no lineal.
            """
            # Cantidad de simbolos
            num_sym = len(syms_tx)
            # Copia del arreglo original
            sig_rx = syms_tx.copy()

            # Ruido de fase
            phase_n = phi * (np.pi / 180) * np.random.randn(1, num_sym)

            # Aplicación del ruido de fase a los símbolos afectados por ruido lineal
            sig_rx *= np.exp(1j * phase_n)

            # Cálculo de la rotación en radianes
            phn = (rot * np.pi) / 180

            # Aplicación de la rotación a los símbolos con ruido
            sig_rx *= np.exp(1j * phn)

            return sig_rx

        def bit_error_rate(self, rx, tx):
            """
            Halla la tasa de error de bit.

            param rx: Señal transmitida.
            param tx: Señal recibida.
            returns: Tasa de error de bit. 
            """
            rx_bit = []
            tx_bit = []
            for i in range(0, len(rx)):
                rx_bit.append(f'{rx[i]:04b}')
                tx_bit.append(f'{tx[i]:04b}')

            str_rx = ''.join(list(map(str, rx_bit)))
            str_tx = ''.join(list(map(str, tx_bit)))

            return sum(str_rx[i] != str_tx[i] for i in range(len(str_rx)))

        def modnorm(self, signalo):
            """ Retorna la señal normalizada """
            constPow = np.mean(np.abs(signalo) ** 2)
            scaleo = np.sqrt(10 / constPow)
            return scaleo
