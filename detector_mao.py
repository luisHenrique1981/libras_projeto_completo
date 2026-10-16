# --- Importar as bibliotecas --- #
import cv2
import mediapipe as mp
import time
import os
import numpy as np
from tensorflow.keras.models import load_model


class DetectorMaos:
    """Classe responsável pela detecção das mãos."""

    def __init__(
        self,
        max_maos=2,
        deteccao_confianca=0.5,
        rastreio_confianca=0.5,
        cor_pontos=(0, 0, 255),
        cor_conexoes=(255, 255, 255)
    ):

        # --- Inicializar os parâmetros --- #
        self.max_maos = max_maos
        self.deteccao_confianca = deteccao_confianca
        self.rastreio_confianca = rastreio_confianca
        self.cor_pontos = cor_pontos
        self.cor_conexoes = cor_conexoes

        self.resultado = None

        # --- Configurar o MediaPipe Hand Landmarker --- #
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        caminho_modelo = os.path.join(
            os.path.dirname(__file__),
            "hand_landmarker.task"
        )

        opcoes = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=caminho_modelo
            ),
            running_mode=RunningMode.VIDEO,
            num_hands=self.max_maos,
            min_hand_detection_confidence=self.deteccao_confianca,
            min_tracking_confidence=self.rastreio_confianca
        )

        self.maos = HandLandmarker.create_from_options(opcoes)

        # --- Tempo inicial para o timestamp do vídeo --- #
        self.inicio = time.time()

        # --- Conexões entre os pontos da mão --- #
        self.conexoes = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]

    def encontrar_maos(self, imagem, desenho=True):
        """Detecta as mãos e desenha os pontos."""

        # --- Converter a imagem de BGR para RGB --- #
        imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

        # --- Converter para formato do MediaPipe --- #
        imagem_mp = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=imagem_rgb
        )

        # --- Criar timestamp em milissegundos --- #
        timestamp = int((time.time() - self.inicio) * 1000)

        # --- Realizar a detecção --- #
        self.resultado = self.maos.detect_for_video(
            imagem_mp,
            timestamp
        )

        # --- Verificar se alguma mão foi detectada --- #
        if self.resultado.hand_landmarks:

            altura, largura, _ = imagem.shape

            for mao in self.resultado.hand_landmarks:

                pontos = []

                # --- Obter os pontos da mão --- #
                for ponto in mao:
                    x = int(ponto.x * largura)
                    y = int(ponto.y * altura)

                    pontos.append((x, y))

                    if desenho:
                        cv2.circle(
                            imagem,
                            (x, y),
                            5,
                            self.cor_pontos,
                            cv2.FILLED
                        )

                # --- Desenhar as conexões --- #
                if desenho:
                    for inicio, fim in self.conexoes:
                        cv2.line(
                            imagem,
                            pontos[inicio],
                            pontos[fim],
                            self.cor_conexoes,
                            2
                        )

        return imagem

    def encontrar_pontos(
        self,
        imagem,
        mao_num=0,
        ponto_detectado=0
    ):
        """Retorna a posição de um ponto específico da mão."""

        lista_pontos = []

        if self.resultado and self.resultado.hand_landmarks:

            if mao_num < len(self.resultado.hand_landmarks):

                mao = self.resultado.hand_landmarks[mao_num]

                altura, largura, _ = imagem.shape

                ponto = mao[ponto_detectado]

                centro_x = int(ponto.x * largura)
                centro_y = int(ponto.y * altura)

                lista_pontos.append(
                    [ponto_detectado, centro_x, centro_y]
                )

        return lista_pontos


def main():

    # --- Carregar modelo treinado e classes --- #
    pasta = os.path.dirname(__file__)
    caminho_rede = os.path.join(pasta, "modelo_libras_mediapipe.h5")
    caminho_classes = os.path.join(pasta, "classes_libras.npy")

    if not os.path.exists(caminho_rede) or not os.path.exists(caminho_classes):
        print("Modelo ou classes não encontrados. Execute o treinamento primeiro.")
        return

    modelo = load_model(caminho_rede, compile=False)
    classes = np.load(caminho_classes, allow_pickle=True)

    # --- Capturar o vídeo pela webcam --- #
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Não foi possível acessar a câmera. Tente trocar 0 por 1.")
        return

    # --- Instanciar o detector --- #
    detector = DetectorMaos()

    while True:

        # --- Obter a imagem --- #
        sucesso, imagem = cap.read()

        if not sucesso:
            print("Não foi possível acessar a câmera.")
            break

        # --- Inverter a imagem --- #
        imagem = cv2.flip(imagem, 1)

        # --- Detectar as mãos --- #
        imagem = detector.encontrar_maos(imagem)

        # --- Obter o ponto 0 da primeira mão --- #
        lista_pontos = detector.encontrar_pontos(imagem)

        if lista_pontos:
            print(lista_pontos)

        # --- Classificar os pontos detectados em tempo real --- #
        if detector.resultado and detector.resultado.hand_landmarks:
            mao = detector.resultado.hand_landmarks[0]
            base_x, base_y, base_z = mao[0].x, mao[0].y, mao[0].z
            dados = []
            for ponto in mao:
                dados.extend([
                    ponto.x - base_x,
                    ponto.y - base_y,
                    ponto.z - base_z
                ])

            probabilidades = modelo.predict(
                np.array([dados], dtype=np.float32), verbose=0
            )[0]
            indice = int(np.argmax(probabilidades))
            confianca = float(probabilidades[indice])
            if confianca >= 0.8:
                palavra = str(classes[indice])
                cv2.putText(
                    imagem, f"{palavra} ({confianca:.0%})", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )

        # --- Mostrar a imagem --- #
        cv2.imshow("Captura", imagem)

        # --- Pressionar Q para sair --- #
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()