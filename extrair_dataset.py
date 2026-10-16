"""Converte vídeos em pontos normalizados da mão; 1 mão x 21 pontos x XYZ = 63."""
import argparse,csv,json
from pathlib import Path
import cv2
import mediapipe as mp

BASE=Path(__file__).resolve().parent
COLUNAS=['label','video','frame']+[f'{e}{i}' for i in range(21) for e in ('x','y','z')]

def listar_videos():
    itens=[]
    lista=BASE/'videos.json'
    if lista.exists():
        for i in json.loads(lista.read_text(encoding='utf-8')):
            if i.get('caminho') and i.get('label'):
                arquivo=Path(i['caminho'])
                itens.append((arquivo if arquivo.is_absolute() else BASE/arquivo,str(i['label']).strip()))
    tabela=BASE/'libras_completo.csv'
    if tabela.exists():
        with tabela.open(encoding='utf-8-sig',newline='') as f:
            for linha in csv.DictReader(f,delimiter=';'):
                if linha.get('video') and linha.get('palavra'):
                    itens.append((BASE/'videos_libras'/Path(linha['video']).name,linha['palavra'].strip()))
    # Evita repetir o mesmo vídeo/classe em execuções.
    return list(dict.fromkeys((str(c.resolve()),l) for c,l in itens))

def processar_video(caminho_video,label,landmarker,writer,amostras_por_segundo=5):
    cap=cv2.VideoCapture(str(caminho_video))
    if not cap.isOpened(): print('Não abriu:',caminho_video);return 0
    fps=cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps<=0:fps=30.0
    passo=max(1,round(fps/amostras_por_segundo))
    frame_index=0;salvos=0;ultimo_ms=-1
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            pos=frame_index;frame_index+=1
            if pos%passo:continue
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            imagem=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
            timestamp=max(ultimo_ms+1,round(pos*1000/fps));ultimo_ms=timestamp
            resultado=landmarker.detect_for_video(imagem,timestamp)
            if not resultado.hand_landmarks:continue
            mao=resultado.hand_landmarks[0]
            origem=(mao[0].x,mao[0].y,mao[0].z)
            dados=[coordenada - origem[k] for ponto in mao for k,coordenada in enumerate((ponto.x,ponto.y,ponto.z))]
            writer.writerow([label,str(caminho_video),pos]+dados)
            salvos+=1
    finally:cap.release()
    return salvos

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--fps-amostras',type=float,default=5)
    a=ap.parse_args()
    if a.fps_amostras<=0:ap.error('fps-amostras precisa ser positivo')
    videos=listar_videos()
    if not videos:
        print('Sem vídeos: execute coletar_videos.py ou crie videos.json (veja README).');return
    modelo=BASE/'hand_landmarker.task'
    if not modelo.exists():raise FileNotFoundError(modelo)
    op=mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(modelo)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,num_hands=1)
    destino=BASE/'dataset_libras.csv'
    total=0
    # Regerar dataset inteiro evita duplicação de frames em execuções sucessivas.
    with destino.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(COLUNAS)
        for caminho,label in videos:
            if not Path(caminho).is_file():print('Não encontrado:',caminho);continue
            with mp.tasks.vision.HandLandmarker.create_from_options(op) as detector:
                n=processar_video(caminho,label,detector,writer,a.fps_amostras)
            total+=n;print(label,Path(caminho).name,n,'amostras')
    print('Dataset:',destino,'amostras:',total)

if __name__=='__main__':main()
