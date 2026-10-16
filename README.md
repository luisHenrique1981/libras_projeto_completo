# Projeto Libras integrado (webcam + preparação + treino)

Conserva a classe `DetectorMaos` do código original; usa 63 coordenadas relativas ao pulso para classificação **estática**, não tradução completa de Libras.

## Ambiente (Windows / PowerShell)
Instale Python 3.11 (64-bit), depois na pasta do projeto:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m playwright install chromium
```

Não é obrigatório ativar o venv, o que evita bloqueio de execução de scripts no PowerShell.

## Etapas

1. **Coleta opcional:** ` .\.venv\Scripts\python.exe coletar_videos.py --limite-por-letra 3` para testar. Para todas as palavras omita o parâmetro. Produz `libras_completo.csv` e `videos_libras/`. **Atenção:** site externo pode mudar, bloquear automação ou impor restrições de uso; confira a licença e os dados extraídos antes de reutilizar. O CSV é índice do material, não um reconhecedor.
2. **Vídeos próprios (alternativa/recomendado):** coloque MP4 em `videos_libras`, copie `videos.exemplo.json` para `videos.json` e ajuste os caminhos/labels. Pode mesclar com os vídeos da coleta. Use **vários vídeos independentes**, idealmente diferentes pessoas, iluminação e ângulos, para cada sinal.
3. **Extração:** `.\.venv\Scripts\python.exe extrair_dataset.py` gera `dataset_libras.csv`. Recria o CSV e usa aproximadamente 5 quadros/segundo.
4. **Treinamento:** `.\.venv\Scripts\python.exe treinar_modelo.py` gera `modelo_libras_mediapipe.h5`, `classes_libras.npy`, `relatorio_treino.json`. Requer >=2 classes e >=2 vídeos independentes por classe, e mais dados podem ser necessários para separar treino e teste.
5. **Webcam:** `.\.venv\Scripts\python.exe detector_mao.py` (tecla Q encerra). Se a câmera não abrir, altere `cv2.VideoCapture(0)` para `1` no arquivo.

## Limitações importantes
- O classificador usa **somente uma mão em um instante**; sinais dinâmicos, expressões faciais, corpo, duas mãos e tradução de frases exigem um pipeline temporal mais avançado.
- A palavra de saída é o rótulo/glosa do sinal, **não a tradução gramatical automática para português**.
- A separação por vídeo evita vazamento de quadros do mesmo arquivo, mas vídeos semelhantes, duplicados ou do mesmo intérprete podem elevar artificialmente as métricas.
- O projeto foi verificado quanto à sintaxe; a coleta nos sites, o treinamento e a webcam precisam ser testados na sua máquina e com vídeos reais.
