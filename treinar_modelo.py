"""Treina classificador estático com separação por vídeo, não por frame."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential,Input
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
BASE=Path(__file__).resolve().parent

def main():
    arquivo=BASE/'dataset_libras.csv'
    if not arquivo.exists():raise FileNotFoundError('Execute extrair_dataset.py primeiro.')
    df=pd.read_csv(arquivo).dropna()
    cols=[f'{e}{i}' for i in range(21) for e in ('x','y','z')]
    faltantes=set(cols+['label','video'])-set(df.columns)
    if faltantes:raise ValueError(f'Colunas ausentes: {faltantes}')
    contagens=df.groupby('label')['video'].nunique()
    validos=contagens[contagens>=2].index
    df=df[df.label.isin(validos)].copy()
    if df.label.nunique()<2:raise ValueError('São necessários ao menos 2 sinais, cada um com pelo menos 2 vídeos diferentes (ideal: vários vídeos por sinal).')
    encoder=LabelEncoder();y=encoder.fit_transform(df.label)
    X=df[cols].to_numpy(dtype=np.float32)
    grupos=df['video'].astype(str).to_numpy()
    # Teste com grupos inteiros: evita vazamento de frames de um vídeo para outro conjunto.
    splitter=GroupShuffleSplit(n_splits=50,test_size=.2,random_state=42)
    divisao=None
    for tr,te in splitter.split(X,y,groups=grupos):
        if len(set(y[tr]))==len(encoder.classes_) and len(set(y[te]))==len(encoder.classes_):
            divisao=(tr,te);break
    if divisao is None:
        raise ValueError('Não foi possível obter treino/teste com todos os sinais sem misturar vídeos. Colete mais vídeos independentes por sinal.')
    tr,te=divisao
    model=Sequential([Input(shape=(63,)),Dense(128,activation='relu'),Dropout(.2),
       Dense(64,activation='relu'),Dropout(.2),Dense(32,activation='relu'),
       Dense(len(encoder.classes_),activation='softmax')])
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit(X[tr],y[tr],validation_data=(X[te],y[te]),epochs=60,batch_size=32,
       callbacks=[EarlyStopping(monitor='val_loss',patience=7,restore_best_weights=True)],verbose=1)
    perda,acuracia=model.evaluate(X[te],y[te],verbose=0)
    model.save(str(BASE/'modelo_libras_mediapipe.h5'))
    np.save(BASE/'classes_libras.npy',encoder.classes_)
    (BASE/'relatorio_treino.json').write_text(json.dumps({
       'classes':encoder.classes_.tolist(),'amostras_treino':int(len(tr)),
       'amostras_teste':int(len(te)),'acuracia_teste':float(acuracia),
       'observacao':'Métrica de quadros estáticos; não mede tradução de frases em Libras.'},indent=2,ensure_ascii=False),encoding='utf-8')
    print('Treinamento concluído. Acurácia no conjunto de teste:',round(float(acuracia),3))

if __name__=='__main__':main()
