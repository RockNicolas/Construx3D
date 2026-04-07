# Construx3D

Editor 3D em tempo real controlado por gestos das maos usando Python, OpenCV e MediaPipe.

## Estrutura

- `src/construx3d/`: codigo da aplicacao
- `config/`: configuracao editavel de gestos e tracking
- `data/exports/`: exportacoes JSON e PNG geradas em runtime
- `main.py`: launcher simples para abrir a aplicacao sem instalar o pacote

## Execucao

```bash
python main.py
```

Na primeira execucao, o app pode baixar automaticamente o modelo `hand_landmarker.task` para `data/models/`.

Opcionalmente, como pacote:

```bash
pip install -e .
construx3d
```

## Calibracao

Edite `config/gesture_settings.json` para ajustar sensibilidade do pinch, tempos de hold/cooldown, zoom e distancia maxima de selecao.

## Gestos

- O app usa um unico bloco 5x1 horizontal, igual ao da referencia.
- Mao rosa: gesto de clicar com o indicador cria e fixa o bloco no grid imediatamente.
- Mao rosa: pinça sobre um bloco duplica a forma e arrasta a copia.
- Mao rosa fechada seleciona todos os blocos criados.
- Mao rosa: soltar a pinça fixa a forma na ultima posicao 3D encaixada na grade.
- Mao azul: gesto com indicador e medio levantados apaga o bloco quando o cursor passa por cima dele.
- Mao rosa: polegar + minimo desfaz a ultima acao.
- Duas maos abertas, como na pose frontal, rotacionam o modulo selecionado em 360 graus.
- Duas pinças ao mesmo tempo controlam o zoom pela distancia entre as maos.
- O encaixe em grade pode ser ajustado em `config/gesture_settings.json` na secao `snap`.
