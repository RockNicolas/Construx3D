# Construx3D

Editor 3D em tempo real controlado por gestos das maos usando Python, OpenCV e MediaPipe.

Ao iniciar, o app tambem abre uma janela de atividades em tempo real com eventos como selecao, criacao de blocos, movimento do conjunto, desfazer e exportacoes.

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

Edite `config/gesture_settings.json` para ajustar sensibilidade do pinch, tempos de hold/cooldown, zoom inicial, limites de zoom, distancia da camera e distancia maxima de selecao.

## Gestos

- O app trabalha com cubos 1x1x1.
- Mao rosa: o gesto sobre um bloco existente apenas seleciona a peca.
- Mao rosa: totalmente aberta move e gira todos os blocos.
- Mao rosa: fechada deixa tudo parado e limpa o bloco ativo.
- Mao azul: gesto tipo OK, sobre um bloco existente, arrasta e expande novos blocos grudados um a um.
- Mao rosa: polegar + minimo desfaz a ultima acao.
- O encaixe em grade pode ser ajustado em `config/gesture_settings.json` na secao `snap`.
