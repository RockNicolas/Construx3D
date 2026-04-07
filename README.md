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
