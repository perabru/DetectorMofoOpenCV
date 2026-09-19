# Detector visual de áreas suspeitas de mofo

Aplicação em Python com OpenCV e interface Tkinter. Permite analisar imagens de:

- webcam integrada;
- câmera USB;
- câmera Wi-Fi/IP por URL HTTP, HTTPS, RTSP ou RTMP.

## Importante

O programa combina cor e textura para **indicar manchas suspeitas**. Ele pode confundir mofo com sombra, sujeira, madeira, tinta ou umidade e não comprova a presença de fungos. Use o resultado para orientar uma inspeção visual. Em áreas extensas ou quando houver sintomas respiratórios, procure avaliação profissional.

## Instalação no Windows

1. Instale o Python e marque a opção `Add Python to PATH`.
2. Abra o terminal dentro da pasta do projeto.
3. Crie um ambiente virtual:

```powershell
py -m venv .venv
.venv\Scripts\activate
```

4. Instale as bibliotecas:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

5. Execute:

```powershell
python app.py
```

O Tkinter normalmente já acompanha o instalador oficial do Python para Windows.

## Como escolher a câmera

### Webcam integrada ou câmera USB

Selecione `Webcam / câmera USB`, clique em **Procurar** e escolha o índice encontrado. Geralmente:

- `0`: webcam integrada;
- `1`: primeira câmera USB;
- `2`: segunda câmera USB.

A numeração pode mudar conforme a ordem em que os dispositivos são conectados.

### Câmera Wi-Fi/IP

Selecione `Câmera Wi-Fi / IP` e informe o endereço de vídeo fornecido pela câmera ou pelo aplicativo. Exemplos:

```text
http://192.168.1.20:8080/video
rtsp://usuario:senha@192.168.1.50:554/stream1
```

O computador e a câmera normalmente precisam estar na mesma rede. O endereço exato depende do fabricante. Evite expor a câmera diretamente à internet.

## Ajustes

- **Sensibilidade:** aumente se manchas não estiverem sendo marcadas; diminua se houver muitos falsos alertas.
- **Área mínima:** aumente para ignorar pontos pequenos e ruído da imagem.
- **Destacar máscara suspeita:** pinta de vermelho os pixels usados pela análise.
- **Salvar imagem analisada:** grava a imagem com as marcações na pasta `capturas`.

Para obter resultados melhores, ilumine bem a parede, mantenha a câmera parada e evite reflexos ou sombras fortes.

## Evolução para inteligência artificial

Para reconhecer mofo com mais precisão, substitua a heurística por um modelo treinado com fotografias reais do ambiente, usando classes como `mofo` e `sem_mofo`. As caixas desenhadas pela interface podem ser mantidas, enquanto a função `MoldDetector.analyze()` passa a usar um modelo YOLO ou de segmentação.
