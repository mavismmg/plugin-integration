import os
import sys
import numpy as np
import json
from osgeo import gdal
from PIL import Image
import onnxruntime as ort

# Ensure we can be imported from any context
try:
    # Add the coreplugins directory to sys.path if not present
    coreplugins_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if coreplugins_path not in sys.path:
        sys.path.append(coreplugins_path)
        
    # Also add the WebODM root directory if possible
    webodm_path = os.path.dirname(coreplugins_path)
    if webodm_path not in sys.path:
        sys.path.append(webodm_path)
except Exception as e:
    print(f"Warning: Could not update sys.path: {str(e)}")

# Set globals to ensure consistent behavior regardless of how we're imported
__name__ = 'coreplugins.objdetect.detect_coffee' if '__name__' in globals() and __name__ == '__main__' else __name__
__package__ = 'coreplugins.objdetect' if '__package__' in globals() and __package__ == '' else __package__

# registrar UI de gerenciamento de modelos
try:
    from .views import plugin_bp
    from app import webodm_app
    webodm_app.register_blueprint(plugin_bp, url_prefix='/coreplugins')
except Exception:
    pass

def detect_coffee_plants(orthophoto_path):
    """
    Detector especializado para plantas de café usando modelo ONNX pré-treinado.
    
    Args:
        orthophoto_path: Caminho para o arquivo de ortofoto GeoTIFF
        
    Returns:
        GeoJSON contendo as detecções de plantas de café
    """
    print(f"Detecting coffee plants in: {orthophoto_path}")
    
    # Ensure models directory exists
    models_dir = None
    try:
        # Find correct path for models directory
        if '__file__' in globals():
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        else:
            # Try common locations
            possible_dirs = [
                "/webodm/coreplugins/objdetect/models",
                "/home/naki-linux/green-growth-plugin/WebODM/coreplugins/objdetect/models",
                os.path.join(os.getcwd(), "coreplugins", "objdetect", "models"),
            ]
            for dir_path in possible_dirs:
                if os.path.exists(dir_path):
                    models_dir = dir_path
                    break

        # Create models dir if needed
        if models_dir and not os.path.exists(models_dir):
            os.makedirs(models_dir)

        # Use the correct model filename
        model_path = os.path.join(models_dir, "modelo_segmentacao_linhas.onnx") if models_dir else "/webodm/coreplugins/objdetect/models/modelo_segmentacao_linhas.onnx"

    except Exception as e:
        print(f"Error setting up models directory: {str(e)}")
        model_path = "/webodm/coreplugins/objdetect/models/modelo_segmentacao_linhas.onnx"

    if not os.path.exists(model_path):
        error_msg = f"Modelo ONNX para café não encontrado em {model_path}. " \
                   f"Por favor, coloque o arquivo modelo_segmentacao_linhas.onnx neste diretório."
        print(f"Erro: {error_msg}")
        return {"type": "FeatureCollection", "features": [], "error": error_msg}
    
    # Carregar o modelo ONNX
    try:
        # Criar uma sessão de inferência ONNX
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        print(f"Modelo ONNX carregado com sucesso. Nome da entrada: {input_name}")
    except Exception as e:
        print(f"Erro ao carregar o modelo ONNX: {str(e)}")
        return {"type": "FeatureCollection", "features": [], "error": f"Erro ao carregar o modelo: {str(e)}"}
    
    # Abrir o arquivo GeoTIFF para processamento
    ds = gdal.Open(orthophoto_path)
    if not ds:
        print("Erro: Não foi possível abrir o arquivo GeoTIFF.")
        return {"type": "FeatureCollection", "features": [], "error": "Não foi possível abrir o arquivo GeoTIFF"}
    
    # Obter informações de georreferenciamento
    geo_transform = ds.GetGeoTransform()
    width = ds.RasterXSize
    height = ds.RasterYSize

    # Processar a imagem para detecção
    try:
        bands_data = []
        gdal_failed = False
        try:
            # Usar GDAL para leitura da imagem em vez de PIL
            # Ler os dados da imagem diretamente do dataset GDAL já aberto
            for i in range(1, min(ds.RasterCount + 1, 4)):  # Ler até 3 bandas (RGB)
                band = ds.GetRasterBand(i)
                data = band.ReadAsArray()
                bands_data.append(data)
        except Exception as e:
            print(f"Warning: GDAL ReadAsArray failed ({e}), falling back to PIL.Image.")
            gdal_failed = True

        if gdal_failed or not bands_data:
            # Fallback: carrega a imagem utilizando PIL quando GDAL falha
            # Esta é uma alternativa para garantir que sempre conseguiremos ler a imagem
            img_path = orthophoto_path
            img = Image.open(img_path)  # Abre o arquivo de imagem
            img = img.convert("RGB")    # Garante que a imagem esteja no formato RGB (3 canais)
            img_data = np.array(img)    # Converte para array NumPy para processamento matemático
            height, width = img_data.shape[:2]  # Extrai dimensões da imagem (altura e largura)
        else:
            # Processa os dados de bandas obtidos via GDAL
            if len(bands_data) == 3:
                # Caso ideal: temos exatamente 3 bandas (R, G, B)
                # Combina as bandas em um único array tridimensional usando dstack
                img_data = np.dstack((bands_data[0], bands_data[1], bands_data[2]))
            elif len(bands_data) == 1:  # Imagem em escala de cinza
                # Se temos apenas uma banda, usamos diretamente
                img_data = bands_data[0]
            else:
                # Caso tenhamos um número diferente de bandas (nem 1, nem 3)
                # Criamos um array RGB vazio e preenchemos com as bandas disponíveis
                img_data = np.zeros((height, width, 3), dtype=np.uint8)
                for i in range(len(bands_data)):
                    img_data[:, :, i] = bands_data[i]  # Preenche cada canal disponível

        # Preparação da imagem para inferência do modelo
        # ---------------------------------------------
        # Obtém informações sobre o formato de entrada esperado pelo modelo ONNX
        inp = session.get_inputs()[0]  # Recupera definição da primeira entrada do modelo
        shp = inp.shape  # Obtém as dimensões esperadas
        
        # Trata valores dinâmicos (None) no formato da entrada
        # Substitui valores None por dimensões padrão [1, 256, 256, 3] (batch, altura, largura, canais)
        _, H, W, C = [
            dim if isinstance(dim, int) else default
            for dim, default in zip(shp, [1, 256, 256, 3])
        ]
        
        # Redimensiona e normaliza a imagem para o formato esperado pelo modelo
        img_pil = Image.fromarray(img_data.astype(np.uint8))  # Reconverte para formato PIL
        img_resized = img_pil.resize((W, H))  # Redimensiona para as dimensões esperadas pelo modelo
        arr = np.array(img_resized).astype(np.float32) / 255.0  # Normaliza pixels para intervalo [0,1]
        
        # Adiciona dimensão de batch se necessário (formato NHWC - [batch, altura, largura, canais])
        if arr.ndim == 3 and arr.shape[2] == C:
            img_data = np.expand_dims(arr, axis=0)  # Adiciona dimensão de batch no início
        else:
            # Levanta erro se o formato após redimensionamento for inconsistente com o esperado
            raise ValueError(f"Formato inesperado após resize: {arr.shape}, esperado (*,{H},{W},{C})")
        
        print(f"Shape da imagem para inferência: {img_data.shape}")
        
        # Execução da inferência com o modelo ONNX
        # ---------------------------------------
        # Passa a imagem processada para o modelo e obtém resultados
        outputs = session.run(None, {input_name: img_data})
        
        # Logs para depuração dos resultados do modelo
        print(f"Saída bruta do modelo: {type(outputs)}")
        if isinstance(outputs, list):
            print(f"Tamanho outputs[0]: {len(outputs[0]) if len(outputs) > 0 else 'N/A'}")
        else:
            print("outputs não é uma lista!")

        detections = []  # Lista para armazenar as detecções encontradas
        
        # Interpretação das detecções do modelo
        # ------------------------------------
        if len(outputs) >= 1:
            # A estrutura exata de saída varia conforme o modelo ONNX
            # modelo retorna: [x, y, w, h, conf, class1, class2, ...]
            detection_results = outputs[0]
            # Achatar array 3D → 2D se necessário
            if isinstance(detection_results, np.ndarray) and detection_results.ndim > 2:
                detection_results = detection_results.reshape(-1, detection_results.shape[-1])
            # Converter para lista de listas
            if isinstance(detection_results, np.ndarray):
                detection_results = detection_results.tolist()
            # Se vier como [[det0, det1, ...]], remover nível extra
            if isinstance(detection_results, list) and len(detection_results) == 1 and isinstance(detection_results[0], list):
                detection_results = detection_results[0]
            print(f"Tipo de detection_results: {type(detection_results)}, count: {len(detection_results)}")
            for detection in detection_results:
                # pular detecções inválidas
                if len(detection) < 6:
                    continue
                # Agora detection já é uma lista de floats
                x1, y1, x2, y2, confidence = map(float, detection[1:6])
                # Converter coordenadas normalizadas para pixels se necessário
                if x1 <= 1.0 and y1 <= 1.0:
                    x1, y1, x2, y2 = x1 * width, y1 * height, x2 * width, y2 * height
                
                # Calcular centro do objeto
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Adicionar à lista de detecções
                detections.append((center_x, center_y, confidence))
            print(f"Detecções encontradas: {len(detections)}")
        else:
            print("outputs não possui elementos suficientes.")

        # Processar as detecções e convertê-las para GeoJSON
        features = []
        
        for i, (x_pixel, y_pixel, confidence) in enumerate(detections):
            # Converter pixel para coordenadas geográficas
            x_geo = geo_transform[0] + x_pixel * geo_transform[1]
            y_geo = geo_transform[3] + y_pixel * geo_transform[5]
            
            # Criar um polígono para representar a detecção
            radius = 2.0  # metros (ajuste conforme necessário)
            circle_points = []
            num_points = 10
            
            for angle in range(0, 360, 360 // num_points):
                rad = angle * np.pi / 180.0
                x = x_geo + radius * np.cos(rad)
                y = y_geo + radius * np.sin(rad)
                circle_points.append([x, y])
                
            # Fechar o polígono
            circle_points.append(circle_points[0])
            
            # Adicionar à lista de features
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [circle_points]
                },
                "properties": {
                    "class": "coffee_plant",
                    "score": float(confidence),
                    "id": i + 1
                }
            })
        
        print(f"Features GeoJSON geradas: {len(features)}")
        return {
            "type": "FeatureCollection",
            "features": features
        }
    
    except Exception as e:
        print(f"Erro durante a detecção: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"type": "FeatureCollection", "features": [], "error": str(e)}
