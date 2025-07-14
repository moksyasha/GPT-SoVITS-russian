import torch
import torch.nn as nn
import os
import utils  # Предполагается, что у вас есть этот модуль

# Импортируем необходимые классы для WavLM
# Убедитесь, что папка с кодом WavLM доступна
from feature_extractor.WavLM import WavLM, WavLMConfig

# Отключаем лишние логи, как в оригинале
from transformers import logging as tf_logging
tf_logging.set_verbosity_error()
import logging
logging.getLogger("numba").setLevel(logging.WARNING)


# Эта переменная теперь будет хранить путь к .pt файлу WavLM
# Вы можете установить ее глобально или передавать в конструктор
cnhubert_base_path = None


class CNHubert(nn.Module):
    def __init__(self, base_path: str = None):
        super().__init__()
        if base_path is None:
            base_path = cnhubert_base_path
        if os.path.exists(base_path):
            ...
        else:
            raise FileNotFoundError(base_path)
        
        print(f"LOAD WAVLM {base_path}")
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Файл модели WavLM не найден: {base_path}")

        print(f"Загрузка модели WavLM из: {base_path}")
        try:
            checkpoint = torch.load(base_path + "/WavLM-Base-plus.pt", map_location="cpu")
            self.cfg = WavLMConfig(checkpoint['cfg'])
            self.model = WavLM(self.cfg)
            self.model.load_state_dict(checkpoint['model'])
            print("Модель WavLM успешно загружена.")
        except Exception as e:
            print(f"Ошибка при загрузке модели WavLM: {e}")
            raise e
            
        # У WavLM нет отдельного feature_extractor, как у моделей HuggingFace.
        # Нормализация - это часть логики модели, которую мы вызовем в forward.
        # Мы создаем этот атрибут для возможной совместимости, но он не будет использоваться.
        self.feature_extractor = None

    def forward(self, x):
        """
        Принимает на вход сырой тензор аудиоволны.
        x: torch.Tensor, форма [batch_size, num_samples]
        """
        # WavLM ожидает, что аудио уже имеет частоту 16000 Гц.
        # Убедимся, что входной тензор имеет 2 измерения [batch, samples]
        # if x.dim() == 1:
        #     x = x.unsqueeze(0)
        
        # Применяем LayerNorm, если это требуется конфигурацией модели
        if self.cfg.normalize:
            x = torch.nn.functional.layer_norm(x, x.shape)
            
        # Извлекаем признаки. model.extract_features возвращает кортеж.
        # Нам нужен первый элемент - тензор признаков.
        # Его форма: [batch, time, features]
        feats = self.model.extract_features(x)[0]
        
        # Старый CNHubert возвращал [batch, time, features], так что транспонировать не нужно.
        # Если ваш старый код ожидал [batch, features, time], то нужно добавить .transpose(1, 2)
        # Судя по вашему коду `get_content`, он ожидает [B, T, F], а потом сам делает transpose.
        # Поэтому здесь мы возвращаем как есть.
        return feats


def get_model(path_to_model=None):
    """
    Фабричная функция для создания и подготовки модели.
    """
    # Если глобальная переменная не установлена, можно передать путь напрямую
    if path_to_model is not None:
        global cnhubert_base_path
        cnhubert_base_path = path_to_model
        
    # Создаем экземпляр нашего нового класса-обертки
    model = CNHubert()
    model.eval()
    return model


def get_content(hmodel, wav_16k_tensor):
    """
    Извлекает контент (признаки) из аудио.
    Эта функция остается почти без изменений, так как интерфейс модели сохранен.
    """
    with torch.no_grad():
        # hmodel.forward(wav_16k_tensor) вернет [B, T, 768]
        feats = hmodel(wav_16k_tensor)
    
    # Транспонируем в [B, 768, T], как это делалось в оригинальном коде
    return feats.transpose(1, 2)


if __name__ == "__main__":

    wavlm_model_path = "/home/mlb/Documents/PycharmProject/ai-collect/develop/GPT-SoVITS/GPT_SoVITS/pretrained_models/wavlm-plus/WavLM-Base-plus.pt" 
    
    # 2. Укажите путь к тестовому аудио
    src_path = "/home/mlb/Documents/PycharmProject/ai-collect/develop/GPT-SoVITS/GPT_SoVITS/feature_extractor/put.wav" 

    if not os.path.exists(wavlm_model_path) or not os.path.exists(src_path):
        print("Пожалуйста, укажите правильные пути к модели WavLM и тестовому аудиофайлу.")
    else:
        # 3. Создаем модель
        # Передаем путь напрямую в get_model
        model = get_model(path_to_model=wavlm_model_path)
        
        # 4. Загружаем и подготавливаем аудио
        # Функция из вашего проекта, которая загружает wav и ресемплирует в 16к
        wav_16k_tensor = torch.randn(1,10000)
        
        # 5. Перемещаем на GPU, если доступно
        if torch.cuda.is_available():
            model = model.cuda()
            wav_16k_tensor = wav_16k_tensor.cuda()
        
        # 6. Получаем признаки
        feats = get_content(model, wav_16k_tensor)
        
        print("Признаки успешно извлечены!")
        print("Форма выходного тензора:", feats.shape) # Ожидаемая форма: [1, 768, time_steps]