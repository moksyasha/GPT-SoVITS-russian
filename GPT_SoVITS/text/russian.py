# GPT_SoVITS/text/russian.py
import os
from ruphon import RUPhon
from ruaccent import RUAccent
import torch
import re
from num2words import num2words

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    accentizer = RUAccent()
    accentizer.load(omograph_model_size='turbo3', use_dictionary=True)

    phonemizer = RUPhon()
    path_model = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
    print(f"Initializing Russian G2P models (RUAccent, RUPhon)... path {path_model}")
    phonemizer = phonemizer.load("big", workdir="./models", device=device)
    
    print(f"Russian G2P models loaded successfully on device: {device}")
    models_loaded = True
except Exception as e:
    print(f"Error loading Russian G2P models: {e}")
    models_loaded = False

def text_normalize(text: str) -> str:
    """
    Улучшенная нормализация текста для TTS:
    1. Приводим к нижнему регистру.
    2. Заменяем 'ё' на 'е' для консистентности.
    3. Заменяем сложную пунктуацию на базовую (точки, запятые).
    4. Удаляем дефисы, кавычки и прочие типографские символы.
    5. Конвертируем числа в слова.
    6. Оставляем только разрешенные символы (кириллица, пробелы, запятые, точки).
    7. Нормализуем пробелы.
    """
    # Шаг 1: Приводим к нижнему регистру
    text = text.lower()
    text = text.replace('..', '.')
    text = text.replace('...', '.')
    text = text.replace('?!', '?')
    text = text.replace('!?', '?')
    text = text.replace('?.', '?')
    text = text.replace('.?', '?')
    if text[-1] not in [".", "?", "!"]:
        text += "."
    # Шаг 2: Заменяем 'ё' на 'е' для единообразия.
    # Модель не должна гадать, какую букву использовать.
    text = text.replace('ё', 'е')

    # Шаг 3: Замена сложной и визуальной пунктуации.
    # Это самый важный шаг для устранения пропусков.
    
    # Удаляем все виды кавычек: « » “ ” „ “ "
    text = re.sub(r'[«»“”„“"]', '', text)
    
    # Все виды тире и дефисов приводим к одному стандартному дефису, чтобы обработать их на следующем шаге.
    text = re.sub(r'[–—‑]', '-', text)
    
    # Точку с запятой (;) заменяем на запятую, так как они обычно означают схожую по длине паузу.
    text = text.replace(';', ',')
    
    # Вопросительные и восклицательные знаки заменяем на точку.
    # Это сохраняет конец предложения, но убирает сложную интонацию, которую модель может не воспроизвести.
    # text = re.sub(r'[?!]', '.', text)

    # Шаг 4: Обработка дефисов.
    # Самый безопасный и надежный способ для TTS - просто удалить их.
    # Это правильно обработает и "то-то" -> "тото", и "посылай-ка" -> "посылайка".
    text = text.replace('-', '')

    # Шаг 5: Конвертация чисел в слова (ваш код, немного улучшен для надежности)
    def number_replacer(match):
        number_str = match.group(0)
        # Просто удаляем все запятые, чтобы обработать и "1,000,000", и "1,5"
        cleaned_number_str = number_str.replace(',', '')
        try:
            # Сначала пытаемся как float, чтобы обработать "123.45"
            return num2words(float(cleaned_number_str), lang='ru')
        except ValueError:
            # Если не float, то как int
            try:
                return num2words(int(cleaned_number_str), lang='ru')
            except ValueError:
                # Если и это не удалось, возвращаем как было
                return match.group(0)

    text = re.sub(r'\d+[.,\d]*', number_replacer, text)

    # Шаг 6: Финальная очистка.
    # Оставляем только кириллицу (с 'е'), латиницу (на всякий случай), пробелы, запятые и точки.
    # Все остальное, что могло просочиться, будет удалено.
    text = re.sub(r"[^а-яеa-z\s,.]", "", text)

    # Шаг 7: Нормализация пробелов
    text = re.sub(r"\s+", " ", text).strip()

    return text

def force_stress_on_yo(text: str) -> str:
    """
    Принудительно ставит ударение на букву 'ё', если его еще нет.
    """
    words = text.split(' ')
    stressed_words = []
    for word in words:
        if 'ё' in word and '+' not in word:
            word = word.replace('ё', '+ё', 1)
        stressed_words.append(word)
    return ' '.join(stressed_words)

def g2p(norm_text: str) -> list:
    """
    Основная функция G2P (Grapheme-to-Phoneme).
    Принимает нормализованный текст и возвращает 
    лист фонем, разбиение слов с промежутками, и колво фонем на слово, маску с порядком слов.
    """
    if not models_loaded or not norm_text:
        return [], [], [], ""

    try:
        accented_text = accentizer.process_all(norm_text)
        accented_text = force_stress_on_yo(accented_text)
        if not accented_text:
            return [], [], [], ""
    except Exception as e:
        print(f"RUAccent error on text '{norm_text}': {e}")
        return [], [], [], ""

    try:
        phonemes_list, words_list, word2ph, string_mask = phonemizer.phonemize_list(accented_text, put_stress=True, stress_symbol="'")
    except Exception as e:
        print(f"RUPhon error on text '{accented_text}': {e}")
        return [], [], [], ""

    return phonemes_list, words_list, word2ph, string_mask


# Original: у неё, ежик ёжик отошли воды!И она т.п. попросила воды. а потом fabric's и все конец.фыв
# Normalized: у неё, ежик ёжик отошли воды!и она т.п. попросила воды. а потом fabric's и все конец.фыв
# Accentizer output (first guess): у неё, +ёжик +ёжик отошл+и вод+ы!и он+а т.п. попрос+ила вод+ы. а пот+ом fabric's и вс+ё кон+ец.фыв
# Phonemes: ['u', 'nʲ', 'ɪ', "'jɵ", ',', "'jɵ", 'ʐ', 'ɨ', 'k', "'jɵ", 'ʐ', 'ɨ', 'k', 'ɐ', 't', 'ɐ', 'ʂ', 'lʲ', "'i", 'v', 'ɐ', 'd', "'ɨ", '!', 'i', "'o", 'n', 'ə', 't', '.', 'p', '.', 'p', 'ə', 'p', 'r', 'ɐ', 'sʲ', "'i", 'ɫ', 'ə', 'v', 'ɐ', 'd', "'ɨ", '.', "'a", 'p', 'ɐ', 't', "'o", 'm', 'f', 'æ', 'b', 'ɹ', 'ɪ', 'k', "'", 's', 'i', 'f', 'sʲ', 'e', 'k', 'ɐ', 'nʲ', "'e", 't~s', '.', 'f', 'ɨ', 'f']
# в-пятых, в регионе должен установиться мир.
# отчетливо проговорила старушка, по-прежнему не отводя своих вопрошающих глаз от его лица.
# Должно быть, молодой человек взглянул на нее каким-нибудь особенным взглядом.
# Нам нужно продвинуться вперед, пока государства-члены не отвернулись от конференции.
# Мы по-прежнему глубоко встревожены хроническим тупиком.
# Масштабы распространения сексуального насилия в условиях вооруженного конфликта по-прежнему вызывают тревогу.
if __name__ == '__main__':
    # Проверка на омографах
    text = "– «То-то, батька мой, – отвечала она, – не тебе бы хитрить; посылай-ка за офицерами»..."
    norm1 = text_normalize(text)
    ph1, words, word2ph, string_mask = g2p(norm1)
    print(f"1Mask: {string_mask} {len(string_mask)}")
    for i in range(len(string_mask)):
        print(f"{string_mask[i]} - {norm1[i]}")
    for i in range(len(string_mask) - 1, -1, -1):
        try:
            if string_mask[i] == (-1):
                string_mask[i] = string_mask[i + 1]
        except Exception as e:
            print(e)

    print(f"Original: {text}")
    print(f"Normalized: {norm1, len(norm1)}")
    print(f"Accentizer output (first guess): {accentizer.process_all(norm1)}")
    print(f"Words: {words} len {len(words)}\n")
    print(f"word2ph: {word2ph}\n")
    print(f"Mask: {string_mask}\n")
    print(f"Phonemes: {ph1}\n")

    bert_pretrained_dir = "/home/mlb/Documents/PycharmProject/ai-collect/develop/GPT-SoVITS/GPT_SoVITS/pretrained_models/USER-bge-m3"
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(bert_pretrained_dir)
    bert_model = AutoModel.from_pretrained(bert_pretrained_dir).to(device).eval()
    print("Модели загружены.")

    def get_bert_feature(norma_text, word_list, phoneme_count_per_word, masked_string):
        """
        Получение эмбед для каждой фонемы с выравниванием.
        На вход - нормализованный текст, лист со словами, колво фонем на каждое слово, лист с номерами слов к которым относится символ в строке нормализованной
        """
        encoding = tokenizer(norma_text, return_offsets_mapping=True, return_tensors="pt")
        bert_token_spans = encoding['offset_mapping'][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0]) # [0] чтобы убрать batch-размерность

        print("Исходный текст:", norma_text)
        print("Токены от BERT:", tokens)
        print("SPANS:", bert_token_spans)
        print(f"Количество токенов: {len(tokens)}")
        # word_ids, которые сопоставляют каждый токен с его словом
        # word_ids() доступен, если токенизатор FastTokenizer
        try:
            word_ids = encoding.word_ids()
            print("IDS: ", word_ids)
        except Exception as e:
            raise Exception("Tokenizer does not support word_ids(). Please use a FastTokenizer.") from e

        assert len(bert_token_spans) == len(word_ids), f"Alignment failed: found {len(tokens)} TOKENS for {len(bert_token_spans)} SPANS."

        # получаем контекстуальные эмбеддинги
        encoding_to_model = {}
        for k in encoding:
            if k != "offset_mapping":
                encoding_to_model[k] = encoding[k].to(device)
        with torch.no_grad():
            res = bert_model(**encoding_to_model)
            all_token_embeddings = res.last_hidden_state.squeeze(0)

        assert len(all_token_embeddings) == len(bert_token_spans), f"Alignment failed: found {len(all_token_embeddings)} embeddings for {len(bert_token_spans)} SPANS."

        # группируем эмбеддинги токенов по словам, используя word_ids
        word_embeddings_grouped = {}
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is not None: # Пропускаем спецтокены [CLS], [SEP]
                # на основе спанов получаем к какому слову относится эмбеддинг тензор с берта
                word_id = max(masked_string[bert_token_spans[token_idx][0]:bert_token_spans[token_idx][1]])
                if word_id not in word_embeddings_grouped:
                    word_embeddings_grouped[word_id] = []
                # print(f"Word id: {word_id} (mask {masked_string[bert_token_spans[token_idx][0]:bert_token_spans[token_idx][1]]}) (str {norma_text[bert_token_spans[token_idx][0]:bert_token_spans[token_idx][1]]})")
                word_embeddings_grouped[word_id].append(all_token_embeddings[token_idx])

        # усредняем эмбеддинги для каждого слова
        word_embeddings = []
        for i in sorted(word_embeddings_grouped.keys()):
            word_emb = torch.mean(torch.stack(word_embeddings_grouped[i]), dim=0)
            word_embeddings.append(word_emb)

        assert len(word_list) == len(word_embeddings), f"Alignment failed: found {len(word_embeddings)} embeddings for {len(word_list)} words."

        # растягиваем эмбеддинги слов на фонемы
        phone_level_feature = []
        for i in range(len(word_embeddings)):
            word_emb = word_embeddings[i]
            num_phonemes = phoneme_count_per_word[i]
            if num_phonemes > 0:
                phone_level_feature.append(word_emb.repeat(num_phonemes, 1))

        if not phone_level_feature:
            return torch.zeros((0, bert_model.config.hidden_size))

        # return torch.cat(phone_level_feature, dim=0).cpu()
        final_features = torch.cat(phone_level_feature, dim=0)
        return final_features.T

    print(get_bert_feature(norm1, words, word2ph, string_mask))
