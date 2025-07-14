# -*- coding: utf-8 -*-

import os

inp_text = os.environ.get("inp_text")
inp_wav_dir = os.environ.get("inp_wav_dir")
exp_name = os.environ.get("exp_name")
i_part = os.environ.get("i_part")
all_parts = os.environ.get("all_parts")
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
opt_dir = os.environ.get("opt_dir")
bert_pretrained_dir = os.environ.get("bert_pretrained_dir")
import torch

is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
version = os.environ.get("version", None)
import traceback
import os.path
from text.cleaner import clean_text
from transformers import AutoModel, AutoTokenizer
from tools.my_utils import clean_path

# inp_text=sys.argv[1]
# inp_wav_dir=sys.argv[2]
# exp_name=sys.argv[3]
# i_part=sys.argv[4]
# all_parts=sys.argv[5]
# os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[6]#i_gpu
# opt_dir="/data/docker/liujing04/gpt-vits/fine_tune_dataset/%s"%exp_name
# bert_pretrained_dir="/data/docker/liujing04/bert-vits2/Bert-VITS2-master20231106/bert/chinese-roberta-wwm-ext-large"

from time import time as ttime
import shutil


def my_save(fea, path):  #####fix issue: torch.save doesn't support chinese path
    dir = os.path.dirname(path)
    name = os.path.basename(path)
    # tmp_path="%s/%s%s.pth"%(dir,ttime(),i_part)
    tmp_path = "%s%s.pth" % (ttime(), i_part)
    print(f"Save bert feat to: {tmp_path} move to {'%s/%s' % (dir, name)}")
    torch.save(fea, tmp_path)
    shutil.move(tmp_path, "%s/%s" % (dir, name))


txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
if os.path.exists(txt_path) == False:
    bert_dir = "%s/3-bert" % (opt_dir)
    os.makedirs(opt_dir, exist_ok=True)
    os.makedirs(bert_dir, exist_ok=True)
    if torch.cuda.is_available():
        device = "cuda:0"
    # elif torch.backends.mps.is_available():
    #     device = "mps"
    else:
        device = "cpu"
    if os.path.exists(bert_pretrained_dir):
        ...
    else:
        raise FileNotFoundError(bert_pretrained_dir)
    tokenizer = AutoTokenizer.from_pretrained(bert_pretrained_dir)
    bert_model = AutoModel.from_pretrained(bert_pretrained_dir)
    if is_half == True:
        bert_model = bert_model.half().to(device)
    else:
        bert_model = bert_model.to(device)
    bert_model.eval()
    print(f"LOADED BERT {bert_pretrained_dir}")
    def get_bert_feature(norma_text, word_list, phoneme_count_per_word, masked_string):
        """
        Получение эмбед для каждой фонемы с выравниванием.
        На вход - нормализованный текст, лист со словами, колво фонем на каждое слово, лист с номерами слов к которым относится символ в строке нормализованной
        """
        encoding = tokenizer(norma_text, return_offsets_mapping=True, return_tensors="pt")
        bert_token_spans = encoding['offset_mapping'][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0]) # [0] чтобы убрать batch-размерность

        # print("Исходный текст:", norma_text)
        # print("Токены от BERT:", tokens)
        # print("SPANS:", bert_token_spans)
        # print(f"Количество токенов: {len(tokens)}")
        # word_ids, которые сопоставляют каждый токен с его словом
        # word_ids() доступен, если токенизатор FastTokenizer
        try:
            word_ids = encoding.word_ids()
            # print("IDS: ", word_ids)
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

        # растягиваем эмбеддинги слов на фонемы усредняя
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

    def process(data, res):
        for name, text, lan in data:
            try:
                base_name = os.path.basename(name)

                name = clean_path(name)

                print(f"Process token g2p for v({version}): {lan} {base_name}")

                try:
                    text_clear = text.replace("%", "-").replace("￥", ",")
                    phones, words, word2ph, norm_text, string_mask = clean_text(text_clear, lan, version)
                except Exception as e:
                    print(f"ERROR CLEANER: {e}, text: {text_clear}")
                    raise
                path_bert = "%s/%s.pt" % (bert_dir, base_name)
            
                if not os.path.exists(path_bert) and (lan == "zh" or lan == "ru"):
                    try:
                        bert_feature = get_bert_feature(norm_text, words, word2ph, string_mask)
                    except Exception as e:
                        print(f"{'*'*10}\nERROR BERT: {e}, text {norm_text}\n{words}\n{word2ph}\n{string_mask}\n{'*'*10}")
                        raise
                    assert bert_feature.shape[-1] == len(phones) , f"Alignment failed main func: found bert_feature.shape {bert_feature.shape[-1]}  for {len(phones)} phones."
                    # torch.save(bert_feature, path_bert)
                    my_save(bert_feature, path_bert)
                    
                phones = " ".join(phones)
                res.append([name, phones, word2ph, norm_text])
            except:
                print(name, text, traceback.format_exc())

    todo = []
    res = []
    with open(inp_text, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")

    language_v1_to_language_v2 = {
        "ZH": "zh",
        "zh": "zh",
        "JP": "ja",
        "jp": "ja",
        "JA": "ja",
        "ja": "ja",
        "EN": "en",
        "en": "en",
        "En": "en",
        "KO": "ko",
        "Ko": "ko",
        "ko": "ko",
        "yue": "yue",
        "YUE": "yue",
        "Yue": "yue",
        "RU": "ru"
    }
    for line in lines[int(i_part) :: int(all_parts)]:
        try:
            wav_name, spk_name, language, text = line.split("|")
            # todo.append([name,text,"zh"])
            if language in language_v1_to_language_v2.keys():
                todo.append([wav_name, text, language_v1_to_language_v2.get(language, language)])
            else:
                print(f"\033[33m[Waring] The {language = } of {wav_name} is not supported for training.\033[0m")
        except:
            print("EXCEPTION", line, traceback.format_exc())

    print("Start process tokenizing")
    process(todo, res)
    opt = []
    for name, phones, word2ph, norm_text in res:
        opt.append("%s\t%s\t%s\t%s" % (name, phones, word2ph, norm_text))
    with open(txt_path, "w", encoding="utf8") as f:
        f.write("\n".join(opt) + "\n")
    print(f"Token finished, good samples {len(res)} all {len(todo)}")
