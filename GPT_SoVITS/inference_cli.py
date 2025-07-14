import argparse
import os
import soundfile as sf
import torchaudio
from tools.i18n.i18n import I18nAuto
from datetime import datetime
import torch
import librosa

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

i18n = I18nAuto()
cnhubert_base_path = "/develop/GPT-SoVITS/GPT_SoVITS/pretrained_models/wavlm-plus"


from feature_extractor import cnhubert

cnhubert.cnhubert_base_path = cnhubert_base_path
ssl_model = cnhubert.get_model()
# if is_half == True:
ssl_model = ssl_model.half().to("cuda:0")
# else:
    # ssl_model = ssl_model.to("cuda:0")

def get_feat(ref_wav_path):
    wav16k, sr = librosa.load(ref_wav_path, sr=16000)
    if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
        # gr.Warning(i18n("参考音频在3~10秒范围外，请更换！"))
        raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
    wav16k = torch.from_numpy(wav16k)
    if sr != 16000:
        wav16k = wav16k.to("cuda:0")
        if wav16k.shape[0] == 2:
            wav16k = wav16k.mean(0)
        wav16k = torchaudio.transforms.Resample(sr, 16000).to("cuda:0")(wav16k)
    else:
        wav16k = wav16k.to("cuda:0")
        if wav16k.shape[0] == 2:
            wav16k = wav16k.mean(0)
    # if is_half == True:
    wav16k = wav16k.half().to("cuda:0")
    # else:
    # wav16k = wav16k.to("cuda:0")
    # wav16k = torch.cat([wav16k, zero_wav_torch])
    ssl_content = ssl_model(wav16k.unsqueeze(0)).transpose(1, 2)  # .float()
    return ssl_content

def synthesize(
    GPT_model_path,
    SoVITS_model_path,
    ref_audio_path,
    ref_text,
    ref_language,
    target_text,
    target_language,
    output_path,
    ssl
):

    # Change model weights
    print("MAIN GPT")
    from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav
    change_gpt_weights(gpt_path=GPT_model_path)
    print("MAIN SOVITS")
    next(change_sovits_weights(sovits_path=SoVITS_model_path))

    # Synthesize audio
    synthesis_result = get_tts_wav(
        ref_wav_path=ref_audio_path,
        prompt_text=ref_text,
        prompt_language=i18n(ref_language),
        text=target_text,
        text_language=i18n(target_language),
        top_k=5,
        top_p=0.4,
        speed=1.2,
        temperature=0.1,
        ssl=ssl
    )

    result_list = list(synthesis_result)

    if result_list:
        last_sampling_rate, last_audio_data = result_list[-1]

        # Получаем текущую дату и время
        now_str = datetime.now().strftime("%H-%M-%S")  # Например: 14-27-08

        # Вставляем время в имя файла
        filename = f"output_{output_path}_{now_str}.wav"
        output_wav_path = os.path.join("./", filename)

        # Сохраняем файл
        sf.write(output_wav_path, last_audio_data, last_sampling_rate)
        print(f"Audio saved to {output_wav_path}")


def main():
    # parser = argparse.ArgumentParser(description="GPT-SoVITS Command Line Tool")
    # parser.add_argument("--gpt_model", required=True, help="Path to the GPT model file")
    # parser.add_argument("--sovits_model", required=True, help="Path to the SoVITS model file")
    # parser.add_argument("--ref_audio", required=True, help="Path to the reference audio file")
    # parser.add_argument("--ref_text", required=True, help="Path to the reference text file")
    # parser.add_argument(
    #     "--ref_language", required=True, choices=["中文", "英文", "日文"], help="Language of the reference audio"
    # )
    # parser.add_argument("--target_text", required=True, help="Path to the target text file")
    # parser.add_argument(
    #     "--target_language",
    #     required=True,
    #     choices=["中文", "英文", "日文", "中英混合", "日英混合", "多语种混合"],
    #     help="Language of the target text",
    # )
    # parser.add_argument("--output_path", required=True, help="Path to the output directory")

    # args = parser.parse_args()
    # ref_wav = "/develop/GPT-SoVITS/GPT_SoVITS/koncevich_16.wav"
    ref_wav = "/develop/GPT-SoVITS/GPT_SoVITS/pugach_16.wav"
    ssl = get_feat(ref_wav)
    while True:
        inp = input("Phrase: ")
        if not inp:
            break
        # привет! что нового случилось за день? может есть чем поделиться со мной? я все выслушаю, говори.
        # привет что нового случилось за день может есть чем поделиться со мной я все выслушаю говори.
        synthesize(
            GPT_model_path = "/develop/GPT-SoVITS/GPT_weights_v2Pro/rus_clear-e20.ckpt",
            SoVITS_model_path = "/develop/GPT-SoVITS/SoVITS_weights_v2Pro/rus_clear_e40_s39360.pth",
            ref_audio_path = ref_wav,
            # ref_text = "Каждому из них нужен особый подход. У каждого свой неповторимый характер, все они требуют к себе чуткого отношения.",
            ref_text = "Мы хотим чтобы всем нашим гостям было у нас комфортно и весело. Приятного вам отдыха.",
            ref_language = "ru",
            target_text = str(inp),
            target_language = "ru",
            output_path = "clear_e20_e40",
            ssl = ssl
        )


if __name__ == "__main__":
    main()
