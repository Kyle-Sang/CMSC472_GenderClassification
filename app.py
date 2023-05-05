# %%
import gradio as gr
import torchaudio
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import librosa
import torch

# %%
def dump_pickle(file_path: str, file, mode: str = "wb"):
    import pickle

    with open(file_path, mode=mode) as f:
        pickle.dump(file, f)


def load_pickle(file_path: str, mode: str = "rb", encoding=""):
    import pickle

    with open(file_path, mode=mode) as f:
        return pickle.load(f, encoding=encoding)

# %%
label2id = load_pickle('label2id.pkl')
id2label = load_pickle('id2label.pkl')

# %%
model = AutoModelForAudioClassification.from_pretrained(
    "facebook/wav2vec2-base", num_labels=len(label2id), label2id=label2id, id2label=id2label
)

# %%
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

# %%
checkpoint = torch.load('pytorch_model.bin', map_location=torch.device('cpu'))

# %%
model.load_state_dict(checkpoint)

# %%
def predict(input):
    if input == None:
        return "Please input a valid file or record yourself by clicking the microphone"
    elif input:
        waveform, sr = librosa.load(input)
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        waveform = torchaudio.transforms.Resample(sr, 16_000)(waveform)
        inputs = feature_extractor(waveform, sampling_rate=feature_extractor.sampling_rate,
                                max_length=16000, truncation=True)
        tensor = torch.tensor(inputs['input_values'][0])
        with torch.no_grad():
            output = model(tensor)
            logits = output['logits'][0]
            label_id = torch.argmax(logits).item()
        label_name = id2label[str(label_id)]

        return label_name
    else:
        return "File is not valid"
# %%
demo = gr.Interface(
    fn=predict,
    title="Audio Gender Classification",
    description="Record your voice or upload an audio file to see what gender our model classifies it as",
    inputs=gr.Audio(source="microphone", type="filepath", optional=False, label="Speak to classify your voice!"), # record audio, save in temp file to feed to inference func
    outputs="text",
    examples= [["male.mp3"], ["female.mp3"]]
)

# %%
demo.launch()

# %%



