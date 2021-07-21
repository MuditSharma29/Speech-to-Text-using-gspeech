import nemo
import ffmpeg
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")
files=["D:\archive\medical speech transcription and intent\Medical Speech, Transcription, and Intent\recordings\test\1249120_1853182_11719913.wav"]
transcription=model.transcribe(paths2audio_files=files)