import tkinter as tk
from tkinter import ttk
import threading
import pyaudio
import numpy as np
from loguru import logger
from funasr import AutoModel
from modelscope.pipelines import pipeline
import soundfile as sf
import os
import time

# 初始化日志
logger.remove()
logger.add(lambda msg: None)  # 禁用日志输出到控制台
from modelscope.utils.constant import Tasks

# 配置音频参数
CHUNK_SIZE_MS = 300
SAMPLE_RATE = 16000
CHUNK_SIZE = int(CHUNK_SIZE_MS * SAMPLE_RATE / 1000)
FORMAT = pyaudio.paInt16
CHANNELS = 1

emo_dict = {
	"<|HAPPY|>": "😊",
	"<|SAD|>": "😔",
	"<|ANGRY|>": "😡",
	"<|NEUTRAL|>": "",
	"<|FEARFUL|>": "😰",
	"<|DISGUSTED|>": "🤢",
	"<|SURPRISED|>": "😮",
}

event_dict = {
	"<|BGM|>": "🎼",
	"<|Speech|>": "",
	"<|Applause|>": "👏",
	"<|Laughter|>": "😀",
	"<|Cry|>": "😭",
	"<|Sneeze|>": "🤧",
	"<|Breath|>": "",
	"<|Cough|>": "🤧",
}

emoji_dict = {
	"<|nospeech|><|Event_UNK|>": "❓",
	"<|zh|>": "",
	"<|en|>": "",
	"<|yue|>": "",
	"<|ja|>": "",
	"<|ko|>": "",
	"<|nospeech|>": "",
	"<|HAPPY|>": "😊",
	"<|SAD|>": "😔",
	"<|ANGRY|>": "😡",
	"<|NEUTRAL|>": "",
	"<|BGM|>": "🎼",
	"<|Speech|>": "",
	"<|Applause|>": "👏",
	"<|Laughter|>": "😀",
	"<|FEARFUL|>": "😰",
	"<|DISGUSTED|>": "🤢",
	"<|SURPRISED|>": "😮",
	"<|Cry|>": "😭",
	"<|EMO_UNKNOWN|>": "",
	"<|Sneeze|>": "🤧",
	"<|Breath|>": "",
	"<|Cough|>": "😷",
	"<|Sing|>": "",
	"<|Speech_Noise|>": "",
	"<|withitn|>": "",
	"<|woitn|>": "",
	"<|GBG|>": "",
	"<|Event_UNK|>": "",
}

lang_dict =  {
    "<|zh|>": "<|lang|>",
    "<|en|>": "<|lang|>",
    "<|yue|>": "<|lang|>",
    "<|ja|>": "<|lang|>",
    "<|ko|>": "<|lang|>",
    "<|nospeech|>": "<|lang|>",
}

emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
event_set = {"🎼", "👏", "😀", "😭", "🤧", "😷",}

def format_str(s):
	for sptk in emoji_dict:
		s = s.replace(sptk, emoji_dict[sptk])
	return s


def format_str_v2(s):
	sptk_dict = {}
	for sptk in emoji_dict:
		sptk_dict[sptk] = s.count(sptk)
		s = s.replace(sptk, "")
	emo = "<|NEUTRAL|>"
	for e in emo_dict:
		if sptk_dict[e] > sptk_dict[emo]:
			emo = e
	for e in event_dict:
		if sptk_dict[e] > 0:
			s = event_dict[e] + s
	s = s + emo_dict[emo]

	for emoji in emo_set.union(event_set):
		s = s.replace(" " + emoji, emoji)
		s = s.replace(emoji + " ", emoji)
	return s.strip()

def format_str_v3(s):
	def get_emo(s):
		return s[-1] if s[-1] in emo_set else None
	def get_event(s):
		return s[0] if s[0] in event_set else None

	s = s.replace("<|nospeech|><|Event_UNK|>", "❓")
	for lang in lang_dict:
		s = s.replace(lang, "<|lang|>")
	s_list = [format_str_v2(s_i).strip(" ") for s_i in s.split("<|lang|>")]
	new_s = " " + s_list[0]
	cur_ent_event = get_event(new_s)
	for i in range(1, len(s_list)):
		if len(s_list[i]) == 0:
			continue
		if get_event(s_list[i]) == cur_ent_event and get_event(s_list[i]) != None:
			s_list[i] = s_list[i][1:]
		#else:
		cur_ent_event = get_event(s_list[i])
		if get_emo(s_list[i]) != None and get_emo(s_list[i]) == get_emo(new_s):
			new_s = new_s[:-1]
		new_s += s_list[i].strip().lstrip()
	new_s = new_s.replace("The.", " ")
	return new_s.strip()


# 初始化模型
sv_pipeline = pipeline(
    task='speaker-verification',
    model='iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k',
    model_revision='v1.0.0',
)

asr_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='iic/SenseVoiceSmall',
    model_revision="master",
    disable_update=True
)

model_asr = AutoModel(
    model="iic/SenseVoiceSmall",
    trust_remote_code=True,
    disable_update=True,
    device="cuda:0"
)
model_vad = AutoModel(
    model="fsmn-vad",
    model_revision="v2.0.4",
    disable_pbar=True,
    max_end_silence_time=500,
    disable_update=True,
    device="cuda:0"
)
def reg_spk_init(files):
    reg_spk = {}
    for f in files:
        data, sr = sf.read(f, dtype="float32")
        k, _ = os.path.splitext(os.path.basename(f))
        reg_spk[k] = {
            "data": data,
            "sr":   sr,
        }
    return reg_spk
def speaker_verify(audio, sv_thr, reg_spks):
    hit = False
    for k, v in reg_spks.items():
        res_sv = sv_pipeline([audio, v["data"]], sv_thr)
        if res_sv["score"] >= sv_thr:
           hit = True
        logger.info(f"[speaker_verify] audio_len: {len(audio)}; sv_thr: {sv_thr}; hit: {hit}; {k}: {res_sv}")
    return hit, k

def asr(input, lang, cache, use_itn=False):
    # with open('test.pcm', 'ab') as f:
    #     logger.debug(f'write {f.write(audio)} bytes to `test.pcm`')
    # result = asr_pipeline(audio, lang)
    start_time = time.time()
    result = model_asr.generate(
        input           = input,
        cache           = cache,
        language        = lang.strip(),
        use_itn         = use_itn,
        batch_size_s    = 60,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.debug(f"asr elapsed: {elapsed_time * 1000:.2f} milliseconds")
    return result
class SpeechRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("流式语音识别")
        self.root.geometry("600x400")

        self.audio_interface = pyaudio.PyAudio()
        self.running = False
        self.language = "auto"
        self.sv = False
        self.sv_threshold = 0.3

        self.selected_device_index = None
        self.reg_spks_files = []
        self.selected_speakers = []
        self.create_widgets()

    def create_widgets(self):
        # 语言选择
        tk.Label(self.root, text="选择语言:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.language_var = tk.StringVar(value="auto")
        self.language_menu = ttk.Combobox(
            self.root, textvariable=self.language_var, values=["auto", "zh", "en", "ja"]
        )
        self.language_menu.grid(row=0, column=1, padx=10, pady=5)

        # 麦克风选择
        tk.Label(self.root, text="选择麦克风:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.device_var = tk.StringVar(value="未选择")
        self.device_menu = ttk.Combobox(self.root, textvariable=self.device_var, state="readonly")
        self.device_menu.grid(row=1, column=1, padx=10, pady=5)

        self.update_device_list()

        # 录制新说话人
        tk.Label(self.root, text="录制新说话人:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.speaker_name_var = tk.StringVar()
        self.speaker_name_entry = ttk.Entry(self.root, textvariable=self.speaker_name_var)
        self.speaker_name_entry.grid(row=2, column=1, padx=10, pady=5)

        self.record_button = ttk.Button(self.root, text="录制", command=self.record_speaker)
        self.record_button.grid(row=2, column=2, padx=10, pady=5)

        # 说话人验证
        self.sv_var = tk.BooleanVar()
        self.sv_checkbox = ttk.Checkbutton(self.root, text="启用说话人验证", variable=self.sv_var, command=self.update_sv_state)
        self.sv_checkbox.grid(row=3, column=0, padx=10, pady=5, sticky="w")

        tk.Label(self.root, text="验证说话人:").grid(row=3, column=1, padx=10, pady=5, sticky="w")
        self.sv_speakers_menu = tk.Listbox(self.root, selectmode="multiple", height=4, exportselection=False)
        self.sv_speakers_menu.grid(row=3, column=2, padx=10, pady=5)
        self.update_sv_speakers()

        # 启动和停止按钮
        self.start_button = ttk.Button(self.root, text="启动识别", command=self.start_recognition)
        self.start_button.grid(row=4, column=0, padx=10, pady=5)
        self.stop_button = ttk.Button(self.root, text="停止识别", command=self.stop_recognition, state=tk.DISABLED)
        self.stop_button.grid(row=4, column=1, padx=10, pady=5)

        # 结果显示框
        self.result_text = tk.Text(self.root, height=10, wrap=tk.WORD)
        self.result_text.grid(row=5, column=0, columnspan=3, padx=10, pady=10)

    def update_device_list(self):
        devices = []
        for i in range(self.audio_interface.get_device_count()):
            device_info = self.audio_interface.get_device_info_by_index(i)
            if device_info["maxInputChannels"] > 0:  # 仅显示输入设备
                devices.append(device_info["name"])
        self.device_menu["values"] = devices

    def update_sv_speakers(self):
        speaker_files = [f for f in os.listdir("speaker") if f.endswith(".wav")]
        self.reg_spks_files = [os.path.join("speaker", f) for f in speaker_files]
        self.sv_speakers_menu.delete(0, tk.END)
        for file in speaker_files:
            self.sv_speakers_menu.insert(tk.END, file)

    def update_sv_state(self):
        if self.sv_var.get():
            self.sv_speakers_menu.config(state="normal")
        else:
            self.sv_speakers_menu.config(state="disabled")

    def record_speaker(self):
        speaker_name = self.speaker_name_var.get().strip()
        if not speaker_name:
            self.log_result("请输入说话人名称。")
            return

        device_index = self.device_menu.current()
        if device_index == -1:
            self.log_result("请选择麦克风。")
            return

        duration = 5  # 录制时长（秒）
        self.log_result(f"开始录制 {duration} 秒...")

        stream = self.audio_interface.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK_SIZE
        )

        frames = []
        for _ in range(0, int(SAMPLE_RATE / CHUNK_SIZE * duration)):
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            frames.append(data)

        stream.stop_stream()
        stream.close()

        filename = f"speaker/{speaker_name}.wav"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with sf.SoundFile(filename, mode="w", samplerate=SAMPLE_RATE, channels=CHANNELS, subtype="PCM_16") as file:
            file.write(np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32767.0)

        self.log_result(f"录制完成，文件保存为 {filename}")
        self.update_sv_speakers()

    def start_recognition(self):
        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.selected_device_index = self.device_menu.current()
        if self.selected_device_index == -1:
            self.log_result("请选择麦克风。")
            return
        self.sv = self.sv_var.get()
        self.language = self.language_var.get()
        self.selected_speakers = ["speaker/"+self.sv_speakers_menu.get(i) for i in self.sv_speakers_menu.curselection()]
        self.log_result(f"选定的说话人验证文件: {self.selected_speakers}")
        self.thread = threading.Thread(target=self.process_audio_stream)
        self.thread.daemon = True
        self.thread.start()

    def stop_recognition(self):
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def process_audio_stream(self):
        stream = self.audio_interface.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=self.selected_device_index,
            frames_per_buffer=CHUNK_SIZE
        )
        audio_vad = np.array([], dtype=np.float32)
        cache = {}
        cache_asr = {}
        offset = 0
        last_vad_beg = last_vad_end = -1
        hit = False
        self.log_result("开始语音识别...")

        try:
            while self.running:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767.0
                audio_vad = np.append(audio_vad, chunk)

                if last_vad_beg > 1:
                    if self.sv:
                        if not hit:
                            hit, speaker = speaker_verify(audio_vad[int((last_vad_beg - offset) * SAMPLE_RATE / 1000):],
                                                          self.sv_threshold,reg_spks=reg_spk_init(self.selected_speakers))
                            if hit:
                                spk = speaker
                            else:
                                spk = 'unknown'
                # 使用 VAD 检测语音段
                vad_result = model_vad.generate(input=chunk, cache=cache, is_final=False, chunk_size=CHUNK_SIZE_MS)
                if len(vad_result[0]["value"]):
                    vad_segments = vad_result[0]["value"]
                    for segment in vad_segments:
                        if segment[0] > -1:  # Speech begin
                            last_vad_beg = segment[0]
                        if segment[1] > -1:  # Speech end
                            last_vad_end = segment[1]
                        if last_vad_beg > -1 and last_vad_end > -1:
                            last_vad_beg -= offset
                            last_vad_end -= offset
                            offset += last_vad_end
                            beg = int(last_vad_beg * SAMPLE_RATE / 1000)
                            end = int(last_vad_end * SAMPLE_RATE / 1000)
                            logger.info(f"[vad segment] audio_len: {end - beg}")

                            result = asr(
                                input=audio_vad[beg:end],
                                cache=cache_asr,
                                lang=self.language,
                                use_itn=True
                            )
                            logger.info(f"asr response: {result}")
                            audio_vad = audio_vad[end:]
                            last_vad_beg = last_vad_end = -1
                            hit = False

                            if result is not None:
                                self.log_result(f"{spk}: {format_str_v3(result[0]['text'])}")
                            else:
                                self.log_result("忽略。")

        except Exception as e:
            self.log_result(f"错误: {str(e)}")
        finally:
            stream.stop_stream()
            stream.close()
            self.log_result("语音识别已停止。")

    def log_result(self, text):
        self.result_text.insert(tk.END, text + "\n")
        self.result_text.see(tk.END)

# 启动应用
if __name__ == "__main__":
    root = tk.Tk()
    app = SpeechRecognizerApp(root)
    root.mainloop()
