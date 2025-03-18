from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QComboBox, QCheckBox, QLineEdit, 
                            QPushButton, QTextEdit, QListWidget, QFrame)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPalette, QColor
import pyaudio
import numpy as np
from loguru import logger
import soundfile as sf
import os
import time
import sys

# 导入原有的模型和处理函数
from STT_tk import (model_asr, model_vad, sv_pipeline, asr, speaker_verify, 
                   format_str_v3, reg_spk_init)

# 音频参数
CHUNK_SIZE_MS = 100
SAMPLE_RATE = 16000
CHUNK_SIZE = int(CHUNK_SIZE_MS * SAMPLE_RATE / 1000)
FORMAT = pyaudio.paInt16
CHANNELS = 1

class StyleSheet:
    MAIN_STYLE = """
        QMainWindow {
            background-color: #1a1a2e;
        }
        QLabel {
            color: #e0e0e0;
            font-size: 14px;
            font-weight: bold;
        }
        QPushButton {
            background-color: #4a69bd;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            font-weight: bold;
            min-width: 100px;
        }
        QPushButton:hover {
            background-color: #1e3799;
        }
        QPushButton:disabled {
            background-color: #485460;
            color: #a5b1c2;
        }
        QComboBox {
            padding: 8px;
            border: 2px solid #4a69bd;
            border-radius: 6px;
            background: rgba(255, 255, 255, 0.1);
            color: white;
            min-width: 200px;
        }
        QComboBox::drop-down {
            border: none;
        }
        QComboBox::down-arrow {
            image: none;
            border: none;
        }
        QTextEdit {
            background-color: rgba(255, 255, 255, 0.05);
            border: 2px solid #4a69bd;
            border-radius: 8px;
            padding: 12px;
            font-size: 14px;
            color: #e0e0e0;
        }
        QCheckBox {
            color: #e0e0e0;
            font-size: 14px;
            spacing: 8px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 2px solid #4a69bd;
            border-radius: 4px;
            background: rgba(255, 255, 255, 0.1);
        }
        QCheckBox::indicator:checked {
            background: #4a69bd;
        }
        QListWidget {
            background-color: rgba(255, 255, 255, 0.05);
            border: 2px solid #4a69bd;
            border-radius: 6px;
            color: #e0e0e0;
            padding: 5px;
        }
        QLineEdit {
            padding: 8px;
            border: 2px solid #4a69bd;
            border-radius: 6px;
            background: rgba(255, 255, 255, 0.05);
            color: white;
        }
        QFrame#controlPanel {
            background-color: rgba(255, 255, 255, 0.05);
            border: 2px solid #4a69bd;
            border-radius: 12px;
            padding: 20px;
        }
    """

class SubtitleWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | 
                          Qt.WindowType.WindowStaysOnTopHint |
                          Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setStyleSheet("""
            QTextEdit {
                color: #e0e0e0;
                background-color: rgba(26, 26, 46, 180);
                border: 2px solid #4a69bd;
                border-radius: 10px;
                padding: 12px;
                font-size: 16px;
            }
        """)
        layout.addWidget(self.text_display)
        
        # 添加一个小按钮用于关闭字幕窗口
        self.close_button = QPushButton("×")
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(74, 105, 189, 0.5);
                color: white;
                border: none;
                border-radius: 10px;
                padding: 2px;
                max-width: 20px;
                max-height: 20px;
            }
            QPushButton:hover {
                background-color: rgba(30, 55, 153, 0.8);
            }
        """)
        self.close_button.clicked.connect(self.hide)
        
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)
        layout.insertLayout(0, button_layout)
        
        self.setLayout(layout)
        
    def update_text(self, text):
        self.text_display.setText(text)
        
    def mousePressEvent(self, event):
        self.old_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        delta = event.globalPosition().toPoint() - self.old_pos
        self.move(self.x() + delta.x(), self.y() + delta.y())
        self.old_pos = event.globalPosition().toPoint()

class AudioThread(QThread):
    text_ready = pyqtSignal(str)
    
    def __init__(self, device_index, language, sv_enabled, selected_speakers):
        super().__init__()
        self.device_index = device_index
        self.language = language
        self.sv_enabled = sv_enabled
        self.selected_speakers = selected_speakers
        self.running = False
        
    def run(self):
        self.running = True
        audio_interface = pyaudio.PyAudio()
        
        stream = audio_interface.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=CHUNK_SIZE
        )
        
        audio_vad = np.array([], dtype=np.float32)
        cache = {}
        cache_asr = {}
        offset = 0
        last_vad_beg = last_vad_end = -1
        hit = False
        spk = 'unknown'
        
        try:
            while self.running:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767.0
                audio_vad = np.append(audio_vad, chunk)

                if last_vad_beg > 1:
                    if self.sv_enabled:
                        if not hit:
                            hit, speaker = speaker_verify(
                                audio_vad[int((last_vad_beg - offset) * SAMPLE_RATE / 1000):],
                                0.3,
                                reg_spks=reg_spk_init(self.selected_speakers)
                            )
                            if hit:
                                spk = speaker
                            else:
                                spk = 'unknown'

                vad_result = model_vad.generate(
                    input=chunk, 
                    cache=cache, 
                    is_final=False, 
                    chunk_size=CHUNK_SIZE_MS
                )
                
                if len(vad_result[0]["value"]):
                    vad_segments = vad_result[0]["value"]
                    for segment in vad_segments:
                        if segment[0] > -1:
                            last_vad_beg = segment[0]
                        if segment[1] > -1:
                            last_vad_end = segment[1]
                        if last_vad_beg > -1 and last_vad_end > -1:
                            last_vad_beg -= offset
                            last_vad_end -= offset
                            offset += last_vad_end
                            beg = int(last_vad_beg * SAMPLE_RATE / 1000)
                            end = int(last_vad_end * SAMPLE_RATE / 1000)

                            result = asr(
                                input=audio_vad[beg:end],
                                cache=cache_asr,
                                lang=self.language,
                                use_itn=True
                            )
                            
                            audio_vad = audio_vad[end:]
                            last_vad_beg = last_vad_end = -1
                            hit = False

                            if result is not None:
                                self.text_ready.emit(f"{spk}: {format_str_v3(result[0]['text'])}")
                            
        finally:
            stream.stop_stream()
            stream.close()
            audio_interface.terminate()

    def stop(self):
        self.running = False

class RecordThread(QThread):
    finished = pyqtSignal(str)
    
    def __init__(self, device_index, speaker_name):
        super().__init__()
        self.device_index = device_index
        self.speaker_name = speaker_name
        
    def run(self):
        audio_interface = pyaudio.PyAudio()
        try:
            duration = 5  # 录制时长（秒）
            stream = audio_interface.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=CHUNK_SIZE
            )
            
            frames = []
            for _ in range(0, int(SAMPLE_RATE / CHUNK_SIZE * duration)):
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                frames.append(data)
                
            stream.stop_stream()
            stream.close()
            
            filename = f"speaker/{self.speaker_name}.wav"
            os.makedirs("speaker", exist_ok=True)
            with sf.SoundFile(filename, mode="w", samplerate=SAMPLE_RATE, channels=CHANNELS, subtype="PCM_16") as file:
                file.write(np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32767.0)
                
            self.finished.emit(filename)
            
        finally:
            audio_interface.terminate()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("语音识别")
        self.setMinimumSize(800, 600)
        self.setStyleSheet(StyleSheet.MAIN_STYLE)
        
        # 创建字幕窗口
        self.subtitle_window = SubtitleWindow()
        
        # 初始化音频接口
        self.audio_interface = pyaudio.PyAudio()
        self.audio_thread = None
        
        self.init_ui()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # 控制面板
        control_panel = QFrame()
        control_panel.setObjectName("controlPanel")
        control_layout = QVBoxLayout(control_panel)
        control_layout.setSpacing(20)
        
        # 顶部设置区域
        settings_layout = QHBoxLayout()
        settings_layout.setSpacing(30)
        
        # 语言选择
        lang_layout = QVBoxLayout()
        lang_label = QLabel("语言选择")
        lang_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["auto", "zh", "en", "ja"])
        lang_layout.addWidget(lang_label)
        lang_layout.addWidget(self.lang_combo)
        
        # 麦克风选择
        mic_layout = QVBoxLayout()
        mic_label = QLabel("麦克风选择")
        mic_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mic_combo = QComboBox()
        self.update_device_list()
        mic_layout.addWidget(mic_label)
        mic_layout.addWidget(self.mic_combo)
        
        settings_layout.addLayout(lang_layout)
        settings_layout.addLayout(mic_layout)
        control_layout.addLayout(settings_layout)
        
        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("background-color: #4a69bd;")
        control_layout.addWidget(line)
        
        # 说话人验证区域
        sv_layout = QHBoxLayout()
        sv_layout.setSpacing(30)
        
        # 左侧说话人列表
        sv_left_layout = QVBoxLayout()
        self.sv_checkbox = QCheckBox("启用说话人验证")
        self.sv_list = QListWidget()
        self.sv_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.sv_list.setMinimumHeight(150)
        self.update_sv_speakers()
        sv_left_layout.addWidget(self.sv_checkbox)
        sv_left_layout.addWidget(self.sv_list)
        
        # 右侧录制区域
        sv_right_layout = QVBoxLayout()
        speaker_label = QLabel("录制新说话人")
        speaker_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.speaker_input = QLineEdit()
        self.speaker_input.setPlaceholderText("输入说话人名称")
        self.record_btn = QPushButton("开始录制")
        sv_right_layout.addWidget(speaker_label)
        sv_right_layout.addWidget(self.speaker_input)
        sv_right_layout.addWidget(self.record_btn)
        sv_right_layout.addStretch()
        
        sv_layout.addLayout(sv_left_layout)
        sv_layout.addLayout(sv_right_layout)
        control_layout.addLayout(sv_layout)
        
        layout.addWidget(control_panel)
        
        # 控制按钮区域
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(20)
        self.start_btn = QPushButton("开始识别")
        self.stop_btn = QPushButton("停止识别")
        self.subtitle_btn = QPushButton("显示字幕")
        
        self.start_btn.clicked.connect(self.start_recognition)
        self.stop_btn.clicked.connect(self.stop_recognition)
        self.subtitle_btn.clicked.connect(self.toggle_subtitle)
        
        self.stop_btn.setEnabled(False)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.subtitle_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # 结果显示区域
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(200)
        layout.addWidget(self.result_text)
        
        # 添加录制按钮的点击事件连接
        self.record_btn.clicked.connect(self.record_speaker)
        
    def update_device_list(self):
        self.mic_combo.clear()
        for i in range(self.audio_interface.get_device_count()):
            device_info = self.audio_interface.get_device_info_by_index(i)
            if device_info["maxInputChannels"] > 0:
                self.mic_combo.addItem(device_info["name"])
                
    def update_sv_speakers(self):
        self.sv_list.clear()
        if not os.path.exists("speaker"):
            os.makedirs("speaker")
        speaker_files = [f for f in os.listdir("speaker") if f.endswith(".wav")]
        self.sv_list.addItems(speaker_files)
        
    def record_speaker(self):
        speaker_name = self.speaker_input.text().strip()
        if not speaker_name:
            return
            
        device_index = self.mic_combo.currentIndex()
        if device_index == -1:
            return
            
        self.record_btn.setEnabled(False)
        self.record_thread = RecordThread(device_index, speaker_name)
        self.record_thread.finished.connect(self.recording_finished)
        self.record_thread.start()
        
    def recording_finished(self, filename):
        self.record_btn.setEnabled(True)
        self.update_sv_speakers()
        self.log_message(f"录制完成: {filename}")
        
    def start_recognition(self):
        device_index = self.mic_combo.currentIndex()
        if device_index == -1:
            return
            
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        selected_speakers = ["speaker/"+item.text() for item in 
                           self.sv_list.selectedItems()]
        
        self.audio_thread = AudioThread(
            device_index,
            self.lang_combo.currentText(),
            self.sv_checkbox.isChecked(),
            selected_speakers
        )
        self.audio_thread.text_ready.connect(self.log_message)
        self.audio_thread.start()
        
    def stop_recognition(self):
        if self.audio_thread:
            self.audio_thread.stop()
            self.audio_thread.wait()
            
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
    def toggle_subtitle(self):
        if self.subtitle_window.isVisible():
            self.subtitle_window.hide()
            self.subtitle_btn.setText("显示字幕")
        else:
            # 设置字幕窗口位置
            screen = QApplication.primaryScreen().geometry()
            self.subtitle_window.setGeometry(
                screen.width() // 4,
                screen.height() * 3 // 4,
                screen.width() // 2,
                screen.height() // 6
            )
            self.subtitle_window.show()
            self.subtitle_btn.setText("隐藏字幕")
        
    def log_message(self, message):
        self.result_text.append(message)
        # 更新字幕窗口
        if self.subtitle_window.isVisible():
            # 只显示最后几行
            lines = self.result_text.toPlainText().split('\n')
            if len(lines) > 3:
                self.subtitle_window.update_text('\n'.join(lines[-3:]))
            else:
                self.subtitle_window.update_text(message)
        
    def closeEvent(self, event):
        self.subtitle_window.close()
        if self.audio_thread and self.audio_thread.isRunning():
            self.audio_thread.stop()
            self.audio_thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())