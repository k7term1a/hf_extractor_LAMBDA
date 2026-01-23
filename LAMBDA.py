import shutil
import gradio as gr
import json
import time
from conversation import Conversation
from prompt_engineering.prompts import *
import yaml
from utils.utils import *
import sys
import os
import re
from dotenv import load_dotenv


def resolve_env_variables(config):
    """
    遞迴解析配置中的環境變數。
    支援格式: %env:VARIABLE_NAME% 或 ${VARIABLE_NAME}
    """
    if isinstance(config, dict):
        return {key: resolve_env_variables(value) for key, value in config.items()}
    elif isinstance(config, list):
        return [resolve_env_variables(item) for item in config]
    elif isinstance(config, str):
        # 解析 %env:VAR_NAME% 格式
        pattern1 = r'%env:([^%]+)%'
        # 解析 ${VAR_NAME} 格式
        pattern2 = r'\$\{([^}]+)\}'
        
        def replace_env(match):
            var_name = match.group(1)
            env_value = os.environ.get(var_name)
            if env_value is None:
                print(f"[WARNING] Environment variable '{var_name}' not found, using original value")
                return match.group(0)
            return env_value
        
        # 先處理 %env:VAR% 格式
        config = re.sub(pattern1, replace_env, config)
        # 再處理 ${VAR} 格式
        config = re.sub(pattern2, replace_env, config)
    return config


class LAMBDA:
    def __init__(self, config_path='config.yaml'):
        ensure_config_file("config.yaml")
        print("Try to load config: ", config_path)

        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            bundle_dir = os.path.dirname(sys.executable)
        else:
            bundle_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 從 .env 檔案載入環境變數
        env_path = os.path.join(bundle_dir, '.env')
        if os.path.exists(env_path):
            load_dotenv(env_path)
            print(f"[INFO] Loaded environment variables from {env_path}")
        else:
            # 嘗試從當前目錄載入
            load_dotenv()
            print("[INFO] Attempting to load .env from current directory")
        
        config_path = os.path.join(bundle_dir, config_path)

        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        # 解析配置中的環境變數
        self.config = resolve_env_variables(self.config)
        
        if self.config["load_chat"] == True:
            self.load_dialogue(self.config["chat_history_path"])
        else:
            self.session_cache_path = self.init_local_cache_path(to_absolute_path(self.config["project_cache_path"]))
            self.config["session_cache_path"] = self.session_cache_path
        print("Session cache path: ", self.session_cache_path)
        self.conv = Conversation(self.config)

        self.conv.programmer.messages = [
            {
                "role": "system",
                "content": PROGRAMMER_PROMPT.format(working_path=self.session_cache_path)
            }
        ]

        if self.conv.retrieval:
            self.conv.programmer.messages[0]["content"] += KNOWLEDGE_INTEGRATION_SYSTEM


    def init_local_cache_path(self, project_cache_path):
        current_fold = time.strftime('%Y-%m-%d', time.localtime())
        hsid = str(hash(id(self)))  # new_uuid = str(uuid.uuid4())
        session_cache_path = os.path.join(project_cache_path, current_fold + '-' + hsid)
        if not os.path.exists(session_cache_path):
            os.makedirs(session_cache_path)
        return session_cache_path

    def open_board(self):
        return self.conv.show_data()

    def add_file(self, files):
        file_path = files.name
        shutil.copy(file_path, self.session_cache_path)
        filename = os.path.basename(file_path)
        file_extension = os.path.splitext(file_path)[1].lower()
        self.conv.file_list.append(filename)
        local_cache_path = os.path.join(self.session_cache_path, filename)

        if file_extension in ['.xlsx', '.xls']:
            self.conv.add_data(file_path)
            gen_info = self.conv.my_data_cache.get_description()
            self.conv.programmer.messages[0][
                "content"] += f"\nNow, user uploads the data in {local_cache_path}\n, and here is the general information of the dataset:\n {gen_info}. \nYou should care about the missing values and type of each column in your later processing."
        else:
            self.conv.programmer.messages[0][
                "content"] += f"\nNow, user uploads the files in {local_cache_path}."

        print(f"Upload file in gradio path: {file_path}, local cache path: {local_cache_path}")

    def rendering_code(self):
        return self.conv.rendering_code()

    def generate_report(self, chat_history):
        down_path = self.conv.document_generation(chat_history)
        return [gr.Button(visible=False), gr.DownloadButton(label=f"Download Report", value=down_path, visible=True)]

    def export_code(self):
        down_path = self.conv.export_code()
        return [gr.Button(visible=False), gr.DownloadButton(label=f"Download Notebook", value=down_path, visible=True)]

    def down_report(self):
        return [gr.Button(visible=True), gr.DownloadButton(visible=False)]

    def down_notebook(self):
        return [gr.Button(visible=True), gr.DownloadButton(visible=False)]

    def chat_streaming(self, message, chat_history, code=None):
        if not code:
            self.conv.programmer.messages.append({"role": "user", "content": message})
        else:
            message = code
        return "", chat_history + [[message, None]]

    def save_dialogue(self, chat_history):
        self.conv.save_conv()
        with open(os.path.join(self.session_cache_path, 'system_dialogue.json'), 'w') as f:
            json.dump(chat_history, f, indent=4)
        print(f"Dialogue saved in {os.path.join(self.session_cache_path, 'system_dialogue.json')}.")

    def load_dialogue(self, dialogue_path):
        try:
            system_dialogue_path = os.path.join(dialogue_path, 'system_dialogue.json')
            system_config_path = os.path.join(dialogue_path, 'config.json')
            with open(system_dialogue_path, 'r') as f:
                chat_history = json.load(f)
            with open(system_config_path, 'r') as f:
                sys_config = json.load(f)
            self.session_cache_path = sys_config["session_cache_path"]
            self.config["session_cache_path"] = self.session_cache_path
            self.config["chat_history_display"] = chat_history
            self.config["figure_list"] = sys_config["figure_list"]
            return chat_history
        except Exception as e:
            print(f"Failed to load the chat history: {e}")
            return []

    def clear_all(self, message, chat_history):
        self.conv.clear()
        return "", []

    def update_config(self, conv_model, programmer_model, inspector_model, api_key,
                      base_url_conv_model, base_url_programmer, base_url_inspector,
                      max_attempts, max_exe_time,
                      load_chat, chat_history_path):

        self.conv.update_config(conv_model=conv_model, programmer_model=programmer_model, inspector_model=inspector_model, api_key=api_key,
                      base_url_conv_model=base_url_conv_model, base_url_programmer=base_url_programmer, base_url_inspector=base_url_inspector,
                      max_attempts=max_attempts, max_exe_time=max_exe_time)

        if load_chat == True:
            self.config['chat_history_path'] = chat_history_path
            chat_history = self.load_dialogue(chat_history_path)
            self.config['load_chat'] = load_chat
            return ["### Config Updated!", chat_history]

        return "### Config Updated!", []
