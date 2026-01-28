import openai

class Inspector:

    def __init__(self, api_key , model="gpt-4o-mini", base_url=''):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.messages = [

        ]
        self.function_repository = {}

    def add_functions(self, function_lib: dict) -> None:
        self.function_repository = function_lib

    def _call_chat_model(self, functions=None, include_functions=False):
        params = {
            "model": self.model,
            "messages": self.messages,
        }

        if include_functions:
            params['functions'] = functions
            params['function_call'] = "auto"

        try:
            return self.client.chat.completions.create(**params)
        except Exception as e:
            print("\n" + "="*80)
            print("❌ Inspector API 呼叫失敗")
            print("="*80)
            print(f"錯誤類型: {type(e).__name__}")
            print(f"錯誤訊息: {e}")
            print(f"模型: {self.model}")
            print(f"訊息數量: {len(self.messages)}")
            print("="*80 + "\n")
            return None

    def clear(self):
        self.messages = []
        self.function_repository = {}