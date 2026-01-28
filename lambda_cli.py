#!/usr/bin/env python3
"""
LAMBDA CLI - Command Line Interface for LAMBDA
使用方式：
    python lambda_cli.py                    # 互動模式
    python lambda_cli.py "your question"    # 單次查詢
    python lambda_cli.py -f data.csv        # 上傳檔案後互動
"""

import sys
import os
from LAMBDA import LAMBDA
from pathlib import Path
import argparse
import shutil


class LAMBDACLI:
    def __init__(self):
        print("🚀 正在初始化 LAMBDA...")
        self.lambda_instance = LAMBDA(config_path='config.yaml')
        print("✅ LAMBDA 初始化完成！")
        print(f"📁 工作目錄: {self.lambda_instance.session_cache_path}\n")

    def send_message(self, message):
        """發送訊息並獲取回應"""
        print(f"\n💬 您的訊息: {message}")
        print("🤖 LAMBDA 正在處理...\n")
        
        # 準備聊天歷史
        chat_history = self.lambda_instance.conv.chat_history_display.copy()
        
        # 添加用戶訊息
        self.lambda_instance.conv.programmer.messages.append({
            "role": "user", 
            "content": message
        })
        chat_history.append([message, None])
        
        # 執行工作流程
        response_text = ""
        for chat_state in self.lambda_instance.conv.stream_workflow(chat_history):
            if chat_state and len(chat_state) > 0:
                latest_response = chat_state[-1][1]
                if latest_response and latest_response != response_text:
                    # 只打印新增的部分
                    new_text = latest_response[len(response_text):]
                    print(new_text, end='', flush=True)
                    response_text = latest_response
        
        print("\n")
        return response_text

    def upload_file(self, file_path):
        """上傳檔案到工作目錄"""
        if not os.path.exists(file_path):
            print(f"❌ 檔案不存在: {file_path}")
            return False
        
        filename = os.path.basename(file_path)
        dest_path = os.path.join(self.lambda_instance.session_cache_path, filename)
        shutil.copy(file_path, dest_path)
        self.lambda_instance.conv.file_list.append(filename)
        
        file_extension = os.path.splitext(file_path)[1].lower()
        local_cache_path = dest_path
        
        if file_extension in ['.csv', '.xlsx', '.xls', '.json']:
            self.lambda_instance.conv.add_data(local_cache_path)
            data_info = f"檔案 '{filename}' 已上傳。資料集資訊：\n{self.lambda_instance.conv.my_data_cache}"
        else:
            data_info = f"檔案 '{filename}' 已上傳到工作目錄。"
        
        print(f"✅ {data_info}\n")
        return True

    def show_dataframe(self):
        """顯示當前資料框"""
        try:
            df = self.lambda_instance.open_board()
            if df is not None:
                print("\n📊 當前資料框:")
                print(df)
                print()
            else:
                print("⚠️  目前沒有資料框")
        except Exception as e:
            print(f"❌ 無法顯示資料框: {e}")

    def interactive_mode(self):
        """互動模式"""
        print("=" * 60)
        print("🎯 LAMBDA CLI 互動模式")
        print("=" * 60)
        print("指令:")
        print("  /upload <file>  - 上傳檔案")
        print("  /show           - 顯示目前資料框")
        print("  /save           - 儲存對話")
        print("  /clear          - 清除對話")
        print("  /help           - 顯示幫助")
        print("  /quit 或 /exit  - 退出")
        print("=" * 60)
        print()
        
        while True:
            try:
                user_input = input("👤 您: ").strip()
                
                if not user_input:
                    continue
                
                # 處理指令
                if user_input.startswith('/'):
                    cmd_parts = user_input.split(maxsplit=1)
                    cmd = cmd_parts[0].lower()
                    
                    if cmd in ['/quit', '/exit', '/q']:
                        print("👋 再見！")
                        break
                    
                    elif cmd == '/help':
                        print("\n可用指令:")
                        print("  /upload <file>  - 上傳檔案")
                        print("  /show           - 顯示目前資料框")
                        print("  /save           - 儲存對話")
                        print("  /clear          - 清除對話")
                        print("  /quit, /exit    - 退出\n")
                    
                    elif cmd == '/upload':
                        if len(cmd_parts) < 2:
                            print("❌ 請指定檔案路徑: /upload <file>")
                        else:
                            self.upload_file(cmd_parts[1])
                    
                    elif cmd == '/show':
                        self.show_dataframe()
                    
                    elif cmd == '/save':
                        self.lambda_instance.save_dialogue(
                            self.lambda_instance.conv.chat_history_display
                        )
                        print(f"✅ 對話已儲存到: {self.lambda_instance.session_cache_path}")
                    
                    elif cmd == '/clear':
                        self.lambda_instance.conv.clear()
                        print("✅ 對話已清除")
                    
                    else:
                        print(f"❌ 未知指令: {cmd}，輸入 /help 查看可用指令")
                    
                    continue
                
                # 發送一般訊息
                self.send_message(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 再見！")
                break
            except Exception as e:
                print(f"\n❌ 錯誤: {e}")
                import traceback
                traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='LAMBDA CLI - Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python lambda_cli.py                        # 啟動互動模式
  python lambda_cli.py "分析這個數據集"         # 單次查詢
  python lambda_cli.py -f data.csv           # 上傳檔案後進入互動模式
  python lambda_cli.py -f data.csv "顯示前5行"  # 上傳檔案並執行查詢
        """
    )
    
    parser.add_argument(
        'message',
        nargs='?',
        help='要發送的訊息（省略則進入互動模式）'
    )
    parser.add_argument(
        '-f', '--file',
        help='要上傳的檔案路徑'
    )
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='強制進入互動模式'
    )
    
    args = parser.parse_args()
    
    # 初始化 CLI
    cli = LAMBDACLI()
    
    # 如果有檔案，先上傳
    if args.file:
        cli.upload_file(args.file)
    
    # 決定執行模式
    if args.interactive or (not args.message):
        # 互動模式
        cli.interactive_mode()
    else:
        # 單次查詢模式
        cli.send_message(args.message)


if __name__ == '__main__':
    main()
