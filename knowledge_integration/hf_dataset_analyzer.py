from .knw import knw


class HuggingFaceDatasetAnalyzer(knw):
    def __init__(self):
        super().__init__()
        self.name = "hf_dataset_analyzer"
        self.description = "Analyze Hugging Face datasets for Traditional Chinese content creation (CP) suitability. Load dataset, analyze fields for Chinese text ratio, empty values, string length, garbled text, and CP usage recommendations."
        self.core_function = "analyze_dataset"
        self.mode = 'full'

    def analyze_dataset(self):
        return """
        # Hugging Face 資料集分析完整程式碼
        from datasets import load_dataset
        import pandas as pd
        import numpy as np
        import json
        import re
        import os
        from dotenv import load_dotenv
        
        # 載入環境變數
        load_dotenv()
        
        # 設置 Hugging Face token（從環境變數讀取）
        hf_token = os.environ.get('HF_KEY') or os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
        if hf_token:
            print(f"✓ 已找到 Hugging Face token (長度: {len(hf_token)})")
        else:
            print("⚠ 未找到 HF_KEY 環境變數，某些私有資料集可能無法存取")
            print("   請確認 .env 文件中已設置 HF_KEY")
        
        # 如果需要繁簡轉換，可安裝並使用 OpenCC
        try:
            from opencc import OpenCC
            cc = OpenCC('t2s')  # 繁體轉簡體
            has_opencc = True
        except:
            has_opencc = False
            print("提示：安裝 opencc-python-reimplemented 可提升繁體中文判斷準確度")
        
        def is_traditional_chinese(text, threshold=0.8):
            '''判斷文字是否為繁體中文'''
            if not isinstance(text, str) or len(text) == 0:
                return False
            
            # 計算中文字符比例
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
            if chinese_chars == 0:
                return False
            
            chinese_ratio = chinese_chars / len(text)
            if chinese_ratio < 0.5:
                return False
            
            # 使用 OpenCC 判斷繁簡差異
            if has_opencc:
                simplified = cc.convert(text)
                diff_ratio = sum(1 for a, b in zip(text, simplified) if a != b) / len(text)
                return diff_ratio > 0.1  # 如果轉換後差異大於10%，可能是繁體
            
            # 簡單判斷：檢查常見繁體字
            traditional_chars = set('繁體中文資料內容創作適用於標題摘要')
            traditional_count = sum(1 for char in text if char in traditional_chars)
            return traditional_count > 0
        
        def contains_garbled_text(text):
            '''檢測亂碼'''
            if not isinstance(text, str):
                return False
            
            # 檢查常見亂碼符號
            garbled_patterns = ['��', 'â', 'Ã', '�']
            for pattern in garbled_patterns:
                if pattern in text:
                    return True
            
            # 檢查異常字符比例
            printable = sum(1 for c in text if c.isprintable() or c in '\\n\\r\\t')
            if len(text) > 0 and printable / len(text) < 0.8:
                return True
            
            return False
        
        def calculate_length_stats(values):
            '''計算字串長度統計'''
            lengths = [len(str(v)) for v in values if v is not None and str(v).strip()]
            if not lengths:
                return {"avg": 0, "std": 0, "min": 0, "max": 0}
            
            return {
                "avg": round(np.mean(lengths), 2),
                "std": round(np.std(lengths), 2),
                "min": int(np.min(lengths)),
                "max": int(np.max(lengths))
            }
        
        def evaluate_cp_suitability(column_analysis):
            '''評估是否適合繁中內容創作'''
            is_tc = column_analysis['is_traditional_chinese']
            non_empty = column_analysis['non_empty_ratio']
            has_garbled = column_analysis['contains_garbled_text']
            avg_length = column_analysis['length_stats']['avg']
            
            # 基本條件：繁體中文、高非空比例、無亂碼
            if not is_tc or non_empty < 0.7 or has_garbled:
                return False, []
            
            # 根據長度判斷用途
            suggestions = []
            if avg_length < 10:
                suggestions = ["標題", "標籤"]
            elif avg_length < 50:
                suggestions = ["摘要", "引言", "短描述"]
            elif avg_length < 200:
                suggestions = ["詳細描述", "段落", "註解"]
            else:
                suggestions = ["完整文章", "故事內容", "長文素材"]
            
            return True, suggestions
        
        def analyze_hf_dataset(dataset_name, split="train", num_samples=100):
            '''分析 Hugging Face 資料集'''
            print(f"正在載入資料集：{dataset_name}")
            
            # 準備載入參數
            load_kwargs = {}
            if hf_token:
                load_kwargs['token'] = hf_token
            
            # 載入資料集
            try:
                dataset = load_dataset(dataset_name, split=f"{split}[:{num_samples}]", **load_kwargs)
            except Exception as e:
                print(f"載入失敗，嘗試其他方式：{e}")
                try:
                    dataset = load_dataset(dataset_name, split=split, **load_kwargs)
                    dataset = dataset.select(range(min(num_samples, len(dataset))))
                except Exception as e2:
                    print(f"仍然失敗：{e2}")
                    print("提示：如需存取私有或需登入的資料集，請確保：")
                    print("1. 已設置 HF_KEY 環境變數")
                    print("2. 或使用 huggingface-cli login 登入")
                    raise
            
            # 轉換為 DataFrame
            df = pd.DataFrame(dataset)
            print(f"\\n資料集包含 {len(df)} 筆資料")
            print(f"欄位：{list(df.columns)}\\n")
            
            # 分析每個欄位
            summary = []
            for column in df.columns:
                values = df[column].tolist()
                
                # 計算非空比例
                non_empty_count = sum(1 for v in values if v is not None and str(v).strip())
                non_empty_ratio = non_empty_count / len(values)
                
                # 判斷繁體中文
                text_values = [str(v) for v in values if v is not None]
                is_tc = False
                tc_count = 0
                for text in text_values[:20]:  # 檢查前20個樣本
                    if is_traditional_chinese(text):
                        tc_count += 1
                if tc_count / max(len(text_values[:20]), 1) > 0.8:
                    is_tc = True
                
                # 檢測亂碼
                has_garbled = any(contains_garbled_text(str(v)) for v in values[:20])
                
                # 長度統計
                length_stats = calculate_length_stats(values)
                
                # 組合分析結果
                column_analysis = {
                    "column": column,
                    "is_traditional_chinese": is_tc,
                    "non_empty_ratio": round(non_empty_ratio, 3),
                    "length_stats": length_stats,
                    "contains_garbled_text": has_garbled
                }
                
                # 評估 CP 適用性
                recommended, suggestions = evaluate_cp_suitability(column_analysis)
                column_analysis["recommended_for_cp"] = recommended
                column_analysis["cp_usage_suggestions"] = suggestions
                
                summary.append(column_analysis)
            
            # 輸出結果
            result = {"summary": summary}
            print("\\n=== 分析結果 ===")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            
            return result
        
        # 使用範例
        # result = analyze_hf_dataset("username/dataset_name", split="train", num_samples=100)
        """


if __name__ == '__main__':
    analyzer = HuggingFaceDatasetAnalyzer()
    print(analyzer.get_core_function())
