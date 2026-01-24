from .knw import knw


class HuggingFaceDatasetAnalyzer(knw):
    def __init__(self):
        super().__init__()
        self.name = "HF資料集分析器"
        self.description = "分析 Hugging Face 資料集 繁體中文 持續預訓練 Continue Pretrain CP 適用性評估 資料集品質檢測 繁簡轉換 亂碼檢測 中文字元統計 OpenCC 語料分析 文本長度統計"
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
            '''檢測亂碼符號'''
            if not isinstance(text, str):
                return False
            
            # 檢查亂碼符號
            garbled_patterns = ['��', 'â', 'Ã', '�']
            garbled_count = sum(text.count(pattern) for pattern in garbled_patterns)
            
            # 如果亂碼符號占比很高，判定為亂碼
            if len(text) > 0 and garbled_count / len(text) > 0.3:
                return True
            
            # 檢查可列印字符比例
            printable = sum(1 for c in text if c.isprintable() or c in '\\n\\r\\t')
            if len(text) > 0 and printable / len(text) < 0.5:
                return True
            
            return False
        
        def collect_garbled_examples(values, max_examples=3):
            '''收集包含亂碼的文字範例'''
            examples = []
            for v in values:
                if v is not None and contains_garbled_text(str(v)):
                    text = str(v)
                    # 限制每個範例在100字以內
                    if len(text) > 100:
                        text = text[:100] + '...'
                    examples.append(text)
                    if len(examples) >= max_examples:
                        break
            return examples
        
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
        
        def calculate_tc_char_count(text):
            '''計算繁體中文字元數量'''
            if not isinstance(text, str) or len(text) == 0:
                return 0
            chinese_chars = len(re.findall(r'[一-鿿]', text))
            return chinese_chars
        
        def calculate_avg_tc_chars(values):
            '''計算平均繁體中文字元數'''
            tc_counts = [calculate_tc_char_count(str(v)) for v in values if v is not None]
            if not tc_counts:
                return 0
            return round(np.mean(tc_counts), 2)
        
        def evaluate_cp_suitability(column_analysis):
            '''評估是否適合繁體中文持續預訓練（Continue Pretrain）'''
            is_tc = column_analysis['is_traditional_chinese']
            tc_avg_chars = column_analysis['traditional_chinese_avg_chars']
            non_empty = column_analysis['non_empty_ratio']
            has_garbled = column_analysis['contains_garbled_text']
            avg_length = column_analysis['length_stats']['avg']
            
            # 基本條件：繁體中文、平均字元數足夠、高非空比例、無亂碼
            if not is_tc or tc_avg_chars < 40 or non_empty < 0.7 or has_garbled:
                return False, []
            
            # 根據長度推薦持續預訓練類型
            suggestions = []
            if avg_length < 50:
                suggestions = ["補充訓練", "短文本理解", "對話訓練"]
            elif avg_length < 200:
                suggestions = ["標準訓練", "段落理解", "上下文建模"]
            elif avg_length < 1000:
                suggestions = ["深度訓練", "長文本建模", "文章理解"]
            else:
                suggestions = ["長文本專訓", "長距離依賴", "文檔級理解"]
            
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
            
            # 顯示每個欄位的樣本
            print("\\n=== 欄位樣本預覽 ===")
            for column in df.columns:
                print(f"\\n【欄位：{column}】")
                samples = df[column].dropna().head(3).tolist()
                for i, sample in enumerate(samples, 1):
                    display_text = str(sample)[:200] + '...' if len(str(sample)) > 200 else str(sample)
                    print(f"  樣本 {i}: {display_text}")
            print("=" * 80)
            
            # 觸發語意品質檢查（針對所有文字欄位）
            print("\\n開始進行語意品質檢查...")
            for column in df.columns:
                # 檢查是否為文字類型欄位
                sample_value = df[column].dropna().head(1).tolist()
                if sample_value and isinstance(sample_value[0], str):
                    samples = df[column].dropna().head(3).tolist()
                    error_msg = f"SEMANTIC_CHECK_REQUEST\\n欄位名稱：{column}\\n"
                    for i, sample in enumerate(samples, 1):
                        error_msg += f"樣本{i}：{sample}\\n"
                    raise ValueError(error_msg)
            
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
                
                # 計算平均繁體中文字元數
                tc_avg_chars = calculate_avg_tc_chars(values)
                
                # 檢測亂碼
                has_garbled = any(contains_garbled_text(str(v)) for v in values[:20])
                
                # 收集亂碼範例
                garbled_examples = []
                if has_garbled:
                    garbled_examples = collect_garbled_examples(values, max_examples=3)
                
                # 長度統計
                length_stats = calculate_length_stats(values)
                
                # 組合分析結果
                column_analysis = {
                    "column": column,
                    "is_traditional_chinese": is_tc,
                    "traditional_chinese_avg_chars": tc_avg_chars,
                    "non_empty_ratio": round(non_empty_ratio, 3),
                    "length_stats": length_stats,
                    "contains_garbled_text": has_garbled,
                    "garbled_text_examples": garbled_examples
                }
                
                # 評估 CP 適用性
                recommended, suggestions = evaluate_cp_suitability(column_analysis)
                column_analysis["recommended_for_cp"] = recommended
                column_analysis["cp_usage_suggestions"] = suggestions
                
                summary.append(column_analysis)
            
            # 輸出結果
            result = {"summary": summary}
            
            # 以表格形式呈現結果
            print("\\n" + "=" * 120)
            print("=== 資料集分析結果 ===")
            print("=" * 120)
            
            # 表格標題
            header = f"{'欄位名稱':<20} {'繁體中文':<10} {'非空比例':<10} {'平均長度':<10} {'亂碼檢測':<10} {'CP適用性':<12} {'CP訓練類型建議':<30}"
            print(header)
            print("-" * 125)
            
            # 表格內容
            for item in summary:
                col_name = item['column'][:18] + '..' if len(item['column']) > 18 else item['column']
                is_tc = '✓' if item['is_traditional_chinese'] else '✗'
                non_empty = f"{item['non_empty_ratio']:.2%}"
                avg_len = f"{item['length_stats']['avg']:.1f}"
                garbled = '✗ 有亂碼' if item['contains_garbled_text'] else '✓ 正常'
                cp_rec = '✓ 適合' if item['recommended_for_cp'] else '✗ 不適合'
                cp_usage = ', '.join(item['cp_usage_suggestions'][:3]) if item['cp_usage_suggestions'] else '-'
                cp_usage = cp_usage[:28] + '..' if len(cp_usage) > 28 else cp_usage
                
                row = f"{col_name:<20} {is_tc:<10} {non_empty:<10} {avg_len:<10} {garbled:<10} {cp_rec:<12} {cp_usage:<30}"
                print(row)
            
            print("=" * 120)
            
            # 檢查是否有繁體中文但有亂碼的欄位，印出範例
            for item in summary:
                if item['is_traditional_chinese'] and item['contains_garbled_text']:
                    column = item['column']
                    print(f"\\n⚠️  警告：欄位 '{column}' 被判定為繁體中文但包含亂碼")
                    print(f"   以下是該欄位的部分樣本供觀察：")
                    print("-" * 80)
                    
                    # 取得該欄位的值並顯示前5個非空範例
                    col_values = df[column].tolist()
                    examples = [str(v) for v in col_values if v is not None and str(v).strip()][:5]
                    
                    for i, example in enumerate(examples, 1):
                        # 截斷過長的文字
                        display_text = example[:200] + '...' if len(example) > 200 else example
                        print(f"   範例 {i}: {display_text}")
                    print("-" * 80)
            
            # 輸出完整 JSON 結果
            print("\\n=== 完整 JSON 結果 ===")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            
            return result
        
        # 使用範例
        # result = analyze_hf_dataset("username/dataset_name", split="train", num_samples=100)
        """


if __name__ == '__main__':
    analyzer = HuggingFaceDatasetAnalyzer()
    print(analyzer.get_core_function())
