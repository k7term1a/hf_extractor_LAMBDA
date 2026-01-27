from .knw import knw


class HuggingFaceDatasetAnalyzer(knw):
    def __init__(self):
        super().__init__()
        self.name = "HF資料集分析器"
        self.description = "分析 Hugging Face 資料集 繁體中文 持續預訓練 Continue Pretrain CP 適用性評估 資料集品質檢測 展示資料集內容 儲存 parquet"
        self.core_function = "analyze_and_save_dataset"
        self.mode = 'full'

    def analyze_and_save_dataset(self):
        return """
        # Hugging Face 資料集分析與儲存完整程式碼
        from datasets import load_dataset
        import pandas as pd
        import os
        from dotenv import load_dotenv
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        # 載入環境變數
        load_dotenv()
        
        # 設置 Hugging Face token（從環境變數讀取）
        hf_token = os.environ.get('HF_KEY') or os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
        if hf_token:
            print(f"✓ 已找到 Hugging Face token (長度: {len(hf_token)})")
        else:
            print("⚠ 未找到 HF_KEY 環境變數，某些私有資料集可能無法存取")
            print("   請確認 .env 文件中已設置 HF_KEY")
        
        def load_and_display_dataset(dataset_name, split="train", num_samples=100):
            '''載入並展示資料集'''
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
                samples = df[column].dropna().head(5).tolist()
                for i, sample in enumerate(samples, 1):
                    display_text = str(sample)
                    # 如果樣本過長，只顯示前500字符
                    if len(display_text) > 500:
                        display_text = display_text[:500] + '... (已截斷)'
                    print(f"  樣本 {i}: {display_text}")
            print("=" * 80)
            
            return df
        
        def save_approved_fields_to_parquet(df, field_name, dataset_name, output_dir="./output", num_samples=None):
            '''
            將被 Inspector 認可的欄位儲存為 parquet 格式
            
            參數：
            - df: DataFrame 包含資料
            - field_name: 要儲存的欄位名稱
            - dataset_name: 資料集名稱（用於檔案命名）
            - output_dir: 輸出目錄
            - num_samples: 要儲存的樣本數量，None 表示全部
            '''
            # 創建輸出目錄
            os.makedirs(output_dir, exist_ok=True)
            
            # 清理資料集名稱用於檔案命名
            safe_dataset_name = dataset_name.replace('/', '_').replace('\\\\', '_')
            safe_field_name = field_name.replace('/', '_').replace('\\\\', '_')
            
            # 檔案名稱
            filename = f"{safe_dataset_name}_{safe_field_name}_cp_data.parquet"
            filepath = os.path.join(output_dir, filename)
            
            # 準備資料
            if num_samples is not None:
                data_to_save = df[field_name].head(num_samples)
            else:
                data_to_save = df[field_name]
            
            # 過濾掉空值
            data_to_save = data_to_save.dropna()
            
            # 建立 DataFrame with id and text schema
            output_df = pd.DataFrame({
                'id': range(len(data_to_save)),
                'text': data_to_save.values
            })
            
            # 儲存為 parquet
            table = pa.Table.from_pandas(output_df)
            pq.write_table(table, filepath)
            
            print(f"✓ 已儲存 {len(output_df)} 筆資料至：{filepath}")
            print(f"  Schema: {{'id': int, 'text': string}}")
            
            return filepath
        
        def trigger_inspector_check(df, text_columns):
            '''觸發 Inspector 進行語意檢查（一次檢查所有欄位）'''
            error_msg = "SEMANTIC_CHECK_REQUEST\\n\\n"
            
            for field_name in text_columns:
                samples = df[field_name].dropna().head(5).tolist()
                error_msg += f"=== 欄位名稱：{field_name} ===\\n"
                for i, sample in enumerate(samples, 1):
                    # 限制每個樣本長度避免訊息過長
                    sample_text = str(sample)
                    if len(sample_text) > 500:
                        sample_text = sample_text[:500] + '... (已截斷)'
                    error_msg += f"樣本{i}：{sample_text}\\n"
                error_msg += "\\n"
            
            raise ValueError(error_msg)
        
        # 主要分析流程
        def analyze_hf_dataset(dataset_name, split="train", num_samples=100):
            '''
            分析 Hugging Face 資料集
            
            步驟：
            1. 載入並展示資料集
            2. 觸發 Inspector 檢查（透過 raise ValueError）
            3. 根據 Inspector 回饋儲存認可的欄位
            '''
            # 步驟 1：載入並展示資料集
            df = load_and_display_dataset(dataset_name, split, num_samples)
            
            # 步驟 2：觸發 Inspector 檢查（針對所有文字欄位）
            print("\\n開始進行 CP 適用性檢查...")
            text_columns = []
            for column in df.columns:
                # 檢查是否為文字類型欄位
                sample_value = df[column].dropna().head(1).tolist()
                if sample_value and isinstance(sample_value[0], str):
                    text_columns.append(column)
            
            # 逐一觸發檢查（系統會在每次 ValueError 後由 Inspector 處理）
            if text_columns:
                print(f"找到 {len(text_columns)} 個文字欄位需要檢查：{text_columns}")
                # 一次性檢查所有文字欄位
                trigger_inspector_check(df, text_columns)
            else:
                print("⚠ 未找到文字類型欄位")
                return None
        
        # 使用範例
        # Step 1: 初次執行會觸發 Inspector 檢查
        # result = analyze_hf_dataset("username/dataset_name", split="train", num_samples=100)
        
        # Step 2: Inspector 回饋後，使用以下代碼儲存認可的欄位
        # inspector_results = {
        #     'field_name': {
        #         'approved': True,  # Inspector 判斷為適合
        #         'reason': '主要為繁體中文，語意清晰',
        #         'suggestions': ['長文本理解', '文章生成']
        #     }
        # }
        # 
        # for field, result in inspector_results.items():
        #     if result['approved']:
        #         save_approved_fields_to_parquet(
        #             df=df,
        #             field_name=field,
        #             dataset_name='username/dataset_name',
        #             num_samples=10  # 測試用，改為 None 儲存全部
        #         )
        """


if __name__ == '__main__':
    analyzer = HuggingFaceDatasetAnalyzer()
    print(analyzer.get_core_function())
