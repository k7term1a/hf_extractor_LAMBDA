# HF Dataset Extractor for Traditional Chinese CP

[![LAMBDA Base](https://img.shields.io/badge/Based%20on-LAMBDA-blue)](https://github.com/AMA-CMFAI/LAMBDA)
[![Python](https://img.shields.io/badge/Python-3.10-green)](https://www.python.org/)

## 專案簡介

本專案基於 [LAMBDA](https://github.com/AMA-CMFAI/LAMBDA) 系統，專門用於分析 Hugging Face 資料集是否適合繁體中文持續預訓練（Continue Pretrain, CP）。

透過 AI Agent 協作機制：
- **Programmer Agent**: 負責載入資料集並展示欄位內容
- **Inspector Agent**: 專業評估每個欄位是否適合繁體中文 CP 訓練
- **自動儲存**: 將認可的欄位自動儲存為 parquet 格式（Schema: `{id, text}`）

## 📺 系統示範

https://github.com/user-attachments/assets/demo.mp4

> 完整示範影片展示了從輸入資料集名稱到自動分析並儲存的完整流程

## ⚠️ 重要提醒

> **建議一次只分析一個資料集**
> 
> 為確保分析品質和系統穩定性，強烈建議每次只提供一個 Hugging Face 資料集連結進行分析。
> 
> 多個資料集請分批處理。

## 主要功能

### **自動處理 Subset 和 Split**
- 自動檢測並選擇有效的資料集 subset
- 智能選擇有內容的 split（優先 train，否則選第一個可用的）
- 確保載入的資料非空

### **一次性檢查所有欄位**
- 展示所有文字欄位的樣本內容（5-10 筆）
- Inspector 一次性評估所有欄位的 CP 適用性
- 提供詳細的判斷理由和訓練建議

### **自動儲存認可欄位**
- 將適合 CP 的欄位儲存為 parquet 格式
- Schema: `{id: int, text: string}`
- 輸出至 `./output/` 目錄

## 使用方式

### 1. 安裝依賴

#### 使用 pip
```bash
conda activate lambda
pip install -r requirements.txt
```

### 2. 啟動系統
```bash
python lambda_app.py
```

### 3. 在對話界面中輸入
```
請分析 Hugging Face 資料集：username/dataset_name
```

### 4. 系統會自動
   - 載入資料集（處理 subset 和 split）
   - 展示所有欄位的樣本內容
   - 觸發 Inspector 進行 CP 適用性評估
   - 儲存認可的欄位為 parquet

> **📝 注意**：目前儲存的資料為**前 10 行**（測試模式）。

## 輸出格式

```
output/
  ├── dataset_name_field1_cp_data.parquet
  ├── dataset_name_field2_cp_data.parquet
  └── ...
```

每個 parquet 檔案包含：
- `id`: 序號（從 0 開始）
- `text`: 欄位內容

---

<div align="center">
  
# LAMBDA - LArge Model-based Data Analysis System
[![Docs](https://img.shields.io/badge/Docs-Online-blue)](https://ama-cmfai.github.io/LAMBDA-Docs/#/)
[![Project](https://img.shields.io/badge/Project-Webpage-brightgreen)](https://www.polyu.edu.hk/ama/cmfai/lambda.html)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/pdf/2407.17535)
[![MacOS](https://img.shields.io/badge/Download-macOS-black?logo=apple)](https://github.com/AMA-CMFAI/LAMBDA/releases/download/app/LAMBDA-MacOS-beta-v0.0.2.zip)
[![Windows](https://img.shields.io/badge/Download-Windows-blue?logo=windows)](https://github.com/AMA-CMFAI/LAMBDA/releases/download/app/LAMBDA-Windows-beta-v0.0.2.zip)

</div>

<body>
<!-- <img src="https://github.com/user-attachments/assets/df454158-79e4-4da4-ae03-eb687fe02f16" style="width: 80%"> -->
<!-- <p align="center">
  <img src="https://github.com/user-attachments/assets/6f6d49ef-40b7-46f2-88ae-b8f6d9719c3a" style="width: 600px;">
  ![lambda_mix](https://github.com/user-attachments/assets/db5574aa-9441-4c9d-b44d-3b225d11e0cc)
</p> -->
  
![LAMBDA_mix_250710](https://github.com/user-attachments/assets/5cdc113b-7d26-4328-8911-d421081f98ce)


We introduce **LAMBDA**, a novel open-source, code-free multi-agent data analysis system that harnesses the power of large models. LAMBDA is designed to address data analysis challenges in complex data-driven applications through the use of innovatively designed data agents that operate iteratively and generatively using natural language.

## News
- LAMBDA App for macOS and Windows has been released. Details can be found in [Released](https://github.com/AMA-CMFAI/LAMBDA/releases/tag/app). (Hint: There are some problems with the kernel installation in the APP. You should run `ipython kernel install --name lambda --user` to install the kernel in advance.)
- [Docs site](https://ama-cmfai.github.io/LAMBDA-Docs/#/) is available!

## Key Features

- **Code-Free Data Analysis**: Perform complex data analysis tasks through human language instruction.
- **Multi-Agent System**: Utilizes two key agent roles, the programmer and the inspector, to generate and debug code seamlessly.
- **User Interface**: This includes a robust user interface that allows direct user intervention in the operational loop.
- **Model Integration**: Flexibly integrates external models and algorithms to cater to customized data analysis needs.
- **Automatic Report Generation**: Concentrate on high-value tasks, rather than spending time and resources on report writing and formatting.
- **Jupyter Notebook Exporting**: Export the code and the results to Jupyter Notebook for reproduction and further analysis flexibly.

## Getting Started
### Installation
First, clone the repository.

```bash
git clone https://github.com/AMA-CMFAI/LAMBDA.git
cd LAMBDA
```

Then, we recommend creating a [Conda](https://docs.conda.io/en/latest/) environment for this project and installing the dependencies by following the commands:
```bash
conda create -n lambda python=3.10
conda activate lambda
```

Then, install the required packages:
```bash
pip install -r requirements.txt
```

Next, you should install the Jupyter kernel to create a local Code Interpreter:
```bash
ipython kernel install --name lambda --user
```

### Configuration to Easy Start
1. To use the Large Language Models, you should have an API key from [OpenAI](https://openai.com/api/pricing/) or other companies. Besides, we support OpenAI-Style interface for your local LLMs once deployed, available frameworks such as [Ollama](https://ollama.com/), [LiteLLM](https://docs.litellm.ai/docs/), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).
> Here are some products that offer free APIkeys for your reference: [OpenRouter](https://openrouter.ai/) and [SILICONFLOW](https://siliconflow.cn/)
2. Set your API key, models and working path in the config.yaml:
```bash
#================================================================================================
#                                       Config of the LLMs
#================================================================================================
conv_model : "gpt-4.1-mini" # Choose the model you want to use. We highly recommned using the advanced model.
programmer_model : "gpt-4.1-mini" 
inspector_model : "gpt-4.1-mini"
api_key : "sk-xxxxxxx" # The API Keys you buy.
base_url_conv_model : 'https://api.openai.com/v1' # The base url from the provider.
base_url_programmer : 'https://api.openai.com/v1'
base_url_inspector : 'https://api.openai.com/v1'


#================================================================================================
#                                       Config of the system
#================================================================================================
streaming : True
project_cache_path : "cache/conv_cache/" # Local cache path
max_attempts : 5 # The max attempts of self-correcting
max_exe_time: 18000 # The maximum time for the execution

#knowledge integration
retrieval : False # Whether to start a knowledge retrieval. If you don't create your knowledge base, you should set it to False
```


Finally, run the following command to start the LAMBDA with GUI:
```bash
python lambda_app.py
```


## Demonstration Videos

The performance of LAMBDA in solving data science problems is demonstrated in several case studies, including:
- **[Data Analysis](https://www.polyu.edu.hk/ama/cmfai/files/lambda/lambda.mp4)**
- **[Integrating Human Intelligence](https://www.polyu.edu.hk/ama/cmfai/files/lambda/knw.mp4)**
- **[Education](https://www.polyu.edu.hk/ama/cmfai/files/lambda/LAMBDA_education.mp4)**


## Planning Works
- [ ] Create a Logger for log.
- [ ] Pre-installation of popular packages in the kernel.
- [ ] Replace Gradio UI with OpenWebUI.
- [ ] Refactor the Knowledge Integration and Knowledge base module by ChromaDB.
- [ ] Add a Docker image for easier use.
- [x] Docsite.


## Updating History
See [Docs site](https://ama-cmfai.github.io/LAMBDA-Docs/#/).


## Related Works
If you are interested in Data Agent, you can take a look at :
- Our survey paper [[A Survey on Large Language Model-based Agents for Statistics and Data Science]](https://www.arxiv.org/pdf/2412.14222)
- and a reading list: [[Paper List of LLM-based Data Science Agents]](https://github.com/Stephen-SMJ/Reading-List-of-Large-Language-Model-Based-Data-Science-Agent)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



## Acknowledgements

Thank the contributors and the communities for their support and feedback.

---

> If you find our work useful in your research, consider citing our paper by:



```bash
@article{sun2025lambda,
  title={Lambda: A large model based data agent},
  author={Sun, Maojun and Han, Ruijian and Jiang, Binyan and Qi, Houduo and Sun, Defeng and Yuan, Yancheng and Huang, Jian},
  journal={Journal of the American Statistical Association},
  pages={1--13},
  year={2025},
  publisher={Taylor \& Francis}
}

@article{sun2025survey,
  title={A survey on large language model-based agents for statistics and data science},
  author={Sun, Maojun and Han, Ruijian and Jiang, Binyan and Qi, Houduo and Sun, Defeng and Yuan, Yancheng and Huang, Jian},
  journal={The American Statistician},
  pages={1--14},
  year={2025},
  publisher={Taylor \& Francis}
}
```
## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=AMA-CMFAI/LAMBDA&type=Date)](https://www.star-history.com/#AMA-CMFAI/LAMBDA&Date)
</body>
