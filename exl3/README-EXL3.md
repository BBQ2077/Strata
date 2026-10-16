# Strata EXL3 版（純 Windows 原生）

> **實驗性質。** 適合整個放得進顯存的 EXL3 MoE 模型（例如 Qwen3.6-35B-A3B）。Qwen3.8-Flash-Next 請用 Strata 原版引擎，它的專家調度（顯卡／RAM／SSD）專為這個模型設計。

把 Strata 的伺服器（OpenAI／Anthropic API、工具呼叫、思考解析、網頁介面、Monitor）接到 ExLlamaV3（EXL3）引擎上跑。不用 WSL，不用 Linux，也不用編譯 C++／HIP。

## 架構

```
用戶端 / 網頁  ──HTTP 8080──>  serve/server.py（Strata 原版伺服器，.venv-exl3）
                                    │ stdin/stdout 逐行協定（GEN / T / DONE ...）
                                    ▼
             RUNTIMES/exl3-gfx1201/python.exe runtime.py script strata_exl3_engine.py --serve ...
                                    │
                                    ▼
                         ExLlamaV3 Generator（ROCm gfx1201，q4 KV，遞迴狀態快取）
```

Strata 的伺服器本來就把引擎當成一個子程序，用文字協定溝通（`serve/server.py` 的 `StrataEngine`）。所以 EXL3 版不用改寫伺服器，只要寫一個講同一套協定的引擎：`exl3/strata_exl3_engine.py`。

## 檔案

| 檔案 | 用途 |
| --- | --- |
| `exl3/strata_exl3_engine.py` | EXL3 引擎：讀 `GEN`／`STOP`／`QUIT`，回 `INFO`／`READY`／`RESUME`／`PP`／`T`／`DONE`／`ERR` |
| `exl3/hf_tokenizer.py` | 從 EXL3 模型資料夾的 `tokenizer.json` 產生 Strata 格式的 tokenizer（vocab／merges／token_type／chat_template） |
| `exl3/strata-exl3.json` | 執行設定：65536 context、q4 KV、遞迴快取 4 GB、chunk 2048 |
| `exl3/START-EXL3.bat` | 一鍵：建 venv → 複製引擎進 runtime → 產生 tokenizer → 啟動伺服器 |
| `exl3/requirements-exl3.txt` | 伺服器端套件（不含 cmake／ninja） |
| `exl3/test_exl3_port.py` | 離線檢查（不用 GPU） |

`serve/server.py` 改了三處：設定檔可以帶 `"launcher"`（用 Python 啟動 .py 引擎）；`backend: exl3` 時拿掉 top_k 64 上限；停止 token 改從實際 tokenizer 讀取。

## 探查到的問題與處理

| # | 問題 | 影響 | 處理 |
| --- | --- | --- | --- |
| 1 | Strata 引擎是 C++／HIP 寫的，專為 Qwen3.8-Flash-Next（PLE、24,576 專家、CPU／GPU 分工）設計，只吃 GGUF 封包 | EXL3 模型無法直接載入 | 換掉引擎，保留伺服器：新寫 `strata_exl3_engine.py`，講 Strata 原本的 stdin/stdout 協定 |
| 2 | 伺服器只能把 `exe` 當執行檔啟動 | .py 引擎跑不起來 | 設定檔新增 `"launcher"`（`python.exe -u runtime.py script`），沒設時行為跟原版一樣 |
| 3 | runtime 的 `script` 模式只能跑 runtime 資料夾內的檔案 | 引擎放在 Strata 資料夾會被拒絕 | `START-EXL3.bat` 每次啟動前把引擎複製進 runtime 資料夾 |
| 4 | runtime 和 ExLlamaV3 載入時會往 stdout 印東西 | 雜訊會被當成協定訊息 | 引擎一開始就把真正的 stdout 複製一份給協定專用，其他輸出全部導到 stderr（會進引擎 log） |
| 5 | Strata 自己做 tokenize，需要 `vocab.json`／`merges.txt`／`token_type.json`；原版是從 GGUF 抽出來的 | EXL3 模型資料夾只有 HF 格式 | `hf_tokenizer.py` 轉換：special=true 是 CONTROL(3)，special=false（`<think>`、`<tool_call>`）是 USER_DEFINED(4)，補位 id 是 UNUSED(5)。實測 248,320 個 token，跟 HF tokenizers 編碼結果完全一致 |
| 6 | pre-tokenizer 正規式寫法不同（HF 用 `(?i:'s\|...)`，Strata 用 `'[sS]\|...`） | 直接比對字串會誤判成不相容 | 兩種寫法都接受；其他切分方式直接拒絕，不會產生錯的 tokenizer |
| 7 | 伺服器把 `top_k` 限制在 64（C++ 引擎的限制），`top_k=0`（關閉）也被改成 64 | EXL3 上取樣跟設定不一樣 | `backend: exl3` 時原值傳入 |
| 8 | 停止 token 寫死成 Flash-Next 的 id（248044／248046） | 換別的詞表會停不下來 | 從實際載入的 tokenizer 讀 `<\|endoftext\|>`／`<\|im_end\|>` |
| 9 | 多輪對話整段重算 | 61K 前文每輪重讀 66 秒 | EXL3 分頁前綴快取＋遞迴狀態檢查點（`-rcs 4`，系統 RAM）。重用了多少 token 會用 `RESUME` 和 `DONE` 回報給 Strata 的 Monitor |
| 10 | Strata 專屬參數（`--slots`、`--expert-profile`、`--layer-split`、`--vision`、`--vram-elastic`…） | 傳給 EXL3 會報錯 | 引擎直接略過，並在 log 列出略過了哪些 |
| 11 | Strata 專屬指令：`GENI`（圖片）、`VRAM`（彈性專家快取）、`SAVE`／`RESTORE`（磁碟 session） | EXL3 沒有對應功能 | 回 `ERR`／`SERR invalid`，伺服器會照原版邏輯回 4xx 給用戶端，不會卡住 |
| 12 | 伺服器不是 FastAPI，是標準庫的 `http.server` | 依賴很少 | 伺服器用獨立 `.venv-exl3`（Python 3.11），引擎用 EXL3 runtime 自己的 Python，兩邊互不污染 |
| 13 | 雙開／殘留子程序 | 殺掉外層後 python 子程序還活著 | Strata 原版就有 `serve/winjob.py`（Windows Job Object，KILL_ON_JOB_CLOSE），伺服器結束時引擎和它的子程序一起被收掉 |

## 目前不支援的功能

- 圖片輸入（`GENI`）：Strata 的視覺編碼器接在它自己的 C++ 引擎上，EXL3 版要另外接 ExLlamaV3 的 vision tower。
- 磁碟 session（`SAVE`／`RESTORE`）、VRAM 彈性釋放（`/v1/vram`）、多 GPU 副本。
- Qwen3.8-Flash-Next 本身：ExLlamaV3 還沒有這個架構（PLE＋超大 MoE），也沒有 EXL3 量化版。EXL3 版跑的是 ExLlamaV3 支援的模型，例如 `Qwen3.6-35B-A3B`。

## 啟動

先指定兩個資料夾：用環境變數，或把 `exl3\local.example.bat` 複製成 `exl3\local.bat` 再改（`local.bat` 不進 git）。

| 變數 | 內容 |
|---|---|
| `STRATA_EXL3_RUNTIME` | EXL3 runtime 資料夾（裡面有 `python\python.exe`、`runtime.py`） |
| `STRATA_EXL3_MODEL` | EXL3 模型資料夾（裡面有 `config.json`、`tokenizer.json`） |

```bat
exl3\START-EXL3.bat
```

第一次會建 `.venv-exl3`、產生 `pack\exl3-<模型資料夾名稱>\tokenizer`，然後在 `http://127.0.0.1:8080` 開服務。埠號和 GGUFRun 的 18435／18436 分開。`strata-exl3.json` 裡的 `${STRATA_EXL3_RUNTIME}`、`${STRATA_EXL3_MODEL}`、`${STRATA_EXL3_TOKENIZER}` 在啟動時展開，log 寫在 Strata 資料夾的 `strata-exl3.log`。

## 離線檢查

```bat
.venv-exl3\Scripts\python.exe exl3\test_exl3_port.py --model <EXL3 模型資料夾>
```
