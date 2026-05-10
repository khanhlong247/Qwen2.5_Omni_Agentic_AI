# 🎙️ Qwen2.5-Omni Audio Insurance Agent

An advanced multimodal AI agent system based on the **Qwen2.5-Omni (7B)** model. The agent can understand customer requests via audio, automatically execute calling tools to query insurance databases, and respond directly with natural-sounding voice.

## 🏗️ Project Architecture (Layered Architecture)

The project applies a layered architecture (Clean Architecture) to separate the AI ​​model logic, business logic, and data retrieval:

```text
QWEN2.5_AUDIO/
├── data/ # Contains input audio files from the client (.wav, .mp3)
├── output/ # Contains the Agent's audio response (.wav)
├── models/ # Model weights and spk_dict.pt
├── src/ # The entire main source code of the application
│ ├── __init__.py
│ ├── core/ # Core layer: Defines the Qwen model architecture
│ │ ├── modeling_qwen2_5_omni.py
│ │ ├── configuration_qwen2_5_omni.py
│ │ └── processing_qwen2_5_omni.py
│ ├── repositories/ # Data Layer: Database Communication via ORM
│ │ └── insurance_manager.py
│ ├── services/ # Business Layer: AI Inference Workflow Coordination
│ │ └── audio_processor.py
│ ├── schemas/ # Definition Layer: Tool Call Structure (JSON/XML)
│ │ └── tools_schema.py
│ └── utils/ # Digital Signal Processing (DSP) Utility Functions
├── .env # Environment Variables (Database URL, API Keys)
├── main.py # Entry Point - Launches the Conversation Flow Agent
├── pyproject.toml # Manages Dependencies (UV/Pip)

└── uv.lock
```

# ✨ Key Features

- **End-to-End Audio Pipeline:** Direct Audio-to-Audio processing flow (without separate ASR/TTS intermediate steps).

- **Intelligent Routing:** Automatically navigates between tools based on user intent.

- **Defensive Parsing:** Flexible Parser for processing JSON results from LLM, preventing hallucination errors.

- **Native Talker Integration:** Uses Qwen's built-in Token2Wav module to generate expressive voiceovers.

# 🛠️ Installation Guide

### 1. Requirements

- Python 3.10 or later.

- UV (recommended) or Pip.

- RAM: Minimum 16GB (32GB+ if running on a CPU).

### 2. Library Installation

Use UV to quickly set up the environment:

```
uv sync
```

### 3. Model Preparation

Download the Qwen2.5-Omni payloads and place them in the `./models` directory. Ensure all configuration files and the spk_dict.pt file are present.

# 🚀 Running the Program

Configure the database in the `.env` file or directly in `insurance_manager.py`.

Place the client's audio files in the `./data` directory.

### Run the application:

```
uv run python main.py
```