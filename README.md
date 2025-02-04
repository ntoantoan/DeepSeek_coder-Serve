# DeepSeek Coder API

A FastAPI-based REST API that provides an OpenAI-compatible interface for the DeepSeek Coder model. This service allows you to use the DeepSeek Coder model with the same API format as OpenAI's ChatGPT API.

## Features

- OpenAI-compatible chat completions endpoint
- Streaming support for real-time responses
- Temperature and max tokens control
- Health check endpoint
- FastAPI-powered with automatic API documentation

## Requirements

- Python 3.8+
- PyTorch
- CUDA-capable GPU (recommended)
- 8GB+ GPU memory (for 1.3B model)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd deepseek-serve
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r deepseek_serve/requirements.txt
```

## Usage

### Starting the Server

Run the following command to start the API server:

```bash
cd deepseek_serve
python -m app.main
```

The server will start on `http://localhost:8000` by default.

### API Endpoints

#### Health Check
```http
GET /health
```
Returns the health status of the API.

#### Chat Completions
```http
POST /v1/chat/completions
```

Request body:
```json
{
    "model": "deepseek-coder",
    "messages": [
        {
            "role": "user",
            "content": "Write a Python function to sort a list"
        }
    ],
    "temperature": 0.7,
    "max_tokens": 4096,
    "stream": false
}
```

Parameters:
- `model` (string, required): Model identifier
- `messages` (array, required): Array of message objects with role and content
- `temperature` (float, optional): Sampling temperature (0.0 to 2.0, default: 0.7)
- `max_tokens` (integer, optional): Maximum number of tokens to generate (default: 4096)
- `stream` (boolean, optional): Enable streaming response (default: false)

### Example Usage

#### Python Client (Non-streaming)
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "deepseek-coder",
        "messages": [
            {
                "role": "user",
                "content": "Write a Python function to sort a list"
            }
        ]
    }
)
print(response.json()["choices"][0]["message"]["content"])
```

#### Python Client (Streaming)
```python
import requests
import json

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "deepseek-coder",
        "messages": [
            {
                "role": "user",
                "content": "Write a Python function to sort a list"
            }
        ],
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            if line == 'data: [DONE]':
                break
            json_data = json.loads(line[6:])
            content = json_data['choices'][0]['delta'].get('content', '')
            if content:
                print(content, end='', flush=True)
```

## API Response Format

### Non-streaming Response
```json
{
    "id": "chatcmpl-123abc...",
    "object": "chat.completion",
    "created": 1677858242,
    "model": "deepseek-coder",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Here's a function..."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 100,
        "total_tokens": 120
    }
}
```

### Streaming Response
Each chunk follows this format:
```json
{
    "id": "chatcmpl-123abc...",
    "object": "chat.completion.chunk",
    "created": 1677858242,
    "model": "deepseek-coder",
    "choices": [
        {
            "index": 0,
            "delta": {
                "content": "Here"
            },
            "finish_reason": null
        }
    ]
}
```

## Development

The project structure is organized as follows:

```
deepseek_serve/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application and endpoints
│   ├── model.py         # DeepSeek model wrapper
│   └── schemas.py       # Pydantic models for request/response
└── requirements.txt     # Project dependencies
```
