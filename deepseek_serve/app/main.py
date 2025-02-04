import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from model import DeepSeekModel
from schemas import (
    ChatCompletionRequest, 
    ChatCompletionResponse, 
    ChatCompletionStreamResponse,
    Choice, 
    StreamChoice,
    DeltaMessage,
    Usage, 
    Message
)
import time
import uuid
import json
from typing import Iterator

app = FastAPI(title="DeepSeek Coder API")
model = DeepSeekModel()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

def stream_response(response_id: str, model_name: str, stream_iterator: Iterator[str]) -> Iterator[str]:
    for text in stream_iterator:
        chunk = ChatCompletionStreamResponse(
            id=response_id,
            created=int(time.time()),
            model=model_name,
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(content=text),
                    finish_reason=None
                )
            ]
        )
        yield f"data: {json.dumps(chunk.model_dump())}\n\n"
    
    # Send the final chunk with finish_reason
    final_chunk = ChatCompletionStreamResponse(
        id=response_id,
        created=int(time.time()),
        model=model_name,
        choices=[
            StreamChoice(
                index=0,
                delta=DeltaMessage(),
                finish_reason="stop"
            )
        ]
    )
    yield f"data: {json.dumps(final_chunk.model_dump())}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    response_id = f"chatcmpl-{str(uuid.uuid4())}"

    if request.stream:
        # Return streaming response
        stream_iterator = model.generate_stream(
            messages=request.messages,
            max_length=request.max_tokens
        )
        return StreamingResponse(
            stream_response(response_id, request.model, stream_iterator),
            media_type="text/event-stream"
        )

    # Non-streaming response
    generated_text = model.generate(
        messages=request.messages,
        max_length=request.max_tokens
    )
    response_message = Message(
        role="assistant",
        content=generated_text
    )
    
    prompt_tokens = len(" ".join([m.content for m in request.messages]).split())
    completion_tokens = len(generated_text.split())
    total_tokens = prompt_tokens + completion_tokens
    
    response = ChatCompletionResponse(
        id=response_id,
        created=int(time.time()),
        model=request.model,
        choices=[
            Choice(
                index=0,
                message=response_message,
                finish_reason="stop"
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        )
    )
    
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
