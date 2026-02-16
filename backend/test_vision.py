import asyncio
import websockets
import json
import base64
from PIL import Image
import io

# 1. Create a dummy red image
img = Image.new('RGB', (100, 100), color = 'red')
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='PNG')
base64_img = "data:image/png;base64," + base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

async def test_vision():
    uri = "ws://localhost:8000/ws/omni"
    async with websockets.connect(uri) as websocket:
        print("Connected to WebSocket.")

        # 1. Send Screen Frame
        print("Sending Red Box Image...")
        await websocket.send(json.dumps({
            "type": "screen_frame",
            "image": base64_img
        }))
        await asyncio.sleep(1) # Wait for update

        # 2. Ask Question
        print("Asking: 'Use the analyze_screen tool...'")
        await websocket.send(json.dumps({
            "type": "text_input",
            "text": "Use the analyze_screen tool to look at the screen and tell me what color is the box. do not guess."
        }))

        # 3. Listen for response
        print("Listening for response...")
        while True:
            response = await websocket.recv()
            data = json.loads(response)
            if data['type'] == 'agent_token':
                print(data['text'], end="", flush=True)
            elif data['type'] == 'agent_response_end':
                print("\n[END]")
                break

if __name__ == "__main__":
    asyncio.run(test_vision())
