import asyncio
import httpx
import time

URL = "http://127.0.0.1:8000/generate_pitch"
NUM_USERS = 20  # simulate 20 users

payload = {
    "script": "Our AI coach helps professionals improve their presentations by giving feedback on tone and structure.",
    "tone": "persuasive",
    "length_minutes": 2
}

async def send_request(client, i):
    try:
        response = await client.post(URL, json=payload)
        print(f"[User {i}] Status: {response.status_code} | Time: {response.elapsed.total_seconds():.2f}s")
    except Exception as e:
        print(f"[User {i}] Error: {e}")

async def main():
    async with httpx.AsyncClient(timeout=15.0) as client:
        tasks = [send_request(client, i) for i in range(NUM_USERS)]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    print(f"Sending {NUM_USERS} concurrent requests to {URL} ...")
    start = time.time()
    asyncio.run(main())
    print(f"Completed in {time.time() - start:.2f}s")
