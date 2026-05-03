import asyncio
import concurrent.futures
from functools import partial
from backend.ingestion import ingest_corpus

_process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=1)

async def main():
    print("Submitting task...")
    loop = asyncio.get_running_loop()
    func = partial(ingest_corpus, False, [])
    try:
        ok = await loop.run_in_executor(_process_pool, func)
        print("Result:", ok)
    except Exception as e:
        print("Error:", repr(e))

if __name__ == "__main__":
    asyncio.run(main())
