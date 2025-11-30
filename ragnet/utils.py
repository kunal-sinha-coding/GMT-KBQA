import asyncio

MAX_CONCURRENT_CALLS = 5
semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)

def limit_concurrency():
    def decorator(func):
        async def wrapper(*args, **kwargs):
            async with semaphore:
                return await func(*args, **kwargs)
        return wrapper
    return decorator