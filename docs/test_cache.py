import requests
import time
import json

API_URL = "http://localhost:3000/api/query"

def test_cache():
    question = "What is Security Cooperation?"
    payload = {"question": question}
    
    print("=" * 60)
    print("CACHE TEST - First Query (Cache Miss)")
    print("=" * 60)
    
    # First query - should be cache miss
    start = time.time()
    response1 = requests.post(API_URL, json=payload)
    time1 = time.time() - start
    
    result1 = response1.json()
    cached1 = result1.get("cached", False)
    
    print(f"Time taken: {time1:.2f} seconds")
    print(f"Cached: {cached1}")
    
    # Handle different response structures
    if "response" in result1 and "answer" in result1["response"]:
        answer1 = result1["response"]["answer"]
    elif "answer" in result1:
        answer1 = result1["answer"]
    else:
        print("Response structure:", json.dumps(result1, indent=2)[:500])
        answer1 = "Unable to extract answer"
    
    print(f"Answer preview: {answer1[:100]}...")
    
    print("\n" + "=" * 60)
    print("CACHE TEST - Second Query (Cache Hit Expected)")
    print("=" * 60)
    
    # Wait a moment
    time.sleep(1)
    
    # Second query - should be cache hit
    start = time.time()
    response2 = requests.post(API_URL, json=payload)
    time2 = time.time() - start
    
    result2 = response2.json()
    cached2 = result2.get("cached", False)
    
    print(f"Time taken: {time2:.2f} seconds")
    print(f"Cached: {cached2}")
    
    if "response" in result2 and "answer" in result2["response"]:
        answer2 = result2["response"]["answer"]
    elif "answer" in result2:
        answer2 = result2["answer"]
    else:
        answer2 = "Unable to extract answer"
    
    print(f"Answer preview: {answer2[:100]}...")
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"First query:  {time1:.2f}s (cached: {cached1})")
    print(f"Second query: {time2:.2f}s (cached: {cached2})")
    if time2 > 0:
        print(f"Speed improvement: {time1/time2:.1f}x faster")
    print("=" * 60)

if __name__ == "__main__":
    test_cache()