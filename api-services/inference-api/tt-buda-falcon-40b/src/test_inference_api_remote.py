import os
import threading
import queue
import time

from reprint import output
import requests

from inference_config import inference_config
from tt_models.falcon40b.multilineoutput import MultiLineOutput

# DEPLOY_URL = "http://127.0.0.1"
# API_BASE_URL = f"{DEPLOY_URL}:{inference_config.backend_server}"
DEPLOY_URL = "https://falcon-api--tenstorrent-playground.workload.tenstorrent.com"
API_BASE_URL = f"{DEPLOY_URL}"
API_URL = f"{API_BASE_URL}/predictions/falcon40b"
HEALTH_URL = f"{API_BASE_URL}/get-health"

headers = {"Authorization": os.environ.get("AUTHORIZATION")}

output_queue = queue.Queue()

prompts = [
    "Write a function that takes two lists and returns a list that has alternating elements from each input list.",
    "Write a plot summary for a comedic novel involving Elon Musk and sea travel.",
    "Arianna has 12 chocolates more than Danny. Danny has 6 chocolates more than Robbie. Arianna has twice as many chocolates as Robbie has. How many chocolates does Danny have?"
    "Does the United States use Celsius or Fahrenheit?",
    "Joe Biden is the Nth president of the United States. What is N?",
    "Write a poem about the sun and moon.",
    "Create a 2 turn conversation between a customer and a grocery store clerk - that is, 2 per person. Then tell me what they talked about.",
    "Write a fun, short story about a young duckling that got lost in a forest but found love.",
    "List the ways that a batter can reach first base in major league baseball.",
    "I am going to a theme park called Cedar Point in Ohio this summer and need a list of the best roller coasters there.",
    "Give me the answer to this trivia question: what team was LeBron James playing for when he broke the NBA all time scoring record?",
    "I have these ingredients in my fridge: flour, eggs, milk, ham, and spinach. Generate a recipe for a meal I can make.",
    "Create a bash script that calculates the sum of all numbers from 1 to 100.",
    "List some interesting things to do in Toronto.",
    "What is the largest heading tag in HTML?",
    "Pitch me a premise for a comedy movie.",
    "Is the continent of Antarctica located at the north or South Pole?",
    "Generate a list of hobbies for a teenager.",
    "When was penicillin first used on humans?",
    "Create a Python script to find the sum of all numbers ranging from k to n inclusive.",
    "Construct a haiku about flowers in a meadow.",
    "What insects chew through wood and can destroy your house?",
    "Explain to me how a rocket goes into space like Im a 5 year old.",
    "Why should you cover your mouth when you cough?",
    "Compose a story, in no more than 3 sentences, about time travel and aliens.",
    "Desk jobs require writing a lot of emails, so it isnt surprising we get tired of repeating ourselves. Come up with several synonyms for the given word. Input: Sincerely",
    "Come up with some search queries on google about coding stuff.",
    "Design a skill assessment questioner for R (Programming Language).",
    "Make a list of the most popular podcasts.",
    "Please suggest a few papers to consider based on the topic of Machine Learning",
    "Create a daily itinerary based on the given information. Input: Our family is looking for a 9-day Morocco trip that has light to moderate activity levels.",
    "Design a template table for keeping track of all subscriptions.",
    "How do you say 'good evening' in French.",
]


def test_api_call(
    prompt="Where should I go in Austin when I visit?", print_output=True, idx=0
):
    # set API prompt and optional parameters
    json_data = {
        "text": prompt,
        "temperature": 1,
        "top_k": 1,
        "top_p": 0.9,
        "max_tokens": 128,
        "stop_sequence": None,
        "return_prompt": None,
    }
    # using requests stream=True, make sure to set a timeout
    response = requests.post(
        API_URL, json=json_data, headers=headers, stream=True, timeout=35
    )
    # Handle chunked response
    if response.headers.get("transfer-encoding") == "chunked":
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            # Process each chunk of data as it's received
            if print_output:
                print(chunk)
            else:
                output_queue.put((idx, chunk))
    else:
        # If not chunked, you can access the entire response body at once
        print(response.text)


def test_api_call_multiline(print_stats=False):
    threads = []
    tokens_generated = [0] * len(prompts)

    start_time = time.time()
    for idx, prompt in enumerate(prompts):
        print(f"Prompt {idx}: {prompt}")
        time.sleep(0.01)
        thread = threading.Thread(target=test_api_call, args=[prompt, False, idx])
        threads.append(thread)
        thread.start()

    print(f"\nGenerating responses from: {API_URL}\n")
    with output(output_type="list", initial_len=len(prompts), interval=100) as out:
        mlo = MultiLineOutput(len(prompts), out)
        # loop to continuously update the lines with new text
        while any([t.is_alive() for t in threads]) or not output_queue.empty():
            idx, text_token = output_queue.get()
            tokens_generated[idx] += 1
            mlo.set_label(idx, f"Response {idx}: n_tokens=[{tokens_generated[idx]}]: ")
            mlo.append_line(idx, text_token)
    end_time = time.time()
    if print_stats:
        print("\nNOTE: lower bound stats, some user rows complete before others.")
        print(f"Total tokens_generated: {sum(tokens_generated)}")
        print(f"tokens/second: {sum(tokens_generated) / (end_time - start_time)}")
        print(
            f"tokens/second/user: {(sum(tokens_generated) / (end_time - start_time)) / 32}"
        )
    assert sum(tokens_generated) > 0


def test_api_call_speed():
    threads = []
    tokens_generated = [0] * len(prompts)
    same_prompts = [prompts[0]] * 32
    start_time = time.time()
    for idx, prompt in enumerate(same_prompts):
        print(f"Prompt {idx}: {prompt}")
        time.sleep(0.005)
        thread = threading.Thread(target=test_api_call, args=[prompt, False, idx])
        threads.append(thread)
        thread.start()

    print(f"\nGenerating responses from: {API_URL}\n")
    # loop to continuously update the lines with new text
    while any([t.is_alive() for t in threads]) or not output_queue.empty():
        idx, _ = output_queue.get()
        tokens_generated[idx] += 1

    end_time = time.time()
    print(f"\nTotal tokens_generated: {sum(tokens_generated)}")
    print(f"tokens/second: {sum(tokens_generated) / (end_time - start_time)}")
    print(
        f"tokens/second/user: {(sum(tokens_generated) / (end_time - start_time)) / 32}"
    )
    assert sum(tokens_generated) > 0


def test_get_health():
    response = requests.get(HEALTH_URL, headers=headers, timeout=35)
    assert response


if __name__ == "__main__":
    test_api_call_multiline()
