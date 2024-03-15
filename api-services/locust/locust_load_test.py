import os
import json

from locust import HttpUser, task, between


TEST_PROMPTS = [
    "Tenstorrent is an AI startup whose RISC-V hardware aims to define a new spatial computing platform for the next century.",
    "It was the best of times, it was the worst of times",
    "I like to think (and the sooner the better!) of a cybernetic meadow",
    "We the People of the United States,",
    "Katherine Johnson (August 26, 1918 - February 24, 2020) was an African-American",
    "Knock, knock. Who's there?",
    "Count to a hundred: 1 2 3 4 5 6 7 8 9 10 11 12 13",
    "Not like the brazen giant of Greek fame,",
    "Roses are red, violets are blue,",
    "Albert Einstein (14 March 1879 - 18 April 1955) was a German-born",
    "The journey of a thousand miles",
    "When I find myself in times of trouble",
    "Shall I compare thee to a summer's day? Thou art more",
    "Rachel Carson (May 27, 1907 - April 14, 1964) was an American marine",
    "Two roads diverged in a yellow wood,",
    "Save tonight and fight the break of dawn / come",
    "The first thousand digits of PI: 3.14159265358979323846",
    "Thirty days hath September, April, June,",
    "If you want to live a happy life",
    "Ada Lovelace (10 December 1815 - 27 November 1852) was an English",
    "Call me Ishmael. Some years ago",
    "The true sign of intelligence is not knowledge",
    "Consider the sequence of prime numbers: 2, 3, 5, 7",
    "Fibonacci sequence unfurls like a mathematical nautilus: 0, 1, 1, 2, 3",
    "Once upon a time, in a land full of dragons",
    "A duck walks into a store and asks the",
    "I heard there was a secret chord",
    "It is a truth universally acknowledged, that a single man",
    "Shakespeare, William (bapt. 26 April 1564 - 23 April 1616) was an English",
    "The quality of mercy is not strained",
    "The Deliverator belongs to an elite order, a hallow",
    "Counting in binary: 0000, 0001, 0010, 0011,",
]


class UserPrompt(HttpUser):
    wait_time = between(1, 3)

    def __init__(self, parent):
        super().__init__(parent)
        self.i_sample = 0
        self.n_samples = len(TEST_PROMPTS)
        self.post_str = "/predictions/falcon40b"
        self.api_key = f"{os.environ['AUTHORIZATION']}"

    @task
    def test(self):
        prompt_text = TEST_PROMPTS[self.i_sample % self.n_samples]
        postdata = {
            "text": prompt_text,
        }
        headers = {
            "Authorization": self.api_key,
            "content-type": "application/json",
        }
        # Note: clear cookies before each request, unless you want to test repeated calls from same user session
        # self.client is an instance of HttpSession, which is a wrapper for a requests.Session
        self.client.cookies.clear()
        res = self.client.post(self.post_str, json=postdata, headers=headers)
        self.i_sample += 1
