"""Prompts for the MTP-head self-distillation data (DECODE-PLAN item #1).

Sources: the local bakeoff suite (coding, everyday, agentic, tool use; repository files inlined), chunks of the
local markdown docs (summaries, questions), and prompts written for this item across household chat, advice,
stories/prose, coding, agent/tool use and reasoning. The decode-suite and chain-record prompts are excluded so
that speed measurements never run on training prompts.

Everything here is text for the model to answer; replies are stored as tokens and never executed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path.cwd()  # the data root: run the tools from it

# the decode suite's own prompts (tf_decode_sampled.py, tf_chain_record.py): never trained on
EXCLUDED = {
    'My sister is visiting this weekend. Suggest a relaxed plan for Saturday and Sunday, with food ideas.',
    'Write a short story about a lighthouse keeper who finds an unexpected letter washed ashore.',
    'How should I start a vegetable garden in a small backyard? Explain step by step for a beginner.',
    'Write a Python function that parses a CSV file of expenses and returns totals per category, with tests.',
}

CHAT = [
    "We're hosting eight people for a casual dinner on Friday, two are vegetarian. What should I cook?",
    "My kids (7 and 10) are bored on a rainy afternoon. Give me some ideas that don't involve screens.",
    "What's a good way to split household chores fairly between three roommates?",
    "I just moved into a new apartment. What are the first ten things I should do?",
    "Plan a cheap but fun birthday party for a 6-year-old who loves dinosaurs.",
    "Help me write a friendly message to my neighbour asking them to keep their dog quiet in the mornings.",
    "What are some good board games for a family with teenagers?",
    "I want to have a movie marathon night. Suggest a theme and a lineup of films with snacks to match.",
    "Can you suggest a weekly meal plan for a busy family of four on a budget?",
    "What should I pack for a three-day camping trip in early autumn?",
    "My partner and I keep arguing about the thermostat. Any ideas for a compromise?",
    "How do I get my toddler to sleep through the night?",
    "Give me a checklist for spring cleaning the whole house.",
    "What are some thoughtful gift ideas for my dad who has everything?",
    "Write a short, warm toast for my best friend's wedding.",
    "Our family wants to eat less meat. How do we get started without everyone complaining?",
    "Suggest some easy houseplants for a dark apartment and how to care for them.",
    "I'm hosting Thanksgiving for the first time. Give me a timeline for the day.",
    "What are some fun things to do in a small town on a Sunday afternoon?",
    "Help me organize a messy garage. Where do I even start?",
    "My elderly mother is moving in with us. What should we prepare in the house?",
    "Tell me about the history of the teddy bear in a few paragraphs.",
    "What's the difference between baking soda and baking powder, and when do I use each?",
    "Give me a packing list for a week-long beach holiday with two small children.",
    "I need to write a thank-you note to my child's teacher at the end of the school year. Can you help?",
    "What are some ways to make a small living room feel bigger?",
    "Suggest a relaxing evening routine to help me wind down after work.",
    "We adopted a rescue cat. How do we help her settle in?",
    "Plan a picnic menu that travels well and doesn't need a fridge.",
    "What's a good chore chart system for kids of different ages?",
    "How can I make my morning routine less chaotic with three kids?",
    "Explain how a dishwasher actually cleans the dishes.",
    "Recommend some cozy mystery novels for a long winter.",
    "What should I consider when choosing a family car?",
    "Give me ideas for a date night at home that feels special.",
    "Write a short poem for my grandmother's 90th birthday card.",
    "How do I remove a red wine stain from a carpet?",
    "What are some good conversation starters for a family dinner?",
    "My teenage son wants a part-time job. What should we talk about first?",
    "How do I plan a road trip across three states with a dog?",
    "Describe a perfect lazy Sunday in autumn.",
    "What are fun science experiments I can do in the kitchen with my kids?",
    "I have leftover rice, eggs, frozen peas and a bit of ham. What can I make?",
    "Suggest a simple weekly cleaning schedule for a two-bedroom flat.",
    "How should I prepare for a job interview at a local bakery?",
    "Tell me a few interesting facts about octopuses.",
    "What's a good way to teach kids about saving money?",
    "Our neighbourhood wants to organize a street party. How do we begin?",
    "Explain the rules of cricket to someone who only knows baseball.",
    "Help me write a polite email to my landlord about a leaking tap.",
]

ADVICE = [
    "How do I build an emergency fund when I live paycheck to paycheck?",
    "What are the pros and cons of renting versus buying a home right now?",
    "How can I become a better listener?",
    "I have trouble falling asleep. What habits could help?",
    "How should I prepare for my first marathon? I can run 10 km now.",
    "What's the best way to learn a new language as an adult with a full-time job?",
    "How do I ask my manager for a raise?",
    "I feel overwhelmed by my to-do list every day. How do I prioritize?",
    "How can I reduce my household's energy bill this winter?",
    "What should I know before adopting a dog?",
    "How do I deal with a coworker who keeps taking credit for my work?",
    "What are some healthy snacks for someone trying to lose weight?",
    "How should I talk to my parents about their retirement plans?",
    "What's a sensible way to start investing with a small amount of money?",
    "How do I keep in touch with friends when everyone is busy?",
    "What are the signs that my car's brakes need replacing?",
    "How can I make my small garden attract more bees and butterflies?",
    "I want to switch careers into nursing at 35. What should I think about?",
    "How do I negotiate the price of a used car?",
    "What's a good approach to paying off credit card debt?",
    "How do I stop procrastinating on important but boring tasks?",
    "How can I help my child who is anxious about starting a new school?",
    "What are good strategies for studying for a big exam?",
    "How do I choose a good mattress?",
    "Give me advice for my first week as a new team lead.",
    "How can I cook healthier meals without spending hours in the kitchen?",
    "What should I check before signing a lease?",
    "How can I improve my posture when working at a desk all day?",
    "What should I do if I think I'm being scammed online?",
    "How do I start composting in an apartment?",
    "What are some tips for travelling alone for the first time?",
    "How can I be more confident when speaking in meetings?",
    "What's a good way to set boundaries with a family member who overshares advice?",
    "How do I make my home safer for a crawling baby?",
    "What are some ways to save money on groceries?",
    "How do I get back into exercise after a long break?",
    "How should I plan my finances for having a first child?",
    "What should a beginner know about growing tomatoes?",
    "How can I write a good cover letter?",
    "How do I handle a friend who always cancels plans at the last minute?",
    "What can I do to keep my mind sharp as I get older?",
    "How do I choose between two job offers?",
    "What are the basics of maintaining a bicycle at home?",
    "How can I make my resume stand out for a data analyst role?",
    "My houseplants keep dying. What am I probably doing wrong?",
    "How should I prepare my house for a long holiday away?",
    "How can we reduce screen time for the whole family?",
    "What are good ways to cope with loneliness after moving to a new city?",
    "How do I start meditating if I can't sit still?",
    "What's the right way to jump-start a car battery?",
]

PROSE = [
    "Write a short story about a girl who discovers her grandfather's old radio can pick up voices from the past.",
    "Write a bedtime story about a sleepy dragon who is afraid of the dark.",
    "Write a short story set in a busy train station on the last evening before a holiday.",
    "Describe a thunderstorm rolling over a wheat field, in vivid prose.",
    "Write a story about two elderly neighbours who start a secret garden together.",
    "Write the opening chapter of a cozy mystery set in a village bakery.",
    "Write a short fable about a fox, a crow and a stubborn tortoise.",
    "Write a letter from a soldier in 1916 to his younger sister.",
    "Write a science fiction story about the first child born on Mars.",
    "Write a humorous story about a cat who believes she runs the household.",
    "Describe your ideal small town as if you were writing a travel guide.",
    "Write a story in which a lost umbrella brings two strangers together.",
    "Write a short essay on why people love autumn.",
    "Write a fairy tale about a baker who can bake people's memories into bread.",
    "Write a monologue from the point of view of an old oak tree.",
    "Write a short story about a robot learning to paint.",
    "Write a diary entry of a teenager on the first day at a new school.",
    "Write a short adventure story about kids who find a map in the attic.",
    "Describe a quiet morning in a mountain cabin.",
    "Write a story about a musician who plays for an empty concert hall.",
    "Write a short ghost story that is more sad than scary.",
    "Write a story about a grandmother teaching her grandson to fish.",
    "Write an article for a local newspaper about a community clean-up day.",
    "Write a short story about the last bookshop in a futuristic city.",
    "Write a poem about the sea in winter.",
    "Write a story about a detective who can only solve cases while cooking.",
    "Write a reflective essay about what home means.",
    "Write a short story about a lost dog finding its way home across the city.",
    "Write a story from the perspective of a kite on a windy day.",
    "Write a speech for a high school graduation about taking small risks.",
    "Write a short historical story set in a medieval market.",
    "Write a story about two friends who open a tiny cafe by the sea.",
    "Write a mythological tale explaining why the moon changes shape.",
    "Describe a night market in a big city, full of sounds and smells.",
    "Write a short story where a clock stops at midnight and time pauses for everyone except one boy.",
    "Write a story about an astronaut who hears music on the space station.",
    "Write a story about a family road trip where everything goes wrong but ends well.",
    "Write a short story about a painter who can only paint in blue.",
    "Write the first page of a fantasy novel about a young mapmaker.",
    "Write a short memoir-style piece about learning to ride a bicycle.",
    "Write an encouraging letter to someone starting their first job.",
    "Write a short romantic story set in a library during a snowstorm.",
    "Write a story about a village where it has not rained for a year.",
    "Write a comic dialogue between a toaster and a coffee machine.",
    "Write a story about a girl who can talk to bees.",
    "Write a nature essay about a walk through a forest after rain.",
    "Write a short story about the last day of summer camp.",
    "Write a story about a lighthouse that turns into a hotel, told by its first guest.",
    "Write a story about an inventor whose inventions always work, just not as intended.",
    "Write a heartfelt short story about a boy and his grandfather's pocket watch.",
]

CODE = [
    "Write a Python class implementing a simple bank account with deposits, withdrawals and a transaction history.",
    "Implement binary search in Rust with tests for edge cases.",
    "Write a JavaScript function that debounces another function, and explain how it works.",
    "Write a SQL query that finds the top three customers by total order value in each country.",
    "Explain the difference between a list and a tuple in Python with examples.",
    "Write a bash script that backs up a directory to a timestamped tar.gz file and keeps the last seven backups.",
    "Implement a thread-safe queue in Go using channels.",
    "Write a Python function to validate an email address without using regular expressions, with tests.",
    "Refactor this code to be more readable:\n\ndef f(l):\n    r=[]\n    for i in range(len(l)):\n        if l[i]%2==0:\n            r.append(l[i]*l[i])\n    return r",
    "Write a React component that shows a list of todos with add and delete buttons.",
    "Implement Dijkstra's algorithm in C++ using a priority queue.",
    "Write a Python script that reads a JSON file of users and prints the ten oldest.",
    "Explain what a closure is in JavaScript and give three practical examples.",
    "Write a TypeScript type for a paginated API response and a function that fetches all pages.",
    "Write a Python decorator that retries a function with exponential backoff.",
    "Implement an LRU cache in Java.",
    "Write unit tests with pytest for a function that converts Celsius to Fahrenheit.",
    "Explain Big-O notation with examples from sorting algorithms.",
    "Write a Python generator that yields prime numbers forever.",
    "Write a Dockerfile for a small Flask application and explain each line.",
    "Implement a trie in Python with insert, search and prefix matching.",
    "Write a function in C that reverses a linked list, and explain the pointer updates.",
    "Explain how git rebase differs from git merge, with an example workflow.",
    "Write a Python function that flattens an arbitrarily nested list.",
    "Write a small command-line todo app in Python using argparse and a JSON file for storage.",
    "Write a regular expression that matches ISO 8601 dates and explain it.",
    "Implement merge sort in Haskell.",
    "Write an async Python function that fetches several URLs concurrently with aiohttp and a concurrency limit.",
    "Explain the difference between processes and threads.",
    "Write a SQL schema for a library system with books, members and loans.",
    "Write a Python dataclass for a 2D vector with addition, scaling and a length method, plus tests.",
    "Write a function that checks whether a Sudoku board is valid.",
    "Implement a simple rate limiter (token bucket) in Python.",
    "Write CSS for a responsive three-column layout that collapses to one column on phones.",
    "Explain how HTTP caching headers work (Cache-Control, ETag, Last-Modified).",
    "Write a Python function that computes the Levenshtein distance between two strings.",
    "Write a Rust function that counts word frequencies in a text and returns the top ten.",
    "Implement a min-heap from scratch in Python.",
    "Write a Python context manager that times a block of code.",
    "Explain what dependency injection is and why it helps testing.",
    "Write a Kotlin data class and a function that groups a list of orders by customer.",
    "Write a Python script that renames all .jpeg files in a folder to .jpg.",
    "Implement the game of life in Python with numpy.",
    "Explain the CAP theorem in simple terms.",
    "Write a Go HTTP server with one JSON endpoint that returns the current time.",
    "Write a function that parses a query string into a dictionary, handling repeated keys.",
    "Review this function and point out bugs:\n\ndef average(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total / len(xs)",
    "Write a Python function that converts a Roman numeral to an integer, with tests.",
    "Explain how a hash map handles collisions.",
    "Write a shell one-liner to find the ten largest files under the current directory and explain it.",
]

AGENT = [
    "You are an agent with shell access. The user says: 'my Python tests fail with ModuleNotFoundError: No module named app'. Describe the commands you would run, one step at a time, and why.",
    "Plan how you would migrate a small Flask app from SQLite to PostgreSQL. List the steps as an agent would execute them.",
    "You are a customer support agent for an online shoe shop. A customer says their order arrived in the wrong size. Reply to them and describe the internal actions you take.",
    "Act as a travel-booking agent. The user wants a 3-night trip to Lisbon in May under 800 euros. Describe the searches you would make and propose an itinerary.",
    "You are a coding agent. A CI job fails with 'npm ERR! peer dep missing: react@^18'. Explain your diagnosis and the patch you would make.",
    "You are an email-triage assistant. Sort these emails into urgent, later and ignore, and explain: 1) invoice overdue notice, 2) newsletter, 3) boss asking for slides by 3pm, 4) dentist reminder for next month.",
    "You are a home assistant agent with access to lights, thermostat and calendar. The user says 'get the house ready for movie night at 8'. What actions do you take?",
    "As a research agent, outline how you would compare three electric cars for a family of five, including which sources you'd check.",
    "You are a data-cleaning agent. A CSV has dates in three formats and some empty rows. Describe your step-by-step cleaning plan and the checks you'd run.",
    "You are an agent that manages a shared family calendar. Two events overlap on Saturday: soccer at 10 and a dentist at 10:30. Propose a resolution and the messages you'd send.",
    "Act as a debugging agent: a web page loads slowly only on mobile. List the investigation steps in order.",
    "You are a DevOps agent. Disk usage on a server hit 95%. Describe what you would inspect and what you would clean up safely.",
    "You are a writing assistant agent. The user shares a messy draft of a club newsletter. Describe how you would restructure it and then write the improved opening paragraph.",
    "You are an agent asked to reconcile two lists of product SKUs from different systems. Describe your approach and edge cases.",
    "You are a shopping agent. Find a good laptop for a student under $700: describe the criteria and the shortlist you'd build.",
    "You are a coding agent and the user asks you to add pagination to a REST endpoint returning 10,000 rows. Explain the change and write the new handler in Python.",
    "Act as an agent that organizes a photo library of 20,000 images. What steps and tools would you use?",
    "You are a meal-planning agent. Given the pantry: pasta, canned tomatoes, onions, garlic, chickpeas, spinach, rice. Plan three dinners and a shopping list.",
    "You are a security review agent. A user pasted a config file containing an API key into a public repo. What do you do, in order?",
    "You are an agent helping a small business set up online bookings. Describe the steps and what information you'd need from the owner.",
]

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a text file from the workspace.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write a text file in the workspace.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "list_files", "description": "List files in a workspace directory.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "search_web", "description": "Search the web and return result snippets.",
                                      "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather and forecast for a city.",
                                      "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "add_calendar_event", "description": "Add an event to the family calendar.",
                                      "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "start": {"type": "string"}, "end": {"type": "string"}}, "required": ["title", "start"]}}},
]

TOOL_TASKS = [
    "What's the weather going to be in Seattle this weekend? Should we plan a hike?",
    "Add dentist appointments for both kids next Tuesday at 3pm and 3:30pm.",
    "Look at the files in the project folder and tell me what this project does.",
    "Read notes.txt and turn it into a tidy shopping list saved as shopping.md.",
    "Find the best-reviewed pizza places near downtown Portland.",
    "Check config.yaml and tell me which port the server listens on.",
    "Create a file called plan.md with a study schedule for my chemistry exam in two weeks.",
    "What's the forecast for London for the next 3 days? I need to decide what to pack.",
    "Search for how long to boil an egg for a runny yolk, then summarize.",
    "Read the README.md and write a CONTRIBUTING.md that matches the project's style.",
    "Schedule a family movie night on Friday at 7pm and tell me the weather that evening in Denver.",
    "List the files in src/ and read the main module; explain its structure.",
]

# multi-turn tool conversations: the model continues after a (fabricated, harmless) tool response
TOOL_DIALOGS = [
    [{"role": "user", "content": "What's the weather in Chicago tomorrow?"},
     {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Chicago", "days": 2}}}]},
     {"role": "tool", "content": '{"city":"Chicago","forecast":[{"day":"today","high_c":14,"low_c":6,"sky":"cloudy"},{"day":"tomorrow","high_c":9,"low_c":2,"sky":"rain, wind 30 km/h"}]}'}],
    [{"role": "user", "content": "What files are in the docs folder and which one explains installation?"},
     {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "list_files", "arguments": {"path": "docs"}}}]},
     {"role": "tool", "content": '["docs/index.md","docs/install.md","docs/usage.md","docs/faq.md","docs/changelog.md"]'}],
    [{"role": "user", "content": "Read todo.txt and tell me what's most urgent."},
     {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "todo.txt"}}}]},
     {"role": "tool", "content": "- renew car insurance (expires Friday)\n- buy birthday present for Sam (party Sunday)\n- fix squeaky door\n- book flights for December\n- call plumber about slow drain"}],
    [{"role": "user", "content": "Find me an easy weeknight lasagna recipe."},
     {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "search_web", "arguments": {"query": "easy weeknight lasagna recipe"}}}]},
     {"role": "tool", "content": '[{"title":"30-Minute Skillet Lasagna","snippet":"Brown beef with onion, add marinara and broken noodles, simmer 15 min, top with ricotta and mozzarella."},{"title":"No-Boil Lasagna","snippet":"Layer no-boil noodles with sauce, ricotta mixture and cheese; bake covered 40 min at 190C."}]'}],
    [{"role": "user", "content": "Put swim lessons on the calendar for Saturday 9 to 10am."},
     {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "add_calendar_event", "arguments": {"title": "Swim lessons", "start": "Saturday 09:00", "end": "Saturday 10:00"}}}]},
     {"role": "tool", "content": '{"status":"conflict","existing":{"title":"Grandma visit","start":"Saturday 09:30","end":"Saturday 12:00"}}'}],
    [{"role": "user", "content": "Check app.py and tell me why the /health endpoint returns 500."},
     {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "app.py"}}}]},
     {"role": "tool", "content": "from flask import Flask, jsonify\nimport db\napp = Flask(__name__)\n\n@app.route('/health')\ndef health():\n    conn = db.connect()\n    ok = conn.ping()\n    return jsonify(status='ok' if ok else 'down', version=VERSION)\n"}],
]

REASONING = [
    "If a recipe needs 3 eggs for 12 cookies, how many eggs do I need for 30 cookies? Show your reasoning.",
    "A shop gives 20% off, then another 10% off the reduced price. Is that the same as 30% off? Explain.",
    "Three friends split a bill of $87.60 with a 15% tip. How much does each pay?",
    "A car uses 6.5 litres per 100 km. How much fuel for a 430 km trip, and what does it cost at 1.85 per litre?",
    "Alice is older than Bob, Bob is older than Carol, and Dan is younger than Carol. Who is the second youngest?",
    "If today is Wednesday, what day of the week will it be 100 days from now? Explain.",
    "A tank fills in 6 hours with pipe A and 4 hours with pipe B. How long with both open?",
    "I have 5 shirts, 3 pairs of trousers and 2 pairs of shoes. How many outfits can I make?",
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much is the ball? Explain carefully.",
    "Is 391 a prime number? Show how you check.",
    "What is the probability of rolling at least one six in four rolls of a fair die?",
    "A room is 4.2 m by 3.5 m. How many boxes of flooring are needed if each box covers 2.1 square metres and I want 10% extra?",
    "Sort these numbers and find the median: 14, 3, 27, 8, 19, 11, 6, 22.",
    "If I save $250 a month at 4% annual interest compounded monthly, roughly how much will I have after 3 years?",
    "A train travels 180 km in 2 hours and 15 minutes. What is its average speed in km/h?",
    "Two workers can paint a fence in 5 hours together. One alone takes 8 hours. How long does the other take alone?",
    "My flight leaves Tokyo at 17:40 local time and lands in Los Angeles 10 hours 5 minutes later. LA is 16 hours behind Tokyo. What is the local arrival time?",
    "There are 12 socks in a drawer: 5 black, 4 white, 3 blue. How many must I pull out in the dark to guarantee a matching pair?",
    "A number doubled and then increased by 7 gives 43. What is the number? Show the steps.",
    "If 8 workers build a wall in 9 days, how long would 12 workers take, assuming the same rate?",
    "Which is the better deal: 750 ml for $4.20 or 1.2 litres for $6.30?",
    "A clock shows 3:15. What is the angle between the hour and minute hands?",
    "Solve for x: 3(x - 4) + 2 = 5x - 14. Explain each step.",
    "In a class of 30, 18 play soccer, 12 play basketball, and 7 play both. How many play neither?",
    "A recipe is for 4 people and uses 350 g of flour. How much flour for 7 people?",
    "Explain why the sum of two odd numbers is always even.",
    "If a population of 5,000 grows by 3% a year, what is it after 5 years?",
    "You have a 3-litre jug and a 5-litre jug. How do you measure exactly 4 litres?",
    "A store sells pens in packs of 6 and 8. Can you buy exactly 22 pens? Which combination?",
    "I leave home at 7:50, walk 1.2 km at 5 km/h to the bus, wait 6 minutes, then ride 25 minutes. When do I arrive?",
    "What is 17% of 240, and what is 240 increased by 17%?",
    "A rectangle's perimeter is 46 cm and its length is 5 cm more than its width. Find its area.",
    "Explain the Monty Hall problem and why switching is better.",
    "Convert 72 degrees Fahrenheit to Celsius and explain the formula.",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?",
    "A family spends 28% of a 4,500 monthly income on rent and 15% on food. How much is left for everything else?",
    "What is the least common multiple of 12, 18 and 30? Show the method.",
    "Estimate how many litres of water a household of four uses in a year, explaining your assumptions.",
    "Five people shake hands with each other once. How many handshakes happen? Generalize to n people.",
    "Is it cheaper to drive or take the train for a 300 km trip for two people? Car: 7 L/100km at 1.9 per litre plus 20 parking. Train: 38 per ticket.",
]

SYSTEMS = [
    None,
    "You are a friendly household assistant. Keep answers practical and warm.",
    "You are a concise assistant. Prefer short paragraphs and bullet points.",
    "You are a patient tutor who explains things step by step.",
]


def _bakeoff() -> list[dict]:
    out = []
    for item in json.loads((ROOT / "bakeoff-suite.json").read_text()):
        text = item["prompt"]
        files = item.get("files") or {}
        if files:
            parts = [f"--- {name} ---\n{body}" for name, body in files.items()]
            text = text + "\n\nWorkspace files:\n" + "\n\n".join(parts)
        kind = {"coding": "code", "everyday": "reasoning", "agentic": "agent", "tool_use": "agent"}[item["category"]]
        out.append({"id": f"bakeoff-{item['id']}", "kind": kind, "messages": [{"role": "user", "content": text}]})
    return out


def _docs() -> list[dict]:
    """Chunks of local markdown docs: summarize / explain / extract (prose over technical text)."""

    tree = ROOT / "docs"
    files = sorted(tree.rglob("*.md")) + sorted(ROOT.glob("[A-Z]*.md"))
    asks = ["Summarize the following notes for a busy reader in a few paragraphs.",
            "Explain the following document to someone new to the project, in plain language.",
            "List the key decisions and open questions in the following notes."]
    out, i = [], 0
    for f in files:
        text = f.read_text(errors="replace")
        for start in range(0, min(len(text), 12000), 4000):
            chunk = text[start:start + 4000].strip()
            if len(chunk) < 800:
                continue
            out.append({"id": f"doc-{f.stem}-{start}", "kind": "docs",
                        "messages": [{"role": "user", "content": f"{asks[i % 3]}\n\n{chunk}"}]})
            i += 1
    return out


def all_prompts() -> list[dict]:
    items: list[dict] = []
    for kind, lst in (("chat", CHAT), ("advice", ADVICE), ("prose", PROSE), ("code", CODE), ("agent", AGENT),
                      ("reasoning", REASONING)):
        for j, text in enumerate(lst):
            if text in EXCLUDED:
                continue
            sysmsg = SYSTEMS[j % len(SYSTEMS)]
            msgs = ([{"role": "system", "content": sysmsg}] if sysmsg else []) + [{"role": "user", "content": text}]
            items.append({"id": f"{kind}-{j:03d}", "kind": kind, "messages": msgs})
    for j, text in enumerate(TOOL_TASKS):
        items.append({"id": f"tool-{j:03d}", "kind": "agent", "tools": True,
                      "messages": [{"role": "user", "content": text}]})
    for j, dialog in enumerate(TOOL_DIALOGS):
        items.append({"id": f"tooldlg-{j:03d}", "kind": "agent", "tools": True, "messages": dialog})
    items += _bakeoff()
    items += _docs()
    return items


def split(pid: str) -> str:
    """10% of prompts held out for evaluation, by a hash of the prompt id (all of a prompt's replies together)."""

    return "eval" if int(hashlib.sha256(pid.encode()).hexdigest(), 16) % 10 == 0 else "train"


def render(item: dict, thinking: bool) -> str:
    import jinja2

    tpl = Path("models/qwen38-flash-next/chat_template.jinja").read_text()
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)

    def raise_exception(msg):
        raise ValueError(msg)

    env.globals["raise_exception"] = raise_exception
    t = env.from_string(tpl)
    return t.render(messages=item["messages"], tools=TOOLS if item.get("tools") else None,
                    add_generation_prompt=True, enable_thinking=thinking)


if __name__ == "__main__":
    import collections

    ps = all_prompts()
    print(len(ps), collections.Counter(p["kind"] for p in ps), collections.Counter(split(p["id"]) for p in ps))
    print(render(ps[0], False))
    print(render([p for p in ps if p.get("tools")][-1], True)[-1500:])
