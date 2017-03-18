# Assignment 1: Basics of Python and String Manipulation
# Author: William Menten-Weil

# Part B. Chat Agent (70 Points)
import json
import requests
from re import *   # Loads the regular expression module.
import nltk
from random import choice
import operator

punctuation_pattern = compile(r"\,|\.|\?|\!|\;|\:")
positive_adj = ["funny", 'good', 'great', 'hilarious', 'amazing', 'wonderful', 'incredible', 'comical',]
negative_adj = ['terrible', 'bad', 'horrible', 'poor', 'weak', 'shitty']

apis = {
    "random": {
        'content_type': 'JSON',
        'content_structure': ['joke'], # the location of the content in the return object
        'url': {
            'base': "http://tambal.azurewebsites.net/joke/random",
            'params': []
        },
    },
    "norris": {
        'content_type': 'JSON',
        'content_structure': ['value'],
        'url': {
            'base': "https://api.chucknorris.io/jokes/random",
            'params': ['category']
        },
    }
}

silence_ans = ["Please say something.", "Are those crickets I hear?", "Huh, tough crowd."]
inappropriate_ans = ["Stop, your making me blush.", "That's inappropriate.", "Shouldn't I be the lewd one?"]
categories = {}

# maps each category to its associated api
def map_categories():
    categories['random'] = 'random'
    req = requests.get('https://api.chucknorris.io/jokes/categories')
    cats = json.loads(req.text)
    for c in cats:
        if c not in categories:
            categories[c] = "norris"

map_categories()

manual = ["Here is a list of command which I understand:",
          "'' - responds to your silence",
          "'what are you wearing?' - responds to your inappropriateness.",
          "[category] jokes - changes category",
          "surprise me - switches to a random category",
          "current category - displays the active category",
          "categories - displays the list of available categories",
          "knock knock - my knock knock joke",
          "who's there? - my knock knock response",
          "{save/store/keep/favorite} && {joke/one/that} - saves the last joke to your favorites",
          "{saves/favorites}/({saved/stored/kept/favorite} && {jokes/ones}) - displays your favorites",
          "{'a joke'/more/again/another} - requests a joke",
          "preferences/my preference - displays the two categories which you enjoy the most",
          "FEEDBACK PROCESSING \n\t {ahahahha/hahahah} - positive +2 \n\t saving a joke - positive +2 \n\t \
ADJ + {joke/one/that} - positive/negative + #ADJ (eg. That was good. == positive +1)"
          ]
# a chat bot to provide witty jokes
class Jokester():
    agent_name = "Jester"
    current_category = "random"
    saved_jokes = []
    last_action = None
    silences = -1
    inappropriate_cycle = 0
    learned_preferences = {}
    jokes_told = 0

    # getter for agent name
    def agentName(self):
        return self.agent_name

    # introduces the agent
    def introduce(self):
        intro = '''
My name is %s Wittaker, and I tell bad - mostly Chuck Norris - jokes.
I was programmed by William Menten-Weil. If I offend you in any way, contact him at wtmenten@uw.edu
Tell me what type of jokes you like? Or, simply ask for 'a joke'. Also, feel free to ask for 'help'.
I know %s and %s jokes.
''' % (self.agent_name, ", ".join(list(categories.keys())[:-1]), list(categories.keys())[-1])
        return intro

    # decorates the response with the agents name
    def respond(self, input):
        response = self.agent_name+": "
        return response + self._respond(input)

    # does the actual responding
    def _respond(self, input):
        wordlist = split(' ', remove_punctuation(input))
        # undo any initial capitalization:
        wordlist[0] = wordlist[0].lower()

        # 1. this rule responds to empty input
        if wordlist[0] == '' and len(wordlist) == 1:
            self.silences += 1
            if self.silences >= len(silence_ans):
                self.silences = 0
                return self.do_joke()
            res = silence_ans[self.silences]
            return res

        # 2. this rule responds to an inappropriate input
        if input.lower() == "what are you wearing?":
            if self.inappropriate_cycle >= len(inappropriate_ans):
                self.inappropriate_cycle = 0
            res = inappropriate_ans[self.inappropriate_cycle]
            self.inappropriate_cycle += 1
            return res

        # 3. this rule displays the manual
        if "help" == input.lower() or "man" == input.lower():
            return "\n\t".join(manual)

        # 4. this rule changes the active category
        if "jokes" in wordlist:
            for word in wordlist:
                if word in categories:
                    self.current_category = word
                    return "Alright I'll only tell %s jokes" % self.current_category

        # 5. this rule selects a random category
        if "surprise me" in input.lower():
            self.current_category = choice(list(categories.keys()))
            return "Alright I'll only tell %s jokes" % self.current_category

        # 6. this rule shows which categories you prefer the most
        if "my preference" in input.lower() or 'preferences' in wordlist:
            if len(list(self.learned_preferences.keys())) == 0:
                return "I don't know yet."
            else:
                top_cats = sorted(self.learned_preferences.items(), key=operator.itemgetter(1))
                return "Your favorite categories are " + " and ".join([x[0] for x in top_cats[:2]])

        # 7. this rule displays the current category
        if "current category" in input.lower():
            return "The current category is %s." % self.current_category

        # 8. this rule displays the list of categories
        if "categories" in wordlist or "category?" in input:
            return "I have jokes in the following categories: " + ", ".join(categories.keys()) + "."

        # 9. this rule responds to an knock knock
        if "knock knock" in input:
            return "Hey, I'm the one telling the jokes here. Knock knock"

        # 10. this rule responds to the second part of knock knock
        if input.lower() == "who's there?" or input.lower() == "who is there?":
            self.last_action = "joke"
            return "Me, Jester!"

        # 11. this rule saves the last joke in your favorites
        if any(x in ['save', 'store', 'keep', 'favorite'] for x in wordlist) \
                and any(x in ['joke', 'one', 'that'] for x in wordlist):
            self.record_sentiment(2)
            if self.last_joke not in self.saved_jokes:
                self.saved_jokes.append(self.last_joke)
                return "Alright, I'll remember that one."
            else:
                return "I already saved that one."

        # 12. this rule displays your favorites
        if ("saves" in wordlist or "favorites" in wordlist) \
                or (
                        any(x in ['saved', 'stored', 'kept', 'favorite'] for x in wordlist)
                        and any(x in ['jokes', 'ones'] for x in wordlist)
                    ):
            return "Here are your favorites:\n\t" + "\n\t".join(self.saved_jokes)

        # 13. this rule requests a joke
        if "a joke" in input.lower():
            self.last_action = "joke"
            return self.do_joke()

        # 14-16. these rule only engage after a joke has been told
        if self.jokes_told > 0:

            # 14. this rule requests another joke
            if "another" in wordlist or "again" in wordlist or "more" in wordlist:
                self.last_action = "joke"
                return self.do_joke()

            # 15. this rule processes a feedback sentence
            if "joke" in wordlist or "one" in wordlist or "that" in wordlist:
                pos_tags = nltk.pos_tag(nltk.word_tokenize(input))
                sentiment = 0
                direction = 1
                for tag in pos_tags:
                    if tag[0] == "not" or "n't" in tag[0]:
                        direction = -1
                    if "JJ" in tag[1]:
                        if tag[0] in positive_adj:
                            sentiment += direction
                            direction = 1
                        if tag[0] in negative_adj:
                            sentiment -= direction
                            direction = 1
                return self.record_sentiment(sentiment)

            # 16. this rule responds to laughter
            if input.lower().replace("a","").replace("h","") == "":  # ahahahah
                sentiment = 2
                return self.record_sentiment(sentiment)
            # catch
            return "Sorry, I didn't catch that."
        # catch
        return "Sorry, I didn't catch that."

    # this function does the sentiment recording and picks which response to give
    def record_sentiment(self, sentiment):
        if self.last_action == "sentiment":
            return choice(["I know that already.", "Tell me something else.", "You're a real critic huh?"])

        if self.current_category in self.learned_preferences:
            self.learned_preferences[self.current_category] += sentiment
        else:
            self.learned_preferences[self.current_category] = sentiment

        self.last_action = "sentiment"

        if sentiment > 0:
            return choice(["I appreciate that.", "Thank you, thank you."])
        elif sentiment < 0:
            return choice(["What do you know.", "Are you not entertained?"])
        else:
            return "Thanks"

    # this function retrieves the joke from the associated api, parses the response and returns it.
    def do_joke(self):
        api = apis[categories[self.current_category]]
        url = api['url']['base']
        if len(api['url']['params']) > 0:
            url += "?"
        for param in api['url']['params']:
            if param == "category":
                url += "category=%s" % self.current_category
        req = requests.get(url)
        parsed = json.loads(req.text)
        for struc in api['content_structure']:
            parsed = parsed[struc]

        self.jokes_told += 1
        self.last_joke = parsed
        return parsed


def remove_punctuation(text):
    'Returns a string without any punctuation.'
    return sub(punctuation_pattern,'', text)
