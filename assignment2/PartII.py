'''PartII.py
William Menten-Weil, CSE 415, Spring 2017, University of Washington
Instructor:  S. Tanimoto.
Assignment 2 Part II.  ISA Hierarchy Manipulation

Status of the implementation of new features:

All forms of redundancy detection and processing are working.
Extra credit (Cycle detection and processing) implemented and working.

'''
# Linneus3.py
# Implements storage and inference on an ISA hierarchy
# This Python program goes with the book "The Elements of Artificial
# Intelligence".
# This version runs under Python 3.x.

# Steven Tanimoto
# (C) 2012.

# The ISA relation is represented using a dictionary, ISA.
# There is a corresponding inverse dictionary, INCLUDES.
# Each entry in the ISA dictionary is of the form
#  ('turtle' : ['reptile', 'shelled-creature'])

from re import *   # Loads the regular expression module.

ISA = {}
INCLUDES = {}
ARTICLES = {}
ALIAS = {}

def store_isa_fact(category1, category2):
    'Stores one fact of the form A BIRD IS AN ANIMAL'
    # That is, a member of CATEGORY1 is a member of CATEGORY2
    try :
        c1list = ISA[category1]
        c1list.append(category2)
    except KeyError :
        ISA[category1] = [category2]
    try :
        c2list = INCLUDES[category2]
        c2list.append(category1)
    except KeyError :
        INCLUDES[category2] = [category1]

def remove_isa_fact(category1, category2):
    ISA[category1].remove(category2)
    INCLUDES[category2].remove(category1)

def get_isa_list(category1):
    'Retrieves any existing list of things that CATEGORY1 is a'
    try:
        c1list = ISA[category1]
        return c1list
    except:
        return []

def get_includes_list(category1):
    'Retrieves any existing list of things that CATEGORY1 includes'
    try:
        c1list = INCLUDES[category1]
        return c1list
    except:
        return []

def isa_test1(category1, category2):
    'Returns True if category 2 is (directly) on the list for category 1.'
    c1list = get_isa_list(category1)
    return c1list.__contains__(category2)

def isa_test(category1, category2, depth_limit = 10):
    'Returns True if category 1 is a subset of category 2 within depth_limit levels'
    if category1 == category2 : return True
    if isa_test1(category1, category2) : return True
    if depth_limit < 2 : return False
    for intermediate_category in get_isa_list(category1):
        if isa_test(intermediate_category, category2, depth_limit - 1):
            return True
    return False

def store_article(noun, article):
    'Saves the article (in lower-case) associated with a noun.'
    ARTICLES[noun] = article.lower()

def get_article(noun):
    'Returns the article associated with the noun, or if none, the empty string.'
    try:
        article = ARTICLES[noun]
        return article
    except KeyError:
        return ''

def linneus():
    'The main loop; it gets and processes user input, until "bye".'
    print('This is Linneus.  Please tell me "ISA" facts and ask questions.')
    print('For example, you could tell me "An ant is an insect."')
    while True :
        info = input('Enter an ISA fact, or "bye" here: ')
        if info == 'bye': return 'Goodbye now!'
        process(info)

# Some regular expressions used to parse the user sentences:
assertion_pattern = compile(r"^(a|an|A|An)\s+([-\w]+)\s+is\s+(a|an)\s+([-\w]+)(\.|\!)*$", IGNORECASE)
query_pattern = compile(r"^is\s+(a|an)\s+([-\w]+)\s+(a|an)\s+([-\w]+)(\?\.)*", IGNORECASE)
what_pattern = compile(r"^What\s+is\s+(a|an)\s+([-\w]+)(\?\.)*", IGNORECASE)
why_pattern = compile(r"^Why\s+is\s+(a|an)\s+([-\w]+)\s+(a|an)\s+([-\w]+)(\?\.)*", IGNORECASE)

def process(info) :
    'Handles the user sentence, matching and responding.'
    result_match_object = assertion_pattern.match(info)
    if result_match_object != None :
        items = result_match_object.groups()
        store_article(items[1], items[0])
        store_article(items[3], items[2])
        a = items[1]
        b = items[3]
        if a in ALIAS:
            a = ALIAS[a]
        if b in ALIAS:
            b = ALIAS[b]
        if b in get_isa_list(a):
            print("You told me that earlier.")
        elif isa_test(a, b):
            print("You don't have to tell me that.")
        elif detect_cycle(items):
            pass
        elif find_transitive_dupes(items):
            pass
        else:
            store_isa_fact(a, b)
            print("I understand.")
        return
    result_match_object = query_pattern.match(info)
    if result_match_object != None :
        items = result_match_object.groups()
        b = items[3]
        if b in ALIAS:
            b = ALIAS[b]
        answer = isa_test(items[1], b)
        if answer :
            print("Yes, it is.")
        else :
            print("No, as far as I have been informed, it is not.")
        return
    result_match_object = what_pattern.match(info)
    if result_match_object != None :
        items = result_match_object.groups()
        supersets = get_isa_list(items[1])
        if supersets != [] :
            first = supersets[0]
            a1 = get_article(items[1]).capitalize()
            a2 = get_article(first)
            print(a1 + " " + items[1] + " is " + a2 + " " + first + ".")
            return
        else :
            subsets = get_includes_list(items[1])
            if subsets != [] :
                first = subsets[0]
                a1 = get_article(items[1]).capitalize()
                a2 = get_article(first)
                print(a1 + " " + items[1] + " is something more general than " + a2 + " " + first + ".")
                return
            else :
                print("I don't know.")
        return
    result_match_object = why_pattern.match(info)
    if result_match_object != None :
        items = result_match_object.groups()
        b = items[3]
        if b in ALIAS:
            b = ALIAS[b]
        if not isa_test(items[1], b) :
            print("But that's not true, as far as I know!")
        else:
            answer_why(items[1], items[3])
        return
    print("I do not understand.  You entered: ")
    print(info)

def answer_why(x, y):
    'Handles the answering of a Why question.'
    if x == y:
        print("Because they are identical.")
        return
    if isa_test1(x, y):
        print("Because you told me that.")
        return
    print("Because " + report_chain(x, y))
    return

from functools import reduce
def report_chain(x, y):
    'Returns a phrase that describes a chain of facts.'
    given_alias = y
    if given_alias in ALIAS:
        y = ALIAS[y];
    chain = find_chain(x, y)
    all_but_last = chain[0:-1]
    last_link = chain[-1]
    if len(chain) == 1:
        main_phrase = reduce(lambda x, y: x + y, map(report_link, chain))
        last_phrase = ""
    else:
        main_phrase = reduce(lambda x, y: x + y, map(report_link, all_but_last))
        last_phrase = "and " + report_link(last_link)
    if given_alias in ALIAS:
        new_last_phrase = last_phrase + "and %s is another name for %s." % (given_alias, y)
    else:
        new_last_phrase = last_phrase[0:-2] + '.'
    return main_phrase + new_last_phrase

def report_link(link):
    'Returns a phrase that describes one fact.'
    x = link[0]
    y = link[1]
    a1 = get_article(x)
    a2 = get_article(y)
    return a1 + " " + x + " is " + a2 + " " + y + ", "

def find_chain(x, z):
    'Returns a list of lists, which each sublist representing a link.'
    if isa_test1(x, z):
        return [[x, z]]
    else:
        for y in get_isa_list(x):
            if isa_test(y, z):
                temp = find_chain(y, z)
                temp.insert(0, [x,y])
                return temp

def test() :
    # process("A hawk is a bird.")
    # process("A hawk is a raptor.")
    # process("A raptor is a bird.")


    # process("A hawk is a raptor")
    # process("A hawk is an animal")
    # process("A bird is an animal")
    # process("A raptor is a bird")

    # process("A chinook is an organism.")
    # process("A sockeye is a salmon.")
    # process("A fish is an animal.")
    # process("A sockeye is an organism.")
    # process("A chinook is an animal.")
    # process("A chinook is a salmon.")
    # process("A sockeye is an animal.")
    # process("A fish is an organism.")
    # process("A salmon is a fish.")

    process("A creature is a being.")
    process("A being is a living-thing.")
    process("A living-thing is a creature.")
    # process("A noun is a creature.")
    # process("A creature is a noun.")
    process("A bug is a creature.")
    process("Is a bug a being?")
    process("Why is a bug a being?")
    process("A living-thing is an organism.")
    process("Is a bug an organism?")
    process("Why is a bug an organism?")

    # process("A turtle is a reptile.")
    # process("A turtle is a shelled-creature.")
    # process("A reptile is an animal.")
    # process("An animal is a thing.")

import itertools
import copy

def get_list_recursive(item, search_list, max_depth = 20):
    flatten = lambda l: [item for sublist in l for item in sublist]

    if item in search_list and max_depth > 0:
        # print("search for %s" % item)
        c1list =  copy.copy(search_list[item])
        c2list = []
        for x in c1list:
            # print("recursing into %s" % x)
            sub_item = get_list_recursive(x, search_list, max_depth=max_depth-1)
            if len(sub_item) != 0:
                for i in sub_item:
                    c2list.append(i)

        if len(c2list) != 0:
            for i in c2list:
                c1list.append(i)
        return c1list
    else:
        # print("key %s not found" % item)
        return []

def find_transitive_dupes(items):
    dupes = []
    a = items [1]
    b = items[3]
    a_includes = get_list_recursive(a, INCLUDES)
    b_isas = get_list_recursive(b, ISA)
    b_isas.append(b)
    for sub in a_includes:
        intersections = set(b_isas).intersection(get_isa_list(sub))
        for inter in intersections:
            dupes.append((sub, inter))

    if len(dupes) == 0:
        return False
    else:
        if len(dupes) > 1:
            print("The following statements you made earlier are now all redundant:")
            [print("a %s is a %s." % d) for d in dupes]
        else:
            print("Your earlier statement that a %s is a %s is now redundant." % (dupes[0]))
        for dupe in dupes:
            remove_isa_fact(*dupe)
        store_isa_fact(a, b)
        return True

def detect_cycle(items):
    if (isa_test(items[3], items[1])):
        chains = find_chain(items[3], items[1])
        chain_ends = [chain[1] for chain in chains]
        chain_ends = ", ".join(chain_ends) + " and " + items[3]
        print("I infer that %s are all names for the same thing, and I'll call it %s." % (chain_ends, items[3]))
        for chain in chains:
            c = chain[1]
            update_aliases(c, items[3])
            for (k, v) in ISA.items():
                v = [items[3] if x==c else x for x in v]
            for (k, v) in INCLUDES.items():
                v = [items[3] if x==c else x for x in v]
            remove_isa_fact(*chain)
        return True
    else:
        return False
def update_aliases(alias, primary):
    for (k, v) in ALIAS.items():
        if v == alias:
            ALIAS[k] = primary
    ALIAS[alias] = primary

test()
linneus()