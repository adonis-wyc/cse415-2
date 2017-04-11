# Assignment 1: Basics of Python and String Manipulation
# Author: William Menten-Weil

# Part A. Defining Functions (30 points)

def four_x_cubed_plus_1(x):
    return 4 * x**3 + 1

print(four_x_cubed_plus_1(1))
print(four_x_cubed_plus_1(2))
print(four_x_cubed_plus_1(5))

alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
def mystery_code(str, shift):
    ret_str = []
    for char in str:
        new_char = char
        if char.lower() in alphabet:
            current_index = alphabet.index(char.lower())
            new_index = (shift + (current_index-2)) % (len(alphabet))
            new_char = alphabet[new_index]
            if char.islower():
                new_char = new_char.upper()
            else:
                new_char = new_char.lower()
        ret_str.append(new_char)
    return "".join(ret_str)

print(mystery_code("abc Iz th1s Secure? n0, no, 9!", 13))
print(mystery_code("abc Iz th1s Secure? n0, no, 9!", 17))
print(mystery_code("abc Iz th1s Secure? n0, no, 9!", 20))

def quintuples(arr):
    ret_arrs = []
    pointer = 0
    while pointer < len(arr):
        ret_arrs.append(arr[pointer: pointer+5])
        pointer += 5
    return ret_arrs

print(quintuples([2, 5, 1.5, 100, 3, 8, 7, 1, 1]))
print(quintuples([2, 5, 1.5, 100, 3, 8, 7, 1, 1, 0, -2, -5]))
print(quintuples([2, 5, 1.5, 100, 3, 8, 7, 1, 1, 0, -2, -5, 1, 4, 5]))

irregular = {
    "have": "had",
    "be": "been",
    "eat": "ate",
    "go": "gone",
    "fall": "fell"
}
vowels = ['a','e','i','o','u']
odd_consonants = ['y','w']
def past_tense(words):
    past = []
    for word in words:
        if word in irregular:
            past.append(irregular[word])
        else:
            if word[-1] == 'e':
                word += "d"
            elif word[-1] == 'y':
                word = word[:-1]
                word += "ied"
            elif word[-2] in vowels \
                    and word[-3] not in vowels \
                    and word[-1] not in vowels \
                    and word[-1] not in odd_consonants:
                word += word[-1] + 'ed'
            else:
                word += 'ed'
            past.append(word)
    return past

print(past_tense(['guess', 'debug', 'return', 'finish']))
print(past_tense(['program', 'debug', 'execute', 'crash', 'repeat', 'eat']))
print(past_tense(['program', 'debug', 'execute', 'crash', 'repeat', 'eat', 'fly', 'fall', 'vomit']))
