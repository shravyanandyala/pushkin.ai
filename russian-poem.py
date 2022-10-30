import csv
import sys
import re 

csv.field_size_limit(sys.maxsize) 

file = open('poems.csv')
list_of_poems =[*csv.DictReader(file)]

final_data = "" 

def has_cyrillic(text):
    return bool(re.search('[\u0400-\u04FF]', text)) 

def is_valid_sentence(sentence): 
    word_list = sentence.split(" ")  
    for word in word_list: 
        if not has_cyrillic(word): 
            return False 
    return True
            
def checkIfRomanNumeral(numeral):
    numeral = numeral.upper()
    validRomanNumerals = ["M", "D", "C", "L", "X", "V", "I", "(", ")"]
    for letters in numeral:
        if letters not in validRomanNumerals:
            return False
    return True

for poem in list_of_poems:  
    data = poem["text"]  
    list_of_lines = data.split("\n") 
    poem_text_string = "" 
    for t in list_of_lines: 
        if not (len(t) <= 1 or t[0] == "." or t.isnumeric() or t == "………………" or checkIfRomanNumeral(t) or t == "* * *" or not is_valid_sentence(t)): 
            poem_text_string += t 
            poem_text_string += "\n"
    final_data += poem_text_string

print(final_data)