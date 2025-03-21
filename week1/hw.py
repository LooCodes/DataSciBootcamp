
# 1.⁠ ⁠Display Fibonacci Series upto 10 terms
class Solution1:
    def __init__(self):
        self.cache = {}
        
    def fib(self, n=10):
        if n <= 1:
            return n
        
        if n in self.cache:
            return self.cache[n]
        self.cache[n] = self.fib(n-1) + self.fib(n-2)
        return self.cache[n]

# 2.⁠ ⁠Display numbers at the odd indices of a list
class Solution2:
    def __init__(self, arr=[]):
        self.arr = arr
    def print_odds(self):
        for i in range(1, len(self.arr), 2):
            print(self.arr[i])

# 3. Your task is to count the number of different words in this text
class Solution3:
    def __init__(self):
        self.string = """

I have provided this text to provide tips on creating interesting paragraphs.

First, start with a clear topic sentence that introduces the main idea.

Then, support the topic sentence with specific details, examples, and evidence.

Vary the sentence length and structure to keep the reader engaged.

Finally, end with a strong concluding sentence that summarizes the main points.

Remember, practice makes perfect!

"""
    def counter(self):
        modified = self.string.strip().split()
        return len(set(modified))

# 4.⁠ ⁠Write a function count_vowels(word) that takes a word as an argument and returns the number of vowels in the word
ARR= ['a', 'e', 'i','o', 'u']
ARR2 = ['A', 'E', 'I', 'O', 'U']
def count_vowels(word):
    sum = 0
    for chr in word:
        if chr in ARR or chr in ARR2:
            sum += 1
    return sum 
# count_vowels('aeiou')

#  5.⁠ ⁠Iterate through the following list of animals and print each one in all caps.
animals=['tiger', 'elephant', 'monkey', 'zebra', 'panther']
for i in range(len(animals)):
    print(animals[i][:].upper())

# 6.⁠ ⁠Write a program that iterates from 1 to 20, printing each number and whether it's odd or even.
for i in range(1, 21):
    if i % 2 ==0:
        print('EVEN', i)
    else:
        print('ODD', i)

# 7.⁠ ⁠Write a function sum_of_integers(a, b) that takes two integers as input from the user and returns their sum.
def sum_of_integers(a, b):
    return a + b

a = int(input("Enter first integer: "))
b = int(input("Enter second integer: "))

print("Sum:", sum_of_integers(a, b))




