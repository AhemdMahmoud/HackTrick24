def most_recurring_words(words, X):
    word_freq = {}

    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1


    sorted_words = sorted(word_freq.keys(), key=lambda x: (-word_freq[x], x))

    return sorted_words[:X]

words_list = ["a", "a", "a", "b", "b", "b", "d", "d"]
X = 3

result = most_recurring_words(words_list, X)
print(result)  

# ["a", "a"] X = 1
# ["b", "a"] X = 1
# ["aaa", "a"] X = 1
# ["a", "a", "a", "b", "b", "b", "d", "d"] X = 2
# ["a", "b", "d"] X = 3