def decode_string(encoded_str):
    decoded_str = ""
    stack = []

    i = 0
    while i < len(encoded_str):
        if encoded_str[i].isdigit():
            num = ""
            while encoded_str[i].isdigit():
                num += encoded_str[i]
                i += 1
            stack.append(int(num))
        elif encoded_str[i] == '[':
            stack.append(encoded_str[i])
            i += 1
        elif encoded_str[i] == ']':
            substr = ""
            while stack[-1] != '[':
                substr = stack.pop() + substr
            stack.pop()  # Discard '['
            repeat = stack.pop()
            decoded_substr = substr * repeat
            stack.append(decoded_substr)
            i += 1
        else:
            stack.append(encoded_str[i])
            i += 1

    while stack:
        decoded_str = stack.pop() + decoded_str

    return decoded_str

# Example
encoded_str = "3[de1[e2[l]]3[o]2[k]]3[a]"
#encoded_str = "10[a]"
decoded_str = decode_string(encoded_str)
print("Decoded string:", decoded_str)
