# Add the necessary imports here
import pandas as pd
import numpy as np
import torch
from joblib import load
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import cv2
from tensorflow.keras.models import load_model

# loaded_svm_model = load('/content/svm_model.joblib')
# loaded_ml_easy = load_model('/content/lstm_model')
# #from utils import *


# def solve_cv_easy(test_case: tuple) -> list:
#     shredded_image = test_case
#     shredded_image = np.array(shredded_image)
#     def calculate_similarity(shred1, shred2):
#         """
#         Calculate the similarity between two shreds based on their overlapping pixels.
#         """
#         overlap_width = shred1.shape[1] // 64  # Assuming 1/4th of the width is the overlap
#         overlap1 = shred1[:, -overlap_width:]
#         overlap2 = shred2[:, :overlap_width]
#         bitwise_and_result = cv2.bitwise_and(overlap1, overlap2)
#         similarity = np.sum(bitwise_and_result)
#         return similarity
#     def find_best_match(current_shred, remaining_shreds):
#         """
#         Find the best match for the current shred from the remaining shreds.
#         """
#         best_match_index = -1
#         best_similarity = -1

#         for i, shred in enumerate(remaining_shreds):
#             similarity = calculate_similarity(current_shred, shred)
#             if similarity > best_similarity:
#                 best_similarity = similarity
#                 best_match_index = i

#         return best_match_index
#     def solve_puzzle(shreds):
#         """
#         Solve the puzzle and return the ordered list of shreds.
#         """
#         ordered_indices = [0]  # Start with the first shred
#         remaining_shreds = list(range(1, len(shreds)))

#         while remaining_shreds:
#             last_shred_index = ordered_indices[-1]
#             last_shred = shreds[last_shred_index]

#             best_match_index = find_best_match(last_shred, [shreds[i] for i in remaining_shreds])
#             ordered_indices.append(remaining_shreds[best_match_index])
#             remaining_shreds.pop(best_match_index)

#         return ordered_indices
#     overlap_width = 2  # Width of the overlapping region between shreds

#     ordered_indices = solve_puzzle(shredded_image)
#     return (ordered_indices)

#     """
#     This function takes a tuple as input and returns a list as output.

#     Parameters:
#     input (tuple): A tuple containing two elements:
#         - A numpy array representing a shredded image.
#         - An integer representing the shred width in pixels.

#     Returns:
#     list: A list of integers representing the order of shreds. When combined in this order, it builds the whole image.
#     """


# def solve_cv_medium(input: tuple) -> list:
#     combined_image_array , patch_image_array = input
#     image1 = np.array(combined_image_array,dtype=np.uint8)
#     image2 = np.array(patch_image_array,dtype=np.uint8)

#     # Convert images to grayscale
#     gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
#     gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

#     # Initialize SIFT detector
#     sift = cv2.SIFT_create()

#     # Find keypoints and descriptors
#     keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
#     keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

#     # Initialize brute force matcher
#     bf = cv2.BFMatcher()

#     # Match descriptors
#     matches = bf.knnMatch(descriptors1, descriptors2, k=2)

#     # Apply ratio test
#     good_matches = []
#     for m, n in matches:
#         if m.distance < 0.75 * n.distance:
#             good_matches.append(m)

#     # If enough good matches found
    
#     src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#     dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

#     # Find homography
#     M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

#     # Warp image2 to image1
#     warped_image2 = cv2.warpPerspective(image2, M, (image1.shape[1], image1.shape[0]))

#     # Inverse warp image1 to remove the target part
#     mask = cv2.cvtColor(warped_image2, cv2.COLOR_BGR2GRAY)
#     mask[mask > 0] = 255
#     result = cv2.inpaint(image1, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

#     return result.tolist()
#     """
#     This function takes a tuple as input and returns a list as output.

#     Parameters:
#     input (tuple): A tuple containing two elements:
#         - A numpy array representing the RGB base image.
#         - A numpy array representing the RGB patch image.

#     Returns:
#     list: A list representing the real image.
#     """
#     return []


# def solve_cv_hard(input: tuple) -> int:
#     extracted_question, image = input
#     image = np.array(image)
#     """
#     This function takes a tuple as input and returns an integer as output.

#     Parameters:
#     input (tuple): A tuple containing two elements:
#         - A string representing a question about an image.
#         - An RGB image object loaded using the Pillow library.

#     Returns:
#     int: An integer representing the answer to the question about the image.
#     """
#     return 0


# def solve_ml_easy(input: pd.DataFrame) -> list:
#     """
#     This function takes a pandas DataFrame as input and returns a list as output.

#     Parameters:
#     input (pd.DataFrame): A pandas DataFrame representing the input data.

#     Returns:
#     list: A list of floats representing the output of the function.
#     """
#     data = pd.DataFrame(data)
#     data['timestamp'] = pd.to_datetime(data['timestamp'])
#     data.set_index('timestamp', inplace=True)

#     scaler = MinMaxScaler()
#     data_scaled = scaler.fit_transform(data)
#     forecast_steps = 50
#     forecast_input = data_scaled.reshape(1, data_scaled.shape[0], data_scaled.shape[1])  # Use the last sequence from test data for forecasting

#     forecast = []
#     for _ in range(forecast_steps):
#         pred = loaded_ml_easy.predict(np.array([forecast_input]))
#         forecast.append(pred[0, 0])
#         forecast_input = np.append(forecast_input[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

#     # Inverse scaling for forecasted values
#     forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
#     # change list of lists to list of floats
#     forecast = [i[0] for i in forecast]
#     return forecast

# def solve_ml_medium(input: list) -> int:
#     """
#     This function takes a list as input and returns an integer as output.

#     Parameters:
#     input (list): A list of signed floats representing the input data.

#     Returns:
#     int: An integer representing the output of the function.
#     """
#     res = loaded_svm_model.predict([input])
#     return res[0]



# def solve_sec_medium(input: torch.Tensor) -> str:
#     img = torch.tensor(img)
#     """
#     This function takes a torch.Tensor as input and returns a string as output.

#     Parameters:
#     input (torch.Tensor): A torch.Tensor representing the image that has the encoded message.

#     Returns:
#     str: A string representing the decoded message from the image.
#     """
#     return ''

# #---------------------------------------------------------------------
# # SEC Hard 
# #initail permutation
# ip_table = [
#     58, 50, 42, 34, 26, 18, 10, 2,
#     60, 52, 44, 36, 28, 20, 12, 4,
#     62, 54, 46, 38, 30, 22, 14, 6,
#     64, 56, 48, 40, 32, 24, 16, 8,
#     57, 49, 41, 33, 25, 17, 9, 1,
#     59, 51, 43, 35, 27, 19, 11, 3,
#     61, 53, 45, 37, 29, 21, 13, 5,
#     63, 55, 47, 39, 31, 23, 15, 7
# ]
# # PC1 permutation table
# pc1_table = [
#     57, 49, 41, 33, 25, 17, 9, 1,
#     58, 50, 42, 34, 26, 18, 10, 2,
#     59, 51, 43, 35, 27, 19, 11, 3,
#     60, 52, 44, 36, 63, 55, 47, 39,
#     31, 23, 15, 7, 62, 54, 46, 38,
#     30, 22, 14, 6, 61, 53, 45, 37,
#     29, 21, 13, 5, 28, 20, 12, 4
# ]
# # Define the left shift schedule for each round
# shift_schedule = [1, 1, 2, 2,
#                   2, 2, 2, 2,
#                   1, 2, 2, 2,
#                   2, 2, 2, 1]

# # PC2 permutation table
# pc2_table = [
#     14, 17, 11, 24, 1, 5, 3, 28,
#     15, 6, 21, 10, 23, 19, 12, 4,
#     26, 8, 16, 7, 27, 20, 13, 2,
#     41, 52, 31, 37, 47, 55, 30, 40,
#     51, 45, 33, 48, 44, 49, 39, 56,
#     34, 53, 46, 42, 50, 36, 29, 32
# ]
# #expension
# e_box_table = [
#     32, 1, 2, 3, 4, 5,
#     4, 5, 6, 7, 8, 9,
#     8, 9, 10, 11, 12, 13,
#     12, 13, 14, 15, 16, 17,
#     16, 17, 18, 19, 20, 21,
#     20, 21, 22, 23, 24, 25,
#     24, 25, 26, 27, 28, 29,
#     28, 29, 30, 31, 32, 1
# ]

# # S-box tables for DES
# s_boxes = [
#     # S-box 1
#     [
#         [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
#         [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
#         [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
#         [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]
#     ],
#     # S-box 2
#     [
#         [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
#         [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
#         [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
#         [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]
#     ],
#     # S-box 3
#     [
#         [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
#         [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
#         [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
#         [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]
#     ],
#     # S-box 4
#     [
#         [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
#         [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
#         [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
#         [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]
#     ],
#     # S-box 5
#     [
#         [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
#         [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
#         [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
#         [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]
#     ],
#     # S-box 6
#     [
#         [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
#         [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
#         [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
#         [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]
#     ],
#     # S-box 7
#     [
#         [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
#         [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
#         [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
#         [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]
#     ],
#     # S-box 8
#     [
#         [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
#         [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
#         [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
#         [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]
#     ]
# ]
# p_box_table = [
#     16, 7, 20, 21, 29, 12, 28, 17,
#     1, 15, 23, 26, 5, 18, 31, 10,
#     2, 8, 24, 14, 32, 27, 3, 9,
#     19, 13, 30, 6, 22, 11, 4, 25
# ]
# ip_inverse_table = [
#     40, 8, 48, 16, 56, 24, 64, 32,
#     39, 7, 47, 15, 55, 23, 63, 31,
#     38, 6, 46, 14, 54, 22, 62, 30,
#     37, 5, 45, 13, 53, 21, 61, 29,
#     36, 4, 44, 12, 52, 20, 60, 28,
#     35, 3, 43, 11, 51, 19, 59, 27,
#     34, 2, 42, 10, 50, 18, 58, 26,
#     33, 1, 41, 9, 49, 17, 57, 25
# ]

# def hex_to_bin(hex_string):
#     return bin(int(hex_string, 16))[2:].zfill(64)

# def binary_to_hex(binary_string):
#     return hex(int(binary_string, 2))[2:].zfill(16).upper()

# def ip_on_binary_rep(binary_representation):

#   ip_result = [None] * 64

#   for i in range(64):
#       ip_result[i] = binary_representation[ip_table[i] - 1]

#   # Convert the result back to a string for better visualization
#   ip_result_str = ''.join(ip_result)

#   return ip_result_str

# def generate_round_keys(key_plain):

#     # Key into binary
#     # binary_representation_key = key_in_binary_conv()
#     binary_representation_key = hex_to_bin(key_plain)
#     pc1_key_str = ''.join(binary_representation_key[bit - 1] for bit in pc1_table)


#     # Split the 56-bit key into two 28-bit halves
#     c0 = pc1_key_str[:28]
#     d0 = pc1_key_str[28:]
#     round_keys = []
#     for round_num in range(16):
#         # Perform left circular shift on C and D
#         c0 = c0[shift_schedule[round_num]:] + c0[:shift_schedule[round_num]]
#         d0 = d0[shift_schedule[round_num]:] + d0[:shift_schedule[round_num]]
#         # Concatenate C and D
#         cd_concatenated = c0 + d0

#         # Apply the PC2 permutation
#         round_key = ''.join(cd_concatenated[bit - 1] for bit in pc2_table)

#         # Store the round key
#         round_keys.append(round_key)
#     return round_keys

# def encryption(user_input, key_plain):
#     binary_rep_of_input = hex_to_bin(user_input)
#     # Initialize lists to store round keys
#     round_keys = generate_round_keys(key_plain)

#     ip_result_str = ip_on_binary_rep(binary_rep_of_input)

#     # the initial permutation result is devided into 2 halfs
#     lpt = ip_result_str[:32]
#     rpt = ip_result_str[32:]



#     # Assume 'rpt' is the 32-bit right half, 'lpt' is the 32-bit left half, and 'round_keys' is a list of 16 round keys

#     for round_num in range(16):
#         # Perform expansion (32 bits to 48 bits)
#         expanded_result = [rpt[i - 1] for i in e_box_table]

#         # Convert the result back to a string for better visualization
#         expanded_result_str = ''.join(expanded_result)

#         # Round key for the current round
#         round_key_str = round_keys[round_num]


#         xor_result_str = ''
#         for i in range(48):
#             xor_result_str += str(int(expanded_result_str[i]) ^ int(round_key_str[i]))


#         # Split the 48-bit string into 8 groups of 6 bits each
#         six_bit_groups = [xor_result_str[i:i+6] for i in range(0, 48, 6)]

#         # Initialize the substituted bits string
#         s_box_substituted = ''

#         # Apply S-box substitution for each 6-bit group
#         for i in range(8):
#             # Extract the row and column bits
#             row_bits = int(six_bit_groups[i][0] + six_bit_groups[i][-1], 2)
#             col_bits = int(six_bit_groups[i][1:-1], 2)

#             # Lookup the S-box value
#             s_box_value = s_boxes[i][row_bits][col_bits]

#             # Convert the S-box value to a 4-bit binary string and append to the result
#             s_box_substituted += format(s_box_value, '04b')

#         # Apply a P permutation to the result
#         p_box_result = [s_box_substituted[i - 1] for i in p_box_table]

#         # # Convert the result back to a string for better visualization
#         # p_box_result_str = ''.join(p_box_result)


#         # Convert LPT to a list of bits for the XOR operation
#         lpt_list = list(lpt)

#         # Perform XOR operation
#         new_rpt = [str(int(lpt_list[i]) ^ int(p_box_result[i])) for i in range(32)]

#         # Convert the result back to a string for better visualization
#         new_rpt_str = ''.join(new_rpt)

#         # Update LPT and RPT for the next round
#         lpt = rpt
#         rpt = new_rpt_str

#         # Print or use the RPT for each round

#     print('\n')
#     # At this point, 'lpt' and 'rpt' contain the final left and right halves after 16 rounds

#     # After the final round, reverse the last swap
#     final_result = rpt + lpt

#     # Perform the final permutation (IP-1)
#     final_cipher = [final_result[ip_inverse_table[i] - 1] for i in range(64)]

#     # Convert the result back to a string for better visualization
#     final_cipher_str = ''.join(final_cipher)

#     # Print or use the final cipher(binary)
#     # print("Final Cipher binary:", final_cipher_str, len(final_cipher_str))


#     # Convert binary cipher to ascii
#     final_cipher_ascii = binary_to_hex(final_cipher_str)
#     # print("Final Cipher text:", final_cipher_ascii)

#     return final_cipher_ascii




# def solve_sec_hard(input:tuple)->str:
#   key, plain_text = input
#   # Encryption
#   enc = encryption(plain_text,key)
#   # Decyption
#   # First we'll convert Final Cipher text into binary
#   # enc_to_binary = hex_to_bin(enc)
#   return enc

# # print(solve_sec_hard(input=("266200199BBCDFF1","0123456789ABCDEF")))
# # Your output should be : ”4E0E6864B5E1CA52”
# #---------------------------------------------------------------------
def solve_problem_solving_easy(input: tuple) -> list:
    """
    This function takes a tuple as input and returns a list as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A list of strings representing a question.
        - An integer representing a key.

    Returns:
    list: A list of strings representing the solution to the problem.
    """

    words, x = input
    word_freq = {}

    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1


    sorted_words = sorted(word_freq.keys(), key=lambda x: (-word_freq[x], x))

    return sorted_words[:x]


def solve_problem_solving_medium(input: str) -> str:
    """
    This function takes a string as input and returns a string as output.

    Parameters:
    input (str): A string representing the input data.

    Returns:
    str: A string representing the solution to the problem.
    """
    encoded_str = input
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


def solve_problem_solving_hard(input: tuple) -> int:
    """
    This function takes a tuple as input and returns an integer as output.

    Parameters:
    input (tuple): A tuple containing two integers representing m and n.

    Returns:
    int: An integer representing the solution to the problem.
    """
    x, y = input
    dp = [[0] * y for _ in range(x)]

    for i in range(x):
        dp[i][0] = 1
    for j in range(y):
        dp[0][j] = 1

    # Fill the rest of the grid
    for i in range(1, x):
        for j in range(1, y):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]


    return dp[x-1][y-1]

riddle_solvers = {
    # 'cv_easy': solve_cv_easy,
    # 'cv_medium': solve_cv_medium,
    # 'cv_hard': solve_cv_hard,
    # 'ml_easy': solve_ml_easy,
    # 'ml_medium': solve_ml_medium,
    # 'sec_medium_stegano': solve_sec_medium,
    # 'sec_hard':solve_sec_hard,
    'problem_solving_easy': solve_problem_solving_easy,
    'problem_solving_medium': solve_problem_solving_medium,
    'problem_solving_hard': solve_problem_solving_hard
}
