import requests
import numpy as np
from LSBSteg import encode
from riddle_solvers import riddle_solvers
import random
base_url = 'http://16.171.171.147:5000'
team_id = "wI4FoRq"

def init_fox(team_id):
    '''
    In this function you need to hit to the endpoint to start the game as a fox with your team id.
    If a successful response is returned, you will receive back the message that you can break into chunks
    and the carrier image that you will encode the chunks in it.
    '''
    response = requests.post(url=base_url+'/fox/start', json={'teamId': team_id})
    status_code = response.status_code
    data = response.json()

    message = data.get('msg')
    image_carrier = data.get('carrier_image')

    print('Status:', status_code)
    print('Data:', data)

    return message, image_carrier

def get_riddle(team_id, riddle_id):

    # API endpoint URL for requesting a riddle
    url = base_url + '/fox/get-riddle'
    payload = {
        "teamId": team_id,
        "riddleId": riddle_id
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200 or response.status_code == 201:
        print(f"Riddle test case of {riddle_id} received successfully")
        return response.json()
    else:
        print(f"Failed to get riddle. Status code: {response.status_code}")
        return response.json()

def solve_riddle(team_id, riddle_id):
    url = base_url+'/fox/solve-riddle'
    riddle_testcase = get_riddle(team_id, riddle_id)
    print(f'get riddle of {riddle_id} success')
    solver_function = riddle_solvers[riddle_id]
    solution = solver_function(riddle_testcase)
    print(f'solved {riddle_id} successfully')

    payload = {
        "teamId": team_id,
        "solution": solution
    }

    # Send POST
    response = requests.post(url, json=payload)

    if response.status_code == 200 or response.status_code == 201:
        print(f'done solution {riddle_id}')
        return response.json()
    else:
        print(f"Failed to solve {riddle_id} riddle. Status code: {response.status_code}")
        return response.json()

def send_message(team_id, messages, message_entities=['F', 'E', 'R']):
    url = base_url + "/fox/send-message"

    payload = {
        "teamId": team_id,
        "messages": messages,
        "messageEntities": message_entities
    }
    response = requests.post(url, json=payload)

    if response.status_code == 200 or response.status_code == 201:
        return response.json()["status"]
    else:
        print(f"Failed to send message. Status code: {response.status_code}")
        return None
    
def generate_message_array(message, image_carrier):  
    '''
    In this function you will need to create your own startegy. That includes:
        1. How you are going to split the real message into chunkcs
        2. Include any fake chunks
        3. Decide what 3 chuncks you will send in each turn in the 3 channels & what is their entities (F,R,E)
        4. Encode each chunck in the image carrier  
    '''

    '''
    our strategy maximizes the score by using 3 chunks, 3 real, 6 fake, 0 empty
    making the fake = 6 maximizes the second parameter,
    making the empty = 0 maximizes the first parameter
    '''
    # split the message into 3 chunks,
    # each one will not contain the same number of characters
    size = len(message)
    chunk1size = random.randint(1, size-2)
    chunk2size = random.randint(1, size-chunk1size-1)
    chunk3size = size - chunk1size - chunk2size
    chunk1 = message[:chunk1size]
    chunk2 = message[chunk1size:chunk1size+chunk2size]
    chunk3 = message[chunk1size+chunk2size:]

    real_chunks = [chunk1, chunk2, chunk3]
    chunk_sizes = [chunk1size, chunk2size, chunk3size]
    fake_chunks = []

    # generate 6 fake chunks which are the same size as the real chunks, each character is a random letter
    for i in range(6):
        fake_chunk = ''.join([chr(random.randint(97, 122)) for _ in range(chunk_sizes[i//2])])
        fake_chunks.append(fake_chunk)
    
    print('done messages generation and start encoding')
    for i in range(3):
        # encode the real chunks into the carrier image
        real_chunk = real_chunks[i]
        real_image = encode(image_carrier, real_chunk)
        real_image = real_image.tolist()
        # encode the 2 fake chunks into the carrier image
        fake_chunk1 = fake_chunks[i]
        fake_chunk2 = fake_chunks[i+1]
        fake_image1 = encode(image_carrier, fake_chunk1)
        fake_image2 = encode(image_carrier, fake_chunk2)
        fake_image1 = fake_image1.tolist()
        fake_image2 = fake_image2.tolist()

        
        print(f'message {i}:', send_message(team_id, [real_image, fake_image1, fake_image2], ['R', 'F', 'F']))
        

def end_fox(team_id):
    url = base_url + "/fox/end-game"

    payload = {
        "teamId": team_id
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200 or response.status_code == 201:
        return response.text
    else:
        print(f"Failed to end game. Status code: {response.status_code}")
        return response.text

def submit_fox_attempt(team_id):

    try:
        # Step 1: Initialize the game as fox
        secret_msg, carrier = init_fox(team_id)

        # Step 2: Solve riddles
        # 'cv_easy', 'cv_medium', 'ml_easy', 'ml_medium', 'sec_hard'
        riddle_ids = ['problem_solving_easy', 'problem_solving_medium', 'problem_solving_hard']
        for riddle_id in riddle_ids:
            solve_riddle(team_id, riddle_id)

        print('done solving riddles')
        # Step 3&4: Strategy & Send the messages
        # Call the send_message function with your messages and their entities
        generate_message_array(secret_msg, carrier)

        # Step 5: End the game
        end_game_response = end_fox(team_id)
        return end_game_response

    except Exception as e:
        print(f"Error submitting fox attempt: {e}")
        return None

submit_fox_attempt(team_id)