import os
import sys
from openai import OpenAI
import base64
import json
import time
import simpleaudio as sa
import errno
from elevenlabs import generate, play, set_api_key, voices
from dotenv import load_dotenv, find_dotenv
import requests
import asyncio
from hume import HumeStreamClient
from hume.models.config import FaceConfig
from typing import Any, Dict, List

_ = load_dotenv(find_dotenv()) # read local .env file
OpenAI.api_key = os.getenv('OPENAI_API_KEY')
elevenlab_key = os.getenv('11Labs_API_KEY')
elevenlab_voice_id = os.getenv('11Labs_VOICE_ID')
guum_api_key = os.getenv('GUUM_API_KEY')
HUME_API_KEY = os.getenv('HUME_API_KEY')

FILEPATH = "./frames/frame.jpg"


client = OpenAI()

#set_api_key(os.environ.get("ELEVENLABS_API_KEY"))
set_api_key(elevenlab_key)

def encode_image(image_path):
    while True:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except IOError as e:
            if e.errno != errno.EACCES:
                # Not a "file in use" error, re-raise
                raise
            # File is being written to, wait a bit and retry
            time.sleep(0.1)


def play_audio(text):
    #audio = generate(text, voice=os.environ.get("ELEVENLABS_VOICE_ID"))
    audio = generate(text, voice=elevenlab_voice_id)

    unique_id = base64.urlsafe_b64encode(os.urandom(30)).decode("utf-8").rstrip("=")
    dir_path = os.path.join("narration", unique_id)
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, "audio.wav")

    with open(file_path, "wb") as f:
        f.write(audio)

    play(audio)


def generate_new_line(base64_image):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                },
            ],
        },
    ]


def analyze_image(base64_image, script):

    system_txt = """
You are a licensed professional psychotherapists. \
Narrate the picture of the patient who you are counseling to asess \
any indications of clinical behavioral disorders or mental health issues. \
Aim to retain the most important points, providing a coherent and \
readable summary that could help the patient to focus on the \
main points. Please avoid unnecessary details or tangential points. \
Please consider the overall emotion conveyed by the picture, and \ 
indicate whether the emotion is generally positive, negative, or \
neutral, and provide brief explanations for your analysis \
where possible. \
Please also use the patient "personality" and "current emotions" \
information provided below, delimited by triple dashes, \
in your assessment.
    """
    response = client.chat.completions.create(
        model="gpt-4-vision-preview", temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_txt,
            },
        ]
        + script
        + generate_new_line(base64_image),
        max_tokens=500,
    )
    response_text = response.choices[0].message.content
    return response_text

def get_api_data(api_url, api_key):
    try:
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        
        response = requests.get(api_url, headers=headers)
        
        if response.status_code == 200:
            #print("API Response: ", response.json())
            print("API Response: SUCCESS")
        else:
            print("Failed to fetch data. Status code:", response.status_code)
    except requests.RequestException as e:
        print("Request Exception:", e)
    return response


def top_n_highest(emotions: List[Dict[str, Any]], n) -> List[Dict[str, Any]]:
    # Sort based on the emotion scores in descending order
    sorted_emotions = sorted(emotions, key=lambda x: x['score'], reverse=True)
    
    # Return the top N highest numbers
    return sorted_emotions[:n]


async def hume():
    try:
        client = HumeStreamClient(HUME_API_KEY)
        
        # Enable face identification to track unique faces over the streaming session
        config = FaceConfig(identify_faces=True)
        async with client.connect([config]) as socket:
            result = await socket.send_file(FILEPATH)
            #print(result)
            emotions = result["face"]["predictions"][0]["emotions"]
            #print(emotions)
            #print_emotions(emotions)
    except Exception:
        print(traceback.format_exc())

    return emotions

async def extract_emotion(count) -> str:
    emotions = await hume()
    topN_emotions = top_n_highest(emotions, count)
    emotion_str = None
    for emotion in topN_emotions:
        #print(emotion['name'], "-", emotion['score'])
        if emotion_str:
            emotion_str += ", " + emotion['name']
        else:
            emotion_str = emotion['name']
    return emotion_str
 

def construct_personality(response, emotion_str):
    response_json = response.json()
    roleName = response_json["roleName"]
    roleDescription = response_json["roleDescription"]
    typeName1 = response_json["typeName1"]
    typeName2 = response_json["typeName2"]
    category = response_json["category"]
    focus = response_json["focus"]
    attention = response_json["attention"]
    engagement = response_json["engagement"]
    #print("role name: ", roleName)
    #print("role desc: ", roleDescription)
    #print("types: {} and {}".format(typeName1, typeName2))
    #print("category: ", category)
    #print("focus: ", focus)
    #print("attention: ", attention)
    #print("engagement: ", engagement)

    personality_txt = """---PERSONALITY: 
Based on the GUUM personality model, the patient falls into \
the {} or {} category. This designation signifies a personality \
blend of {} and {}. Notably, this individual demonstrates strengths \
in {}. The patient ability to focus on tasks is quite {}, and \
the patient has {} ability to maintain attention during conversations \
or group interactions. Additionally, the engagement levels during \
activities appear to be {}. Understanding these aspects of their \
personality can help tailor the therapeutic approach to better \
support the patient needs.

CURRENT EMOTIONS: 
{}.
---        
    """.format(category, roleName, typeName1, typeName2,
                roleDescription, focus, attention, engagement,
                emotion_str)
    return personality_txt
    

def main():
    # Check if arguments are provided
    if len(sys.argv) > 1:
        guum_ref_id = sys.argv[1]
        print("Input guum_reference_id:", guum_ref_id)
    else:
        print("USAGE: sense.py <guum_reference_id>")
        sys.exit()

    # hume emotion
    emotion_str = asyncio.run(extract_emotion(5))
    #print(emotion_str)

    guum_api_url = f"https://personality-api.unseenidentity.xyz/screening/result/{guum_ref_id}"
    response = get_api_data(guum_api_url, guum_api_key)
    if response.status_code == 200:
        personality_txt = construct_personality(response, emotion_str)
    else:
        personality_txt = """---PERSONALITY: 
the patient personality has not been identified.

CURRENT EMOTIONS:
{}.
---
        """.format(emotion_str)
    #print("Personality: \n", personality_txt)
    script = [{"role": "user", "content": personality_txt}]

    while True:
        # path to your image
        image_path = os.path.join(os.getcwd(), "./frames/frame.jpg")

        # getting the base64 encoding
        base64_image = encode_image(image_path)

        # analyze posture
        print("👀 Psychotherapists is watching...")
        analysis = analyze_image(base64_image, script=script)

        print("🎙️ Psychotherapists says:")
        print(" ")
        print(analysis)

        play_audio(analysis)

        script = script + [{"role": "assistant", "content": analysis}]

        # wait for 5 seconds
        time.sleep(500)


if __name__ == "__main__":
    main()
    