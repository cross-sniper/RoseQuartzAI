import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "Rose.mdl"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Define your chatbot's name and personality
bot_name = "Rose"
bot_personality = "Ai girlfriend"

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

emotions = {
    "happy": ["Great!", "I'm feeling cheerful today!", "😄"],
    "sad": ["I'm sorry to hear that.", "Cheer up! 😔", "😢"],
    "neutral": ["I'm here to help.", "Ask me anything.", "😐"],
    "surprised": ["Wow, that's unexpected!", "I didn't see that coming!", "😲"],
    "excited": ["That's fantastic!", "I'm thrilled for you!", "🎉"],
    "angry": ["Calm down, let's talk it out.", "I understand your frustration.", "😡"],
    "confused": ["I'm a bit puzzled too.", "Let's try to clarify.", "😕"],
    "grateful": ["Thank you for sharing!", "I appreciate your kindness.", "🙏"],
    # You can continue adding more emotions and responses as needed
}


print(f"{bot_name}: Let's chat! (type 'quit' to exit)")

while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                # Randomly select an emotion for the response
                emotion = random.choice(list(emotions.keys()))
                response = random.choice(intent['responses'])
                print(f"{bot_name} ({bot_personality}, {emotion}): {response} {emotions[emotion][-1]}")
    else:
        # If the chatbot doesn't understand, respond with a neutral emotion
        emotion = "neutral"
        response = "Sorry love, but I do not understand..."
        print(f"{bot_name} ({bot_personality}, {emotion}): {response} {emotions[emotion][-1]}")
