- [Implementation of a Contextual Chatbot in PyTorch with a Girlfriend-Like Personality](#implementation-of-a-contextual-chatbot-in-pytorch-with-a-girlfriend-like-personality)
- [original](#original)
  - [Features](#features)
  - [todo](#todo)
  - [Installation](#installation)
    - [Create a Virtual Environment](#create-a-virtual-environment)
    - [Activate the Virtual Environment](#activate-the-virtual-environment)
    - [Install PyTorch and Dependencies](#install-pytorch-and-dependencies)
  - [Usage](#usage)
  - [Customization](#customization)

# Implementation of a Contextual Chatbot in PyTorch with a Girlfriend-Like Personality

This is a simple chatbot implementation in PyTorch that's designed to provide a basic understanding of chatbots. The chatbot has been customized to have a girlfriend-like personality, with affectionate and loving responses, named Rose.

# original

the original model generation, and data is by [patrickloeber](https://github.com/patrickloeber/pytorch-chatbot)

## Features

- The chatbot uses a Feed Forward Neural Network with 2 hidden layers for its implementation.
- It's easy to customize the chatbot for your own use case. Just modify the `intents.json` file with possible patterns and responses to fit your desired conversation style (see below for more info).

The approach is inspired by this article and has been adapted to PyTorch: [https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077](https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077).

## todo

 - [ ] GUI for modifying the data set
 - [ ] GUI for chatting with the bot

## Installation

### Create a Virtual Environment

You can create a virtual environment using your preferred method (e.g., `conda` or `venv`).

```console
mkdir myproject
$ cd myproject
$ python3 -m venv venv
```

### Activate the Virtual Environment

For Mac / Linux:

```console
. venv/bin/activate
```

For Windows:

```console
venv\Scripts\activate
```

### Install PyTorch and Dependencies

You'll need to install PyTorch, which you can do from the [official website](https://pytorch.org/).

Additionally, you'll need to install the `nltk` library:

```console
pip install nltk
```

If you encounter an error during the first run, you may also need to install `nltk.tokenize.punkt`. Run the following command in your terminal:

```console
$ python
>>> import nltk
>>> nltk.download('punkt')
```

## Usage

To train the chatbot, run:

```shell
python train.py
```

This will create a `Rose.mdl` file. Afterward, you can interact with the chatbot by running:

```shell
python chat.py
```

## Customization

To customize the chatbot's conversation style, you can modify the `intents.json` file. Here, you can define new `tag`s, possible `patterns`, and their corresponding `responses`. If you make any changes to this file, remember to re-run the training process to apply your modifications.

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": [
        "Hi",
        "Hey",
        "How are you",
        "Is anyone there?",
        "Hello",
        "Good day"
      ],
      "responses": [
        "Hey, sweetie! 😘",
        "Hello, my love! 💕",
        "Hi there, handsome! 😊",
        "Hi there, how's your day been, darling? ❤️"
      ]
    },

  ]
}
```

Feel free to continue customizing the chatbot's responses and conversations to make it even more personalized and engaging, and remember, your imagination is the limit
