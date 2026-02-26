import os
import json
import random
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#Run once
#nltk.download('punkt_tab')

class ChatBot(nn.Module):
    
    #output_size = number of intentions
    def __init__(self, input_size, output_size):
        super(ChatBot, self).__init__()

        #128 neurons
        #represent string as numbers
        self.fc1 = nn.Linear(input_size,128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        #Break linearity
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        #learn a lot of patterns
        #input to 128 values with Linear function then ReLu (negative value to 0)
        x = self.relu(self.fc1(x))
        #no overfitting
        x = self.dropout(x)
        #Compress to keep essential 128 -> 64
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        #Final decision
        x = self.fc3(x)

class ChatBotAssistant():

    def __init__(self, intents_path, function_mappings=None):
        self.model = None
        self.intents_path = intents_path
        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}
        self.function_mappings = function_mappings
        self.X = None
        self.y = None
    
    @staticmethod
    def tokenize_and_lemmatize(text):
        #Get all instance of words
        # cars -> car
        # running -> run
        # better -> good
        lemattizer = nltk.WordNetLemmatizer()
        #Split text in tokens
        words = nltk.word_tokenize(text)
        words = [lemattizer.lemmatize(word.lower()) for word in words]

        return words

    @staticmethod
    def bag_of_words(words, vocab):
        return [1 if word in words else 0 for word in vocab]
    
    
    def parse_intents(self):
        lemmatizer = nltk.WordNetLemmatizer()

        if(os.path.exists((self.intents_path))):
            with open(self.intents_path, 'r') as f:
                intents_data = json.load(f)

            #loop through intents
            #save tag
            #set responses to the tag
            for intent in intents_data['intents']:
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_responses[intent['tag']] = intent['responses']
                for pattern in intent['patterns']:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append((pattern_words, intent['tag']))
                #eliminate duplicate
                self;vocabulary = sorted(set(self.vocabulary))



#chatbot = ChatBotAssistant('intents.json')
#print(chatbot.tokenize_and_lemmatize('Hello world how are you, I am programming in Python today'))
#print(chatbot.tokenize_and_lemmatize('run runnings runs ran'))