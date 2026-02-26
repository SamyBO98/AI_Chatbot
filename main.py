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

    #self.vocabulary = ["hello", "bye", "thanks"]
    #words = ["hello", "thanks"]
    #bag_of_words(words)  # [1, 0, 1]
    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]
    
    
    def parse_intents(self):
        #reduce words
        #lemmatizer = nltk.WordNetLemmatizer()

        #Load json
        if(os.path.exists((self.intents_path))):
            with open(self.intents_path, 'r') as f:
                intents_data = json.load(f)

            #loop through intents
            #save tag
            #set responses to the tag
            #self.intents = ["google", "greeting"]
            #self.intents_responses = {
            # "google": ["Redirecting to Google..."],
            # "greeting": ["Hello", "Hi there, how can I help?"]
            #}
            for intent in intents_data['intents']:
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_responses[intent['tag']] = intent['responses']
                for pattern in intent['patterns']:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append((pattern_words, intent['tag']))
                #eliminate duplicate
                self.vocabulary = sorted(set(self.vocabulary))
    


    #Input
    #self.vocabulary = ["be", "google", "good", "hello", "hi", "how", "internet", "there", "you"]
    #self.intents = ["google", "greeting"]
    #self.documents = [
        #(["google"], "google"),
        #(["search"], "google"),
        #(["internet"], "google"),
        #(["hi", "there"], "greeting"),
        #(["how", "be", "you"], "greeting")
    #]
    #Output
    # Pattern "google" → bag_of_words(["google"]) → [0,1,0,0,0,0,0,0,0] → tag "google" → index 0
    # Pattern "search" → [0,0,0,0,0,0,0,0,0] → index 0
    # Pattern "internet" → [0,0,0,0,0,0,1,0,0] → index 0
    # Pattern "hi there" → [0,0,0,0,1,0,0,1,0] → index 1
    # Pattern "how be you" → [1,0,0,0,0,1,0,0,1] → index 1

    #self.X = np.array([
    # [0,1,0,0,0,0,0,0,0],  # google
    # [0,0,0,0,0,0,0,0,0],  # search
    # [0,0,0,0,0,0,1,0,0],  # internet
    # [0,0,0,0,1,0,0,1,0],  # hi there
    # [1,0,0,0,0,1,0,0,1]   # how be you
    # ])
    #self.y = np.array([0, 0, 0, 1, 1])
    #self.X ready for input as data and self.y ready for input as labels
    def prepare_data(self):
        bags = []
        indices = []

        for document in self.documents:
            words = document[0]
            bag = self.bag_of_words(words)
            intent_index = self.intents.index[document[1]]
            bags.append(bag)
            indices.append(intent_index)
        self.X = np.array(bags)
        self.y = np.array(indices)


    def train_model(self, batch_size, lr, epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor,y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model=ChatBot(self.X.shape[1], len(self.intents))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs,batch_y)





#chatbot = ChatBotAssistant('intents.json')
#print(chatbot.tokenize_and_lemmatize('Hello world how are you, I am programming in Python today'))
#print(chatbot.tokenize_and_lemmatize('run runnings runs ran'))