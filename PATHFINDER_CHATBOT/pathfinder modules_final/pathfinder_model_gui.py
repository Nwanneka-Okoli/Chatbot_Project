# import the libraries
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
# tokenizes and removes punctuation
tokenizer = RegexpTokenizer(r'\w+')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
import numpy as np
import tensorflow
import tflearn
import random
import pickle
import time
import pathfinder_nlp as pnlp
import os
from tkinter import *

# load the dataset using the load_data function from pathfinder_nlp module
chatbot_data = pnlp.load_data()

# error handling for loading the dataset
try:
    with open("chatbot_data.pickle", "rb") as file:
        total_words, titles, x_train, target = pickle.load(file)
except:
    # total_words represents the patterns from the data set after tokenization
    total_words = []
    # title contains all the tags from the data set
    titles = []

    # patterns
    X = []
    # tags
    Y = []
    # loop through data
    for intent in chatbot_data["intents"]:
        for pattern in intent["patterns"]:
            words = pnlp.sentence_tokenize(pattern)
            # appending the tokenized words from our patterns to the total_words list
            total_words.extend(words)
            # tokenized patterns
            X.append(words)
            # tokenized tags
            Y.append(intent["tag"])
        # appending the tags to the titles list
        if intent["tag"] not in titles:
            titles.append(intent["tag"])
    total_words = [pnlp.word_stem(w) for w in total_words if w not in stopwords.words('english')]
    # sort and remove duplicates in the total_words and titles
    total_words = sorted(list(set(total_words)))
    titles = sorted(titles)
    # create x train and target
    x_train = []
    target = []

    # Initialised the bag with zeros
    empty_output = [0] * len(titles)
    for x, doc in enumerate(X):
        bag = []
        words = [pnlp.word_stem(w) for w in doc]
        # if the word exist in words append 1 in the bag else append 0
        for w in total_words:
            if w in words:
                bag.append(1)
            else:
                bag.append(0)
        output = empty_output[:]
        output[titles.index(Y[x])] = 1
        # append bag to the x_train and output to the target
        x_train.append(bag)
        target.append(output)

    # Transform x_train and target to an array using numpy.
    x_train = np.array(x_train)
    target = np.array(target)

    # Save total_words, titles, x_train and  target using pickle.
    with open("chatbot_data.pickle", "wb") as file:
        pickle.dump((total_words, titles, x_train, target), file)


# the function bag_words will transform the user input to ones and zeros.
def bag_words(sentences, total_words):
    bag_of_words = [0] * len(total_words)
    sentences_total_words = pnlp.sentence_tokenize(sentences)
    sentences_total_words = [pnlp.word_stem(w) for w in sentences_total_words if w not in stopwords.words('english')]
    for sentence in sentences_total_words:
        for i, w in enumerate(total_words):
            if w == sentence:
                bag_of_words[i] = 1
    return np.array(bag_of_words)


# create the model
tensorflow.compat.v1.reset_default_graph()
neural_network = tflearn.input_data(shape=[None, len(x_train[0])])
neural_network = tflearn.fully_connected(neural_network, 8)
neural_network = tflearn.fully_connected(neural_network, 8)
neural_network = tflearn.fully_connected(neural_network, len(target[0]), activation="softmax")
neural_network = tflearn.regression(neural_network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)
model = tflearn.DNN(neural_network)

# if the model already exist, load the model and not train again
try:
    # check if the model exists using os
    if os.path.exists('models/'):
        model.load("models/Pathfinder")
        print("the model has been loaded")
    else:
        raise ValueError
except ValueError:
    # Train the model, n_epoch is the number of iterations.
    # Batch_size is the size of the group the model uses in the training.
    # Show_metrics will show the accuracy and the loss.
    model.fit(x_train, target, n_epoch=500, batch_size=8, show_metric=True)
    # create directory for the model
    os.makedirs('models/')
    # save the model
    model.save("models/Pathfinder")
    print('The model has been created and saved')


# chat function will take the input from the user and returns response based on probabilities.
def chat(msg):
    bot_name = "Pathfinder"
    # bag_words function will transform the user input to bag of words.
    outcomes = model.predict([bag_words(msg, total_words)])
    # Using argmax the model will predict the class.
    outcomes_index = np.argmax(outcomes)
    title = titles[outcomes_index]

    for tag in chatbot_data["intents"]:
        if tag['tag'] == title:
            responses = tag['responses']
            # delay responses to make the conversation more human
            time.sleep(2)
            # The model will choose a random choice from the class predicted.
            results = random.choice(responses)
            return results


# create the graphical user interface
# function for the send interface
def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        results = chat(msg)
        ChatLog.insert(END, 'You:' + msg + '\n\n')  # I changed outcomes to outcomes1
        ChatLog.config(foreground="#442265", font=('Verdana', 12))
        ChatLog.insert(END, 'Pathfinder:' + results + '\n\n')  # I changed outcomes to outcomes1
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


base = Tk()
base.title('Pathfinder')
base.geometry("450x550")
base.resizable(width=FALSE, height=FALSE)

# Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="10", width=55, font="Arial", )
ChatLog.pack()
ChatLog.config(state=DISABLED)
# Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set
# Create Button to send message
SendButton = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#0000ff", activebackground="#3c9d9b", fg='#ffffff',
                    command=send)
# Create the box to enter message
EntryBox = Text(base, bd=0, bg="white", width=30, height="7", font="Arial")
# EntryBox.bind("<Return>", send)
# Place all components on the screen
scrollbar.place(x=375, y=7, height=387)
ChatLog.place(x=7, y=7, height=388, width=375)
EntryBox.place(x=127, y=400, height=85, width=259)
SendButton.place(x=7, y=400, height=84)
base.mainloop()