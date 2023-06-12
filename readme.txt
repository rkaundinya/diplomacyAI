Diplomacy AI NLP Agent
by Ram Eshwar Kaundinya, Andrew Ryznik, and Kyle Coughlin

Project Structure:
- Main project files exist in the root directory
- Developer specific versions of files exist within developers directory
- Datasets exist in data folder
- Graphs and Results folder (this is empty, but may be filled in the future!)

Generator Script:
- generatorTradeAndNonTrade.py
This script generates the data on which the NERClassifier trains and on which the baseline tests are run. 
It draws randomly from chatGPT template prompts of two kinds - trades and non-trades. One loop uses both 
sets to create a tradeAndNonTradePrompts.csv on which the baseline trade classifiers are run. 
Another loop generates tradeAndNotTradeAnnotated.csv which saves trade and not trade NER annotated data 
into both a csv and numpy .npy file (used by the NERClassifier later to load in the data in trainable format).

Created Data Files:
Located in the data folder - there are two csv's which are described above. One is a trade and no trade annotated
csv with prompts and their binary classifications. Another is a trade and not trade annotated csv with NER annotations
to prompts. Finally there is a NPY_Files folder which is a bit of a misnomer since it contains both saved tensor
data and numpy data. The .npy file is the numpy formatted data from tradeAndNotTradeAnnotated.csv. The 
tensor data is embeddings.pt which is simply word embedding vectors saved off so that when the game loads this
to load the model for the application, it doesn't need to reload from a database. 

Baseline Tests:
baselineLogisticTradeAndNoTrade.py trains a logistic regression model on the trade and no trade binary 
classification data and reports results. 
tradeClassifierBaselines.py has more baseline tests for trade and no trade binary classification. This file
takes naive bayes, random forest, and decision tree approaches to classification and reports results. 
Finally, RuleBaseBaseline.py uses a rule based method to extract items and amounts from prompts by 
recognizing keywords or specific prompt template styles. 

Building Block Files:
These include utilities.py and vocab.py. These are some commonly used utility functions (like to_gpu) extracted
into a separate file and a vocabulary class extract into a separate file for object oriented programming purpsoes. 

Model:
EmbNet.py contains the class and structure for the NER Classifier model. The architecture here is very simple - 
it takes in pre-trained word embeddings and builds a network with a linear, sigmoid activation, and linear prediction 
layer. 
NERClassifier.py uses this network architecture to feed in the tokenized NER Annotated data from the generator script
to this model. The model uses this to train with cross entropy loss and classifies predictions into 5 categories, 
one for each item and value and one for not an item or value. 
The model's parameters are saved to a file in case you want to just run diagnostic tests on a pre-trained model. The flag
SHOULD_TRAIN can be set to True if you wish to retrain the model. Also the PATH of the data can be changed as needed. 

The Game:
The game consists of player.py which holds logic related to each player agent and their respective resource management.
app.py is the main game code which has some functions to parse through user input text and the main game code. 

Game Rules:
- Default to win conditions of whomever reaches above 7 points first
- Ties are possible between player and AI
- If the player goes below -7 points, the game ends

Game Setup:
- Resources are allocated such that they total to 10 items for each player
- One shared resource exists between both players
- Values are allocated such that they total 10 for each player
- The shared resource is assigned the median value of 3 random value numbers (totaling to 10) for 
  each player
- The player and AI have different values for each resource
- The player does not see the values the AI has for resources (and likewise neither does the AI)
- Each player has two unique resources other than the shared resource

AI Policy:
The AI policy is simple, if a deal gives it more than it loses, then it will accept. Otherwisee it will reject. 
Some simple logic exists to detect if the player is offering an item he/she does not have or is attempting to 
offer more than they have.
Same logic guards against asking for more than the AI has or a resource the AI does not have. 

Game Controls:
Enter your deal when prompted - at any time type q to quit, i to show inventories of player and AI, v to 
print your value table (how much you value each resource). 

Challenges and Limitations:
The program is not robust in recognizing all possible ways of offering deals. There are some ways of prompting
which cause confusion on which item you are asking for and which is being traded. 
Conversations are only understood in isolation meaning the AI holds no history of the conversation and cannot
associate words across prompts. 
The program only deals with single item for single item trades. 

Note - you will need nltk and some libraries from that for certain scripts