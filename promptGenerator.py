# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 23:53:11 2023

@author: andrewriznyk
"""

import random
import csv

# Agent 1 is making the offer for the trade -  The Offer Agent 


opponentTask = "stoneMasonery"

itemList = ["gold", "stones", "wood", "wheat", "coal", "iron", "aluminum", "horses"]

selfValue = 0
selfItem = "gold"
oppValue = 0
oppItem = "stones"

#Standardize generated datasets by making a uniform random seed for the generator
#Ensures uniformity across developers
random.seed(10)

def randomSentence(selfValue,selfItem,oppValue,oppItem):
    StartingVerbageSelf = [
            "I have " + str(selfValue) + " " + selfItem + " that I'm willing to trade for " + str(oppValue) + " " + oppItem + ". Would you be interested?",
            "How about a trade? I'll give you " + str(selfValue) + " " + selfItem + " in exchange for " + str(oppValue) + " " + oppItem + ".",
            "I'm interested in acquiring " + str(oppValue) + " " + oppItem + ". I have " + str(selfValue) + " " + selfItem + " that I can offer in exchange. Would you consider the trade?",
            "I'm willing to offer " + str(selfValue) + " " + selfItem + " for " + str(oppValue) + " " + oppItem + ". Are you open to a trade?",
            "I have " + str(selfValue) + " " + selfItem + " that I'm looking to exchange for " + str(oppValue) + " " + oppItem + ". Would you like to make a trade?",
            "If you're interested in trading, I have " + str(selfValue) + " " + selfItem + " that I'm willing to offer for " + str(oppValue) + " " + oppItem + ".",
            "I'm looking to acquire " + str(oppValue) + " " + oppItem + " and am willing to trade " + str(selfValue) + " " + selfItem + " for them. Would you be interested?",
            "I have " + str(selfValue) + " " + selfItem + " that I can trade for " + str(oppValue) + " " + oppItem + ". Are you interested in making a deal?",
            "Would you be willing to trade " + str(oppValue) + " " + oppItem + " for " + str(selfValue) + " " + selfItem + " ? Let me know.",
            "I'm interested in trading " + str(selfValue) + " " + selfItem + " for " + str(oppValue) + " " + oppItem + ". What do you think?",
            "I'd like to trade " + str(selfValue) + " " + selfItem + " for " + str(oppValue) + " " + oppItem + ". Let me know if you're interested.",
            "I have " + str(selfValue) + " " + selfItem + " and I'm interested in trading them for " + str(oppValue) + " " + oppItem + ". Would you consider the offer?",
            "How about a trade? I have " + str(selfValue) + " " + selfItem + " that I can offer in exchange for " + str(oppValue) + " " + oppItem + ".",
            "I'm willing to trade " + str(selfValue) + " " + selfItem + " for " + str(oppValue) + " " + oppItem + ". Let me know if you're interested.",
            "I'd like to make a trade. I have " + str(selfValue) + " " + selfItem + " that I can offer for " + str(oppValue) + " " + oppItem + ".",
            "Are you open to a trade? I have " + str(selfValue) + " " + selfItem + " that I'm willing to exchange for " + str(oppValue) + " " + oppItem + ".",
            "I'm looking to acquire " + str(oppValue) + " " + oppItem + " and am willing to offer " + str(selfValue) + " " + selfItem + " in exchange. Would you be interested?",
            "I'd like to offer " + str(selfValue) + " " + selfItem + " for " + str(oppValue) + " " + oppItem + ". Are you open to a trade?",
            "I have " + str(selfValue) + " " + selfItem + " that I'm interested in trading for " + str(oppValue) + " " + oppItem + ". Would you consider the offer?",
            "I'm willing to exchange " + str(selfValue) + " " + selfItem + " for " + str(oppValue) + " " + oppItem + ". Let me know if you're interested.",
            "I'm looking to acquire " + str(oppValue) + " " + oppItem + " and can offer " + str(selfValue) + " " + selfItem + " in exchange. Would you be interested in a trade?",
            "I'm interested in trading " + str(selfValue) + " " + selfItem + " for " + str(oppValue) + " " + oppItem + ". Let me know if you're interested.",
            "How about a trade? I have " + str(selfValue) + " " + selfItem + " that I can offer in exchange for " + str(oppValue) + " " + oppItem + ".",
            "I have " + str(selfValue) + " " + selfItem + " and am interested in trading them for " + str(oppValue) + " " + oppItem + ". What do you think?",
            "I'd like to offer " + str(selfValue) + " " + selfItem + " for " + str(oppValue) + " " + oppItem + ". Would you be interested in making a trade?",
            "I'm willing to trade " + str(selfValue) + " " + selfItem + " for " + str(oppValue) + " " + oppItem + ". Let me know if you're interested.",
            "I have " + str(selfValue) + " " + selfItem + " and am looking to trade them for " + str(oppValue) + " " + oppItem + ". Would you consider the offer?",
            "I'm interested in acquiring " + str(oppValue) + " " + oppItem + " and am willing to offer " + str(selfValue) + " " + selfItem + " in exchange. Are you open to a trade?",
            "I'd like to make a trade. I have " + str(selfValue) + " " + selfItem + " that I'm willing to offer for " + str(oppValue) + " " + oppItem + ".",
            "I'm willing to offer " + str(selfValue) + " " + selfItem + " for " + str(oppValue) + " " + oppItem + ". Are you interested in making a trade?",
            "How about we make a deal? I have " + str(selfValue) + " " + selfItem + " that I can trade for " + str(oppValue) + " " + oppItem + ".",
            "I'm interested in trading " + str(selfValue) + " " + selfItem + " for " + str(oppValue) + " " + oppItem + ". Would you be open to a trade?",
            "I have " + str(selfValue) + " " + selfItem + " that I'm willing to exchange for " + str(oppValue) + " " + oppItem + ". Let me know if you're interested in a trade.",
            "I'd like to acquire " + str(oppValue) + " " + oppItem + " and can offer " + str(selfValue) + " " + selfItem + " in exchange. Would you be interested in trading?",
            "I'm looking to diversify my assets and would like to trade " + str(selfValue) + " " + selfItem + " for " + str(oppValue) + " " + oppItem + ". What do you think?",
            "I'm interested in acquiring " + str(oppValue) + " " + oppItem + " and am willing to offer " + str(selfValue) + " " + selfItem + " in exchange. Are you open to a trade?",
            "I have " + str(selfValue) + " " + selfItem + " that I'm willing to trade for " + str(oppValue) + " " + oppItem + ". Would you be interested in making a deal?",
            "I'd like to acquire " + str(oppValue) + " " + oppItem + " and can offer " + str(selfValue) + " " + selfItem + " in exchange. Let me know if you're interested in a trade.",
            "I'm looking to trade " + str(selfValue) + " " + selfItem + " for " + str(oppValue) + " " + oppItem + ". Would you consider the offer?",
            "I have " + str(selfValue) + " " + selfItem + " and am interested in exchanging them for " + str(oppValue) + " " + oppItem + ". Are you open to a trade?",
            "I'm willing to offer " + str(selfValue) + " " + selfItem + " for " + str(oppValue) + " " + oppItem + ". Let me know if you're interested in making a trade.",
            "I'd like to trade " + str(selfValue) + " " + selfItem + " for " + str(oppValue) + " " + oppItem + ". What do you think?",
            "I'm interested in acquiring " + str(oppValue) + " " + oppItem + " and am willing to trade " + str(selfValue) + " " + selfItem + " in exchange. Would you consider the offer?",
            "How about a trade? I have " + str(selfValue) + " " + selfItem + " that I'm willing to exchange for " + str(oppValue) + " " + oppItem + ".",
            "I'm looking to acquire " + str(oppValue) + " " + oppItem + " and am willing to offer " + str(selfValue) + " " + selfItem + " in exchange. Let me know if you're interested in a trade.",
            "I'm interested in trading " + str(selfValue) + " " + selfItem + " for " + str(oppValue) + " " + oppItem + ". Would you be open to a trade?",
            "I have " + str(selfValue) + " " + selfItem + " that I can trade for " + str(oppValue) + " " + oppItem + ". Are you interested?",
            "I'd like to make a trade. I have " + str(selfValue) + " " + selfItem + " that I'm willing to offer for " + str(oppValue) + " " + oppItem + ".",
            "I'm willing to trade " + str(selfValue) + " " + selfItem + " for " + str(oppValue) + " " + oppItem + ". Would you be interested in making a deal?",
            "I'm interested in acquiring " + str(oppValue) + " " + oppItem + " and am willing to offer " + str(selfValue) + " " + selfItem + " in exchange. Are you open to a trade?",
            "I have " + str(selfValue) + " " + selfItem + " that I'm looking to trade for " + str(oppValue) + " " + oppItem + ". Let me know if you're interested.",
            "I'd like to offer " + str(selfValue) + " " + selfItem + " for " + str(oppValue) + " " + oppItem + ". Would you consider the trade?",
            "I'm willing to exchange " + str(selfValue) + " " + selfItem + " for " + str(oppValue) + " " + oppItem + ". Let me know if you're interested in a trade.",
            "I'm interested in trading " + str(selfValue) + " " + selfItem + " for " + str(oppValue) + " " + oppItem + ". What do you think?",
            "I'm willing to trade " + str(selfValue) + " " + selfItem + " for " + str(oppValue) + " " + oppItem + ". Is that something you'd be interested in?",
            "I'd like to acquire " + str(oppValue) + " " + oppItem + " and am willing to offer " + str(selfValue) + " " + selfItem + " in exchange. Are you open to a trade?",
            "I have " + str(selfValue) + " " + selfItem + " that I'm willing to trade for " + str(oppValue) + " " + oppItem + ". Would you consider the offer?",
            "How about we make a deal? I have " + str(selfValue) + " " + selfItem + " and I'm interested in trading them for " + str(oppValue) + " " + oppItem + ".",
            "I'm looking to diversify my holdings and would like to trade " + str(selfValue) + " " + selfItem + " for " + str(oppValue) + " " + oppItem + ". Would you be interested?",
            "I have " + str(selfValue) + " " + selfItem + " that I'm looking to exchange for " + str(oppValue) + " " + oppItem + ". Are you open to a trade?",
            "I'd like to offer " + str(selfValue) + " " + selfItem + " in exchange for " + str(oppValue) + " " + oppItem + ". Let me know if you're interested in making a trade.",
            "I'm interested in acquiring " + str(oppValue) + " " + oppItem + " and am willing to trade " + str(selfValue) + " " + selfItem + " in exchange. Would you be open to that?",
            "I'm looking to trade " + str(selfValue) + " " + selfItem + " for " + str(oppValue) + " " + oppItem + ". Would you consider the offer?",
            "I'm willing to exchange " + str(selfValue) + " " + selfItem + " for " + str(oppValue) + " " + oppItem + ". Is that something you'd be interested in?",
            "I have " + str(selfValue) + " " + selfItem + " that I'm willing to offer in exchange for " + str(oppValue) + " " + oppItem + ". Let me know if you're interested in a trade.",
            "I'm interested in making a trade. I have " + str(selfValue) + " " + selfItem + " that I can offer for " + str(oppValue) + " " + oppItem + ". Would you consider it?",
            "I'd like to acquire " + str(oppValue) + " " + oppItem + " and am willing to trade " + str(selfValue) + " " + selfItem + " in exchange. Let me know if you're interested in a deal.",
            "I have " + str(selfValue) + " " + selfItem + " that I'm looking to trade for " + str(oppValue) + " " + oppItem + ". Would you be open to that exchange?",
            "I'm interested in trading " + str(selfValue) + " " + selfItem + " for " + str(oppValue) + " " + oppItem + ". What do you think?",
            "How about we make a trade? I'm willing to offer " + str(selfValue) + " " + selfItem + " in exchange for " + str(oppValue) + " " + oppItem + ".",
            "I'd like to acquire " + str(oppValue) + " " + oppItem + " and am willing to offer " + str(selfValue) + " " + selfItem + " in exchange. Would you be interested in a trade?",
            "I'm looking to diversify my assets and would like to trade " + str(selfValue) + " " + selfItem + " for " + str(oppValue) + " " + oppItem + ". Would you consider it?",
            "I have " + str(selfValue) + " " + selfItem + " that I'm willing to trade for " + str(oppValue) + " " + oppItem + ". Are you open to a deal?",
            "I'm interested in acquiring " + str(oppValue) + " " + oppItem + " and am willing to offer " + str(selfValue) + " " + selfItem + " in exchange. Let me know if you're interested in a trade.",
            "I'd like to make a trade. I have " + str(selfValue) + " " + selfItem + " that I'm willing to offer for " + str(oppValue) + " " + oppItem + ". Would you be interested?",
            "I'm willing to trade " + str(selfValue) + " " + selfItem + " for " + str(oppValue) + " " + oppItem + ". Would you consider the offer?",
            "I'm interested in trading " + str(selfValue) + " " + selfItem + " for " + str(oppValue) + " " + oppItem + ". Let me know if you'd like to make a deal.",
            "I have " + str(selfValue) + " " + selfItem + " and am looking to exchange them for " + str(oppValue) + " " + oppItem + ". Would you be open to a trade?",
            "I'd like to acquire " + str(oppValue) + " " + oppItem + " and am willing to offer " + str(selfValue) + " " + selfItem + " in exchange. Are you interested in making a trade?"
        ]
    return random.choice(StartingVerbageSelf)


with open('./data/tradePrompts.csv', 'w') as csvfile: 
    writer = csv.writer(csvfile) 
    writer.writerow(["Trade","OfferAmount","OfferItem","DesiredAmount","DesiredItem"])

    for x in range(0,5000):
        selfValue = random.randint(1, 50)
        oppValue = random.randint(1, 50)
        selfItem = random.choice(itemList)
        oppItem = random.choice(itemList)
        while (oppItem == selfItem):
            oppItem = random.choice(itemList)
        
        writer.writerow([randomSentence(selfValue,selfItem,oppValue,oppItem), selfValue , selfItem, oppValue, oppItem])