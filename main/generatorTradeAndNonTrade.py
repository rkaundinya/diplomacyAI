#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 23:53:11 2023

@author: andrewriznyk
@author: rkaundinya
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
    return "\"" + random.choice(StartingVerbageSelf) + "\""

def randomNonTrade(selfValue, selfItem):
    StartingVerbageSelf = [
            "I have " + str(selfValue) + " " + selfItem + " in my possession.",
            "I am in possession of " + str(selfValue) + " " + selfItem + ".",
            "I possess a total of " + str(selfValue) + " " + selfItem + ".",
            "" + str(selfValue) + " " + selfItem + " are in my ownership.",
            "I own " + str(selfValue) + " " + selfItem + ".",
            "My collection includes " + str(selfValue) + " " + selfItem + ".",
            "I have at my disposal " + str(selfValue) + " " + selfItem + ".",
            "There are " + str(selfValue) + " " + selfItem + " in my possession.",
            "I hold " + str(selfValue) + " " + selfItem + ".",
            "My inventory includes " + str(selfValue) + " " + selfItem + ".",
            "" + str(selfValue) + " " + selfItem + " are currently in my possession.",
            "I currently have " + str(selfValue) + " " + selfItem + ".",
            "" + str(selfValue) + " " + selfItem + " belong to me.",
            "I am the owner of " + str(selfValue) + " " + selfItem + ".",
            "I am the proud possessor of " + str(selfValue) + " " + selfItem + ".",
            "" + str(selfValue) + " " + selfItem + " are under my control.",
            "" + str(selfValue) + " " + selfItem + " are in my hands.",
            "I currently hold onto " + str(selfValue) + " " + selfItem + ".",
            "I am in charge of " + str(selfValue) + " " + selfItem + ".",
            "" + str(selfValue) + " " + selfItem + " are in my possession at the moment.",
            "My stock includes " + str(selfValue) + " " + selfItem + ".",
            "" + str(selfValue) + " " + selfItem + " are in my custody.",
            "I have a total of " + str(selfValue) + " " + selfItem + " in my possession.",
            "I have possession of " + str(selfValue) + " " + selfItem + ".",
            "" + str(selfValue) + " " + selfItem + " are under my possession.",
            "I have in my possession a collection of " + str(selfValue) + " " + selfItem + ".",
            "There are " + str(selfValue) + " " + selfItem + " that I possess and treasure dearly.",
            "I am the proud owner of " + str(selfValue) + " beautiful " + selfItem + ".",
            "Among my prized possessions are " + str(selfValue) + " stunning " + selfItem + ".",
            "I have amassed a small but impressive collection of " + str(selfValue) + " " + selfItem + ".",
            "I possess a handful of " + str(selfValue) + " unique and exquisite " + selfItem + ".",
            "There are " + str(selfValue) + " gems that I hold dear to my heart.",
            "My collection of " + selfItem + " includes " + str(selfValue) + " rare and precious finds.",
            "I have come into possession of " + str(selfValue) + " " + selfItem + " that hold a special place in my heart.",
            "My collection boasts " + str(selfValue) + " magnificent " + selfItem + " that are truly one-of-a-kind.",
            "My collection of " + str(selfValue) + " " + selfItem + " is nothing short of magnificent.",
            "The " + str(selfValue) + " " + selfItem + " I possess are absolutely stunning.",
            "I am the proud owner of " + str(selfValue) + " exquisite " + selfItem + ".",
            "My collection of " + str(selfValue) + " " + selfItem + " is an awe-inspiring sight to behold.",
            "The " + str(selfValue) + " " + selfItem + " in my possession are a testament to my impeccable taste.",
            "I have " + str(selfValue) + " " + selfItem + " that are truly one-of-a-kind.",
            "The " + str(selfValue) + " " + selfItem + " that I possess are incredibly rare and valuable.",
            "My " + str(selfValue) + " precious " + selfItem + " are nothing less than works of art.",
            "The " + str(selfValue) + " " + selfItem + " in my possession are an impressive feat of geological fortune.",
            "I am fortunate to have " + str(selfValue) + " remarkable " + selfItem + " that are the envy of many.",
            "My possession of " + str(selfValue) + " " + selfItem + " is like having the seven wonders of the mineral world at my fingertips.",
            "The " + str(selfValue) + " " + selfItem + " I possess are like a sparkling constellation in my collection.",
            "The " + str(selfValue) + " " + selfItem + " that I have amassed are like " + str(selfValue) + " pillars of strength that hold up my empire.",
            "My " + str(selfValue) + " " + selfItem + " are like a family of gems that I cherish and adore.",
            "The " + str(selfValue) + " " + selfItem + " that I possess are like the rarest and most precious treasure hidden deep in a secret vault.",
            "The " + str(selfValue) + " " + selfItem + " I own are like a powerful talisman that brings me good fortune and prosperity.",
            "My " + str(selfValue) + " " + selfItem + " are like a reflection of my soul - unique, brilliant, and full of depth.",
            "The " + str(selfValue) + " " + selfItem + " that I have acquired are like a manifestation of my determination and hard work.",
            "The " + str(selfValue) + " " + selfItem + " that I possess are like a glittering crown that adorns my collection.",
            "My " + str(selfValue) + " " + selfItem + " are like a symphony of colors and textures that leave me breathless every time I look at them.",
            "It's hard not to be impressed by the stunning collection of " + str(selfValue) + " " + selfItem + " that you have amassed.",
            "I couldn't help but notice the remarkable array of " + str(selfValue) + " " + selfItem + " in your possession - truly a sight to behold.",
            "My eyes were immediately drawn to the breathtaking collection of " + str(selfValue) + " " + selfItem + " that you have acquired.",
            "Your possession of " + str(selfValue) + " stunning " + selfItem + " is nothing short of remarkable.",
            "It's not every day that one comes across a collection of " + str(selfValue) + " " + selfItem + " as impressive as yours.",
            "Your ownership of " + str(selfValue) + " dazzling " + selfItem + " is truly an inspiration.",
            "The " + str(selfValue) + " " + selfItem + " that you possess are like a shining beacon that captures everyone's attention.",
            "I was awestruck by the sight of the " + str(selfValue) + " remarkable " + selfItem + " that you have collected.",
            "I couldn't help but marvel at the " + str(selfValue) + " gorgeous " + selfItem + " that are under your possession.",
            "Your possession of " + str(selfValue) + " stunning " + selfItem + " is like having a treasure trove of riches at your fingertips.",
            "The " + str(selfValue) + " " + selfItem + " in your possession are like the precious jewels in a king's crown.",
            "I was taken aback by the sheer beauty and brilliance of the " + str(selfValue) + " " + selfItem + " that you possess.",
            "Your ownership of " + str(selfValue) + " exquisite " + selfItem + " is like having a window into the natural beauty of the earth.",
            "The " + str(selfValue) + " " + selfItem + " in your collection are like a dazzling rainbow of colors and textures.",
            "I couldn't help but admire the unique and intricate details of each of the " + str(selfValue) + " " + selfItem + " that you have amassed.",
            "The " + str(selfValue) + " " + selfItem + " that you possess are like a work of art, each one a masterpiece in its own right.",
            "Your collection of " + str(selfValue) + " " + selfItem + " is like a magical portal to another world, full of wonder and awe.",
            "The " + str(selfValue) + " " + selfItem + " in your possession are like a rare and precious gift from the earth itself.",
            "Your possession of " + str(selfValue) + " magnificent " + selfItem + " is like a symbol of your strength and resilience, shining bright and steadfast.",
            "You are in possession of a stunning collection of " + str(selfValue) + " " + selfItem + ".",
            "There are " + str(selfValue) + " " + selfItem + " that you possess and treasure dearly.",
            "You are the proud owner of " + str(selfValue) + " beautiful " + selfItem + ".",
            "Among your prized possessions are " + str(selfValue) + " stunning " + selfItem + ".",
            "You have amassed a small but impressive collection of " + str(selfValue) + " " + selfItem + ".",
            "You possess a handful of " + str(selfValue) + " unique and exquisite " + selfItem + ".",
            "There are " + str(selfValue) + " gems that you hold dear to your heart.",
            "Your collection of " + selfItem + " includes " + str(selfValue) + " rare and precious finds.",
            "You have come into possession of " + str(selfValue) + " " + selfItem + " that hold a special place in your heart.",
            "Your collection boasts " + str(selfValue) + " magnificent " + selfItem + " that are truly one-of-a-kind.",
            "Let's make a trade and both come out ahead.",
            "Trading can be a great way to diversify your assets.",
            "I'm open to a trade if the offer is right.",
            "The art of trading is all about finding a win-win solution.",
            "Trade negotiations can be a delicate balance.",
            "Trades can help both parties get what they need.",
            "A good trade can leave both parties feeling satisfied.",
            "Trade agreements can be complex, but well worth it in the end.",
            "The beauty of a trade is that both parties can benefit from it.",
            "Trading can be a way to get something you need without spending money.",
            "Trading is a valuable skill to have in today's economy.",
            "A well-executed trade can be a smart investment strategy.",
            "The key to a successful trade is finding the right match.",
            "Trades can be a great way to acquire new assets or divest old ones.",
            "The art of negotiation is crucial when it comes to trade.",
            "A fair trade can be mutually beneficial for both parties.",
            "Trading is an ancient practice that remains relevant today.",
            "Trades can be a way to access goods or services that might otherwise be out of reach.",
            "The world of trade is constantly evolving, with new opportunities emerging all the time.",
            "Trades can be a way to build relationships and establish trust between parties.",
            "The potential for profit is one of the key attractions of trading.",
            "Trades can be a way to balance out your portfolio.",
            "The success of a trade often depends on timing and market conditions.",
            "Trading can be a way to turn your assets into something more valuable.",
            "A trade can be a creative way to solve a problem or meet a need.",
            "Trades can be a way to build connections and networks with other traders.",
            "The principles of fair trade and ethical trading are increasingly important in today's world.",
            "Trading can be a way to explore new markets and expand your horizons.",
            "Trades can be a way to pool resources and collaborate with others.",
            "The psychology of trading, including risk management and decision-making, is a fascinating area of study."
        ]
    return "\"" + random.choice(StartingVerbageSelf) + "\""



with open('../data/tradeAndNonTradePrompts.csv', 'w') as csvfile: 
    writer = csv.writer(csvfile) 
    writer.writerow(["Trade","isTrade"])

    for x in range(0,3000):
        selfValue = random.randint(1, 50)
        oppValue = random.randint(1, 50)
        selfItem = random.choice(itemList)
        oppItem = random.choice(itemList)
        while (oppItem == selfItem):
            oppItem = random.choice(itemList)
        
        writer.writerow([randomSentence(selfValue,selfItem,oppValue,oppItem), 1])
    
    for x in range(0,3000):
        selfValue = random.randint(1, 50)
        selfItem = random.choice(itemList)

        writer.writerow([randomNonTrade(selfValue,selfItem), 0])