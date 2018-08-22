# -*- coding: utf-8 -*-
from __future__ import division, print_function
from sklearn.base import BaseEstimator,TransformerMixin
import numpy as np
from preprocess import ark_tweet_tokenizer
import sys
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')

__all__ = ['LIWCFeatures']

emotions = {
    'Funct' : ["a", "about", "above", "absolutely", "across", "actually", "after", "again", "against", "ahead", "aint", "ain't", "all", "allot", "along", "alot", "also", "although", "am", "among", "an", "and", "another", "any", "anybod", "anymore", "anyone", "anything", "anyway", "anywhere", "apparently", "are", "arent", "aren't", "around", "as", "at", "atho", "atop", "away", "back", "basically", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "behind", "being", "below", "beneath", "beside", "besides", "best ", "between", "beyond", "billion", "both", "bunch", "but", "by", "can", "cannot", "cant", "can't", "cetera", "clearly", "completely", "constantly", "could", "couldnt", "couldn't", "couldve", "could've", "couple", "cuz", "definitely", "despite", "did", "didnt", "didn't", "difference", "do", "does", "doesnt", "doesn't", "doing", "done", "dont", "don't", "doubl", "down", "dozen", "during", "each", "eight", "either", "eleven", "else", "enough", "entire", "equal", "especially", "etc", "even", "eventually", "ever", "every", "everybod", "everyone", "everything", "example", "except", "extenet", "extra", "extremely", "fairly", "few", "fift", "first", "firstly", "firsts", "five", "for", "form", "four", "frequently", "from", "full", "generally", "greater", "greatest", "had", "hadnt", "hadn't", "half", "has", "hasnt", "hasn't", "have", "havent", "haven't", "having", "hed", "he'd", "her", "here", "heres", "here's", "hers", "herself", "hes", "highly", "him", "himself", "his", "hopefully", "how", "however", "hundred", "i", "Id", "I'd", "if", "I'll", "Im", "I'm", "immediately", "in", "infinit", "inside", "insides", "instead", "into", "is", "isnt", "isn't", "it", "itd", "it'd", "item", "itll", "it'll", "its", "it's", "itself", "ive", "I've", "just", "lack", "lately", "least", "less", "let", "lets", "let's", "loads", "lot", "lotof", "lots", "lotsa", "lotta", "main", "many", "may", "maybe", "me", "might", "mightve", "might've", "million", "mine", "more", "most", "mostly", "much", "mucho", "must", "mustnt", "must'nt", "mustn't", "mustve", "must've", "my", "myself", "near", "nearly", "neednt", "need'nt", "needn't", "negat", "neither", "never", "nine", "no", "nobod", "none", "nope", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "ones ", "oneself", "only", "onto", "or", "other", "others", "otherwise", "ought", "oughta", "oughtnt", "ought'nt", "oughtn't", "oughtve", "ought've", "our", "ours", "ourselves", "out", "outside", "over", "own", "part", "partly", "perhaps", "piec", "plenty", "plus", "primarily", "probably", "quarter", "quick", "rarely", "rather", "really", "remaining", "rest", "same", "second", "section", "seriously", "seven", "several", "shall", "shant", "shan't", "she", "she'd", "she'll", "shes", "she's", "should", "shouldnt", "should'nt", "shouldn't", "shouldve", "should've", "since", "singl", "six", "so", "some", "somebod", "somehow", "someone", "something", "somewhat", "somewhere", "soon", "sooo", "still", "stuff", "such", "ten", "tenth", "than", "that", "thatd", "that'd", "thatll", "that'll", "thats", "that's", "the", "thee", "their", "them", "themselves", "then", "there", "theres", "there's", "these", "they", "theyd", "they'd", "theyll", "they'll", "theyre", "they're", "theyve", "they've", "thine", "thing", "third", "thirt", "this", "tho", "those", "thou", "though", "thousand", "thoust", "three", "through", "thru", "thy", "til", "till", "to", "ton", "tons", "too", "total", "totally", "toward", "trillion", "tripl", "truly", "twel", "twent", "twice", "two", "uhuh", "under", "underneath", "unique", "unless", "until", "unto", "up", "upon", "us", "usually", "various", "very", "wanna", "was", "wasnt", "wasn't", "we", "we'd", "well", "we'll", "were", "we're", "weren't", "weve", "we've", "what", "whatever", "whats", "what's", "when", "whenever", "where", "whereas", "wheres", "where's", "whether", "which", "whichever", "while", "who", "whod", "who'd", "whole", "wholl", "who'll", "whom", "whose", "will", "with", "within", "without", "wont", "won't", "worst", "would", "wouldnt", "wouldn't", "wouldve", "would've", "ya", "yall", "y'all", "ye", "yet", "you", "youd", "you'd", "youll", "you'll", "your ", "youre", "you're", "yours", "youve", "you've", "zero", "zillion", "he's"],

    'Pronoun' : ["anybod", "anyone", "anything", "everybod", "everyone", "everything", "hed", "he'd", "her", "hers", "herself", "hes", "he's", "him", "himself", "his", "i", "Id", "I'd", "I'll", "Im", "I'm", "it", "itd", "it'd", "itll", "it'll", "its", "it's", "itself", "ive", "I've", "lets", "let's", "me", "mine", "my", "myself", "nobod", "oneself", "other", "others", "our", "ours", "ourselves", "she", "she'd", "she'll", "shes", "she's", "somebod", "someone", "something", "somewhere", "stuff", "that", "thatd", "that'd", "thatll", "that'll", "thats", "that's", "thee", "their", "them", "themselves", "these", "they", "theyd", "they'd", "theyll", "they'll", "theyve", "they've", "thine", "thing", "this", "those", "thou", "thoust", "thy", "us", "we", "we'd", "we'll", "we're", "weve", "we've", "what", "whatever", "whats", "what's", "which", "whichever", "who", "whod", "who'd", "wholl", "who'll", "whom", "whose", "ya", "yall", "y'all", "ye", "you", "youd", "you'd", "youll", "you'll", "your ", "youre", "you're", "yours", "youve", "you've"],

    'Ppron' : ["hed", "he'd", "her", "hers", "herself", "hes", "he's", "him", "himself", "his", "i", "Id", "I'd", "I'll", "Im", "I'm", "ive", "I've", "lets", "let's", "me", "mine", "my", "myself", "oneself", "our", "ours", "ourselves", "she", "she'd", "she'll", "shes", "she's", "thee", "their", "them", "themselves", "they", "theyd", "they'd", "theyll", "they'll", "theyve", "they've", "thine", "thou", "thoust", "thy", "us", "we", "we'd", "we'll", "we're", "weve", "we've", "ya", "yall", "y'all", "ye", "you", "youd", "you'd", "youll", "you'll", "your ", "youre", "you're", "yours", "youve", "you've"],

    'I' : ["i", "Id", "I'd", "I'll", "Im", "I'm", "ive", "I've", "me", "mine", "my", "myself"],

    'We' : ["lets", "let's", "our", "ours", "ourselves", "us", "we", "we'd", "we'll", "we're", "weve", "we've"],

    'You' : ["thee", "thine", "thou", "thoust", "thy", "ya", "yall", "y'all", "ye", "you", "youd", "you'd", "youll", "you'll", "your ", "youre", "you're", "yours", "youve", "you've"],

    'SheHe' : ["he", "hed", "he'd", "her", "hers", "herself", "hes", "he's", "him", "himself", "his", "oneself", "she", "she'd", "she'll", "shes", "she's"],

    'They' : ["their", "them", "themselves", "they", "theyd", "they'd", "theyll", "they'll", "theyve", "they've"],

    'Ipron' : ["anybod", "anyone", "anything", "everybod", "everyone", "everything", "it", "itd", "it'd", "itll", "it'll", "its", "it's", "itself", "nobod", "other", "others", "somebod", "someone", "something", "somewhere", "stuff", "that", "thatd", "that'd", "thatll", "that'll", "thats", "that's", "these", "thing", "this", "those", "what", "whatever", "whats", "what's", "which", "whichever", "who", "whod", "who'd", "wholl", "who'll", "whom", "whose"],

    'Article' : ["a", "alot", "an", "the"],

    'Verbs' : ["accepted", "admit", "admits", "admitted", "affected", "aint", "ain't", "am", "appear", "appeared", "appears", "are", "arent", "aren't", "arrive", "arrived", "arrives", "ask", "asked", "asks", "ate", "be", "became", "become", "becomes", "becoming", "been", "began", "begin", "begins", "being", "believe", "believed", "believes", "bought", "bring", "brings", "brought", "called", "came", "can", "cannot", "cant", "can't", "care", "cared", "cares", "carried", "carries", "carry", "caught", "changed", "come", "comes", "could", "couldnt", "couldn't", "couldve", "could've", "cried", "depended", "depends", "describe", "described", "did", "didnt", "didn't", "died", "dies", "dislike", "disliked", "dislikes", "do", "does", "doesnt", "doesn't", "doing", "done", "dont", "don't", "drank", "driven", "drove", "eaten", "emailed", "ended", "entered", "explain", "explained", "explains", "fed", "feel", "feels", "felt", "fled", "flew", "follow", "followed", "follows", "forgot", "fought", "found", "fuck", "fucked", "fucks", "gave", "get", "gets", "give", "given", "gives", "go", "goes", "gone", "gonna", "got", "gotta", "gotten", "guess", "guessed", "guesses", "had", "hadnt", "hadn't", "happen", "happened", "happens", "has", "hasnt", "hasn't", "hate", "hated", "hates", "have", "havent", "haven't", "having", "hear", "heard", "hears", "hed", "he'd", "held", "he'll", "helped", "helps", "heres", "here's", "hes", "he's", "hit", "hope", "hoped", "hopes", "hows", "how's", "Id", "I'd", "I'll", "Im", "i'm", "is", "isnt", "isn't", "itd", "it'd", "itll", "it'll", "it's", "ive", "I've", "keep", "keeps", "kept", "knew", "know", "knows", "let ", "lets", "let's", "lied", "liked", "listen", "listened", "listens", "lived", "look", "looked", "looks", "lost", "love", "loved", "loves", "made", "make", "makes", "may", "mean", "means", "meant", "met", "might", "mightve", "might've", "miss", "missed", "misses", "moved", "must", "mustnt", "must'nt", "mustn't", "mustve", "must've", "need", "needed", "needs", "ought", "oughta", "oughtnt", "ought'nt", "oughtn't", "oughtve", "ought've", "owe", "owed", "owes", "paid", "played", "ran", "said", "sat", "saw", "see", "seem", "seemed", "seems", "seen", "sees", "sensed", "sent", "shall", "shant", "shan't", "shared", "she'd", "she'll", "shes", "she's", "should", "shouldnt", "should'nt", "shouldn't", "shouldve", "should've", "showed", "slept", "sold", "spent", "spoke", "start", "started", "starts", "stayed", "stood", "stopped", "stuck", "studied", "suck", "sucked", "sucks", "suffered", "support", "supported", "supports", "suppose", "supposed", "supposes", "take", "taken", "takes", "taking", "talked", "taught", "tend", "tended", "tends", "thank", "thanked", "thanks", "thatd", "that'd", "thatll", "that'll", "thats", "that's", "theres", "there's", "theyd", "they'd", "theyll", "they'll", "theyre", "they're", "theyve", "they've", "think", "thinks", "thought", "threw", "told", "took", "tried", "tries", "try", "turn", "turned", "turns", "understand", "understands", "understood", "use", "used", "uses", "using", "viewed", "wait", "waited", "waits", "walked", "want", "wanted", "wants", "was", "wasnt", "wasn't", "we'd", "we'll", "went", "were", "we're", "weren't", "weve", "we've", "whats", "what's", "wheres", "where's", "whod", "who'd", "wholl", "who'll", "whos", "who's", "will", "wish", "wished", "wishes", "woke", "woken", "won", "wonder", "wondered", "wont", "won't", "wore", "worked", "worn", "would", "wouldnt", "wouldn't", "wouldve", "would've", "written", "wrote", "youd", "you'd", "youll", "you'll", "youre", "you're", "youve", "you've"],

    'AuxVb' : ["aint", "ain't", "am", "are", "arent", "aren't", "be", "became", "become", "becomes", "becoming", "been", "being", "can", "cannot", "cant", "can't", "could", "couldnt", "couldn't", "couldve", "could've", "did", "didnt", "didn't", "do", "does", "doesnt", "doesn't", "doing", "done", "dont", "don't", "had", "hadnt", "hadn't", "has", "hasnt", "hasn't", "have", "havent", "haven't", "having", "hed", "he'd", "heres", "here's", "hes", "he's", "Id", "I'd", "I'll", "Im", "I'm", "is", "isnt", "isn't", "itd", "it'd", "itll", "it'll", "it's", "ive", "I've", "let ", "may", "might", "mightve", "might've", "must", "mustnt", "must'nt", "mustn't", "mustve", "must've", "ought", "oughta", "oughtnt", "ought'nt", "oughtn't", "oughtve", "ought've", "shall", "shant", "shan't", "she'd", "she'll", "shes", "she's", "should", "shouldnt", "should'nt", "shouldn't", "shouldve", "should've", "thatd", "that'd", "thatll", "that'll", "thats", "that's", "theres", "there's", "theyd", "they'd", "theyll", "they'll", "theyre", "they're", "theyve", "they've", "was", "wasnt", "wasn't", "we'd", "we'll", "were", "weren't", "weve", "we've", "whats", "what's", "wheres", "where's", "whod", "who'd", "wholl", "who'll", "will", "wont", "won't", "would", "wouldnt", "wouldn't", "wouldve", "would've", "youd", "you'd", "youll", "you'll", "youre", "you're", "youve", "you've"],


    'Past' : ["accepted", "admitted", "affected", "appeared", "arrived", "asked", "ate", "became", "been", "began", "believed", "bought", "brought", "called", "came", "cared", "carried", "caught", "changed", "couldve", "could've", "cried", "depended", "described", "did", "didnt", "didn't", "died", "disliked", "done", "drank", "driven", "drove", "eaten", "emailed", "ended", "entered", "explained", "fed", "felt", "fled", "flew", "followed", "forgot", "fought", "found", "fucked", "gave", "given", "gone", "got", "gotten", "guessed", "had", "hadnt", "hadn't", "happened", "hated", "heard", "held", "helped", "hoped", "kept", "knew", "lied", "liked", "listened", "lived", "looked", "lost", "loved", "made", "meant", "met", "missed", "moved", "mustve", "must've", "needed", "owed", "paid", "played", "ran", "said", "sat", "saw", "seemed", "seen", "sensed", "sent", "shared", "shouldve", "should've", "showed", "slept", "sold", "spent", "spoke", "started", "stayed", "stood", "stopped", "stuck", "studied", "sucked", "suffered", "supported", "supposed", "taken", "talked", "taught", "tended", "thanked", "thought", "threw", "told", "took", "tried", "turned", "understood", "used", "viewed", "waited", "walked", "wanted", "was", "wasnt", "wasn't", "went", "were", "weren't", "weve", "we've", "wished", "woke", "woken", "won", "wondered", "wore", "worked", "worn", "wouldve", "would've", "written", "wrote"],


    'Present' : ["admit", "admits", "aint", "ain't", "am", "appear", "appears", "are", "arent", "aren't", "arrive", "arrives", "ask", "asks", "become", "becomes", "begin", "begins", "believe", "believes", "bring", "brings", "can", "cannot", "cant", "can't", "care", "cares", "carries", "carry", "come", "comes", "depends", "describe", "describes", "dies", "dislike", "dislikes", "do", "does", "doesnt", "doesn't", "dont", "don't", "explain", "explains", "feel", "feels", "follow", "follows", "fuck", "fucks", "get", "gets", "give", "gives", "go", "goes", "guess", "guesses", "happen", "happens", "has", "hasnt", "hasn't", "hate", "hates", "have", "havent", "haven't", "hear", "hears", "helps", "heres", "here's", "hes", "he's", "hope", "hopes", "hows", "how's", "Im", "i'm", "is", "isnt", "isn't", "it's", "ive", "I've", "keep", "keeps", "know", "knows", "lets", "let's", "listen", "listens", "look", "looks", "love", "loves", "make", "makes", "mean", "means", "miss", "misses", "need", "needs", "owe", "owes", "see", "seem", "seems", "sees", "shes", "she's", "start", "starts", "suck", "sucks", "support", "supports", "suppose", "supposes", "take", "takes", "taking", "tend", "tends", "thank", "thanks", "thats", "that's", "theres", "there's", "theyre", "they're", "theyve", "they've", "think", "thinks", "tries", "try", "turn", "turns", "understand", "understands", "use", "uses", "using", "wait", "waits", "want", "wants", "we're", "whats", "what's", "wheres", "where's", "whos", "who's", "wish", "wishes", "wonder", "youre", "you're", "youve", "you've"],


    'Future' : ["couldve", "could've", "gonna", "gotta", "he'll", "I'll", "itll", "it'll", "may", "might", "must", "mustnt", "must'nt", "mustn't", "mustve", "must've", "ought", "oughta", "oughtnt", "ought'nt", "oughtn't", "oughtve", "ought've", "shall", "she'll", "should", "shouldnt", "should'nt", "shouldn't", "shouldve", "should've", "thatll", "that'll", "theyll", "they'll", "we'll", "wholl", "who'll", "will", "wont", "won't", "would", "wouldnt", "wouldn't", "wouldve", "would've", "youll", "you'll"],

    'Adverbs' : ["about", "absolutely", "actually", "again", "also", "anyway", "anywhere", "apparently", "around", "back", "basically", "beyond", "clearly", "completely", "constantly", "definitely", "especially", "even", "eventually", "ever", "frequently", "generally", "here", "heres", "here's", "hopefully", "how", "however", "immediately", "instead", "just", "lately", "maybe", "mostly", "nearly", "now", "often", "only", "perhaps", "primarily", "probably", "push", "quick", "rarely", "rather", "really", "seriously", "simply", "so", "somehow", "soon", "sooo", "still", "such", "there", "theres", "there's", "tho", "though", "too", "totally", "truly", "usually", "very", "well", "when ", "whenever", "where", "yet"],
    'Prep' : ["about", "above", "across", "after", "against", "ahead", "along", "among", "around", "as", "at", "atop", "away", "before", "behind", "below", "beneath", "beside", "besides", "between", "beyond", "by", "despite", "down", "during", "except", "for", "from", "in", "inside", "insides", "into", "near", "of", "off", "on", "onto", "out", "outside", "over", "plus", "since", "than", "through", "thru", "til ", "till", "to", "toward", "under", "underneath", "unless", "until", "unto", "up", "upon", "wanna", "with", "within", "without"],

    'Conj' : ["also", "although", "and", "as", "altho ", "because", "but", "cuz", "how", "however", "if", "nor", "or", "otherwise", "plus", "so", "then", "tho", "though", "til", "till", "unless", "until", "when", "whenever", "whereas", "whether", "while"],

    'Negate' : ["aint", "ain't", "arent", "aren't", "cannot", "cant", "can't", "couldnt", "couldn't", "didnt", "didn't", "doesnt", "doesn't", "dont", "don't", "hadnt", "hadn't", "hasnt", "hasn't", "havent", "haven't", "isnt", "isn't", "mustnt", "must'nt", "mustn't", "neednt", "need'nt", "needn't", "negat", "neither", "never", "no", "nobod", "none", "nope", "nor", "not", "nothing", "nowhere", "oughtnt", "ought'nt", "oughtn't", "shant", "shan't", "shouldnt", "should'nt", "shouldn't", "uhuh", "wasnt", "wasn't", "weren't", "without", "wont", "won't", "wouldnt", "wouldn't"],

    'Quant' : ["all", "allot", "alot", "amount", "another", "any", "anymore", "besides", "best ", "bit ", "bits", "both", "bunch", "cetera", "couple", "difference", "doubl", "each", "either", "else", "enough", "entire", "equal", "etc", "every", "example", "extent", "extra", "extremely", "fairly", "few", "form", "full", "greater", "greatest", "highly", "increas", "item", "lack", "least", "less", "loads", "lot", "lotof", "lots", "lotsa", "lotta", "main", "major", "majority", "many", "more", "most", "much", "mucho", "neither", "none", "ones", "own", "page", "part", "partly", "percent", "piec", "plenty", "portion", "remaining", "rest", "same", "section", "segment", "selection", "series", "several", "significant", "simple", "singl", "some", "somewhat", "term", "ton", "tons", "total", "tripl", "unique", "various", "version", "whole", "worst"],

    'Numbers' : ["billion", "dozen", "eight", "eleven", "fift", "first", "firstly", "firsts", "five", "four", "half", "hundred", "infinit", "million", "nine", "once", "one", "quarter", "second", "seven", "six", "ten", "tenth", "third", "thirt", "thousand", "three", "trillion", "twel", "twent", "twice", "two", "zero", "zillion"],

    'Swear' : ["arse", "arsehole", "arses", "ass", "asses", "asshole", "bastard", "bitch", "bloody", "boob", "butt", "butts", "cock", "cocks", "crap", "crappy", "cunt", "damn", "dang", "darn", "dick", "dicks", "dumb", "dyke", "fuck", "fucked", "fucker", "fuckin", "fucks", "goddam", "heck", "hell", "homo", "jeez", "mofo", "motherf", "nigger", "piss", "prick", "pussy", "queer", "screw", "shit", "sob", "sonofa", "suck", "sucked", "sucks", "tit", "tits", "titties", "titty", "wanker"],

    'Social' : ["acquainta", "admit", "admits", "admitted", "admitting", "adult", "adults", "advice", "advis", "affair", "amigo", "anybod", "anyone", "apolog", "argu", "armies", "army", "ask", "asked", "asking", "asks", "assembl", "aunt", "babe", "babies", "baby", "bambino", "band", "bands", "bf", "blam", "boy", "boyf", "boy's", "boys", "bro", "bros", "brother", "bud", "buddies", "buddy", "bye", "call", "called", "caller", "calling", "calls", "captain", "celebrat", "cell", "cellphon", "cells", "cellular", "chat", "chick", "chick'", "chicks", "child", "children", "child's", "citizen", "citizen'", "citizens", "colleague", "comment", "commun", "companion", "companions", "companionship", "compassion", "complain", "comrad", "confess", "confide", "confided", "confides", "confiding", "congregat", "consult", "contact", "contradic", "convers", "counc", "couns", "cousin", "coworker", "crowd", "cultur", "dad", "dating", "daughter", "deal", "describe", "described", "describes", "describing", "disclo", "discuss", "divorc", "email", "email'", "emailed", "emailer", "emailing", "emails", "encourag", "enemie", "enemy", "everybod", "everyone", "everything", "ex", "exbf", "exboyfriend", "excus", "exes", "exgf", "exgirl", "exhubby", "exhusband", "explain", "explained", "explaining", "explains", "express", "exwife", "exwive", "families", "family", "father", "fellow", "female", "feud", "fiance", "fight", "flatter", "folks", "forgave", "forgiv", "fought", "friend", "game", "gather", "gave", "gentlem", "gf", "girl", "girlfriend", "girl's", "girls", "give", "giver", "gives", "giving", "gossip", "grandchil", "granddad", "granddau", "grandf", "grandkid", "grandm", "grandpa", "grandson", "granny", "group", "grownup", "grudge", "guest", "guy", "he", "hear", "heard", "hearing", "hears", "hed", "he'd", "he'll", "hello", "help", "helper", "helpful", "helping", "helps", "her", "hers", "herself", "hes", "he's", "hey", "hi", "him", "himself", "his", "honey", "hubby", "human", "husband", "individual", "infant", "infant's", "infants", "inform", "informs", "insult", "interact", "interrup", "interview", "involv", "kid", "kid'", "kidding", "kids", "kin", "ladies", "lady", "lady's", "language", "lets", "let's", "letter", "listen", "listened", "listener", "listening", "listens", "love ", "loved", "lover", "loves", "loving", "ma", "ma'am", "mail", "mailed", "mailer", "mailing", "mails", "male", "males", "male's", "mam", "man", "man's", "marriag", "marrie", "ma's", "mate", "mates", "mate's", "mating", "meet", "meeting", "meets", "members", "men", "men'", "mention", "messag", "met", "mob", "mobb", "mobs", "mom", "momma", "mommy", "moms", "mom's", "mother", "motherly", "mothers", "mr ", "mrs", "mum", "mummy", "mums", "mum's", "name", "negotiat", "neighbor", "neighbour", "nephew", "newborn", "niece", "offer", "organiz", "our", "ours", "ourselves", "outsider", "overhear", "owner", "pa", "pal", "pals", "pappy", "parent", "participant", "participat", "partie", "partner", "party", "pa's", "people", "person", "personal", "persons", "person's", "persua", "phone", "phoning", "prais", "private", "provide", "public", "question", "reassur", "receiv", "refus", "relationship", "relatives", "replie", "reply", "request", "respond", "role", "roomate", "roomed", "roomie", "rooming", "roommate", "rumor", "rumour", "said", "say", "secret", "secretive", "secrets", "self ", "send", "sent", "share", "shared", "shares", "sharing", "she", "she'd", "she'll", "shes", "she's", "sir", "sis", "sister", "social", "societ", "somebod", "someone", "son", "sons", "son's", "soulmate", "speak", "speaking", "speaks", "spoke", "spous", "stepchild", "stepfat", "stepkid", "stepmot", "stories", "story", "suggest", "sweetheart", "sweetie", "talk", "talkative", "talked", "talker", "talking", "talks", "team", "teas", "telephon", "tell", "telling", "tells", "thee", "their", "them", "themselves", "they", "theyd", "they'd", "theyll", "they'll", "theyre", "they're", "theyve", "they've", "thine", "thou", "thoust", "thy", "told", "transact", "uncle", "uncles", "uncle's", "ur", "us", "visit", "we", "wed", "we'd", "wedding", "weds", "welcom", "we'll", "we're", "weve", "we've", "who", "whod", "who'd", "wholl", "who'll", "whom", "whos", "who's", "whose", "wife", "willing", "wive", "woman", "womanhood", "womanly", "woman's", "women", "write", "writing", "wrote", "ya", "yall", "y'all", "ye", "you", "youd", "you'd", "youll", "you'll", "your ", "youre", "you're", "yours", "youve", "you've"],


    'Family' : ["aunt", "bro", "bros", "brother", "cousin", "dad", "daughter", "ex", "exes", "exhubby", "exhusband", "exwife", "exwive", "families", "family", "father", "folks", "grandchil", "granddad", "granddau", "grandf", "grandkid", "grandm", "grandpa", "grandson", "granny", "hubby", "husband", "kin", "ma", "marrie", "ma's", "mom", "momma", "mommy", "moms", "mom's", "mother", "mothers", "mum", "mummy", "mums", "mum's", "nephew", "niece", "pa", "pappy", "parent", "pa's", "relatives", "sis", "sister", "son", "sons", "son's", "spous", "stepchild", "stepfat", "stepkid", "stepmot", "uncle", "uncles", "uncle's", "wife", "wive"],

    'Friends' : ["acquainta", "amigo", "bf", "boyf", "bud", "buddies", "buddy", "colleague", "companion", "companions", "comrad", "exbf", "exboyfriend", "exgf", "exgirl", "fellow", "fiance", "friend", "gf", "girlfriend", "guest", "honey", "lover", "mate", "mates", "mate's", "neighbor", "neighbour", "pal", "pals", "partner", "roomate", "roomie", "roommate", "soulmate", "sweetheart", "sweetie"],

    'Humans' : ["adult", "adults", "babe", "babies", "baby", "bambino", "boy", "boy's", "boys", "chick", "chick'", "chicks", "child", "children", "child's", "citizen", "citizen'", "citizens", "female", "gentlem", "girl", "girl's", "girls", "grownup", "guy", "human", "individual", "infant", "infant's", "infants", "kid", "kid'", "kids", "ladies", "lady", "lady's", "ma'am", "male", "males", "male's", "mam", "man", "man's", "members", "men", "men'", "mr ", "mrs", "newborn", "participant", "partner", "people", "person", "persons", "person's", "self ", "sir", "societ", "woman", "woman's", "women"],

    'Affect' : ["abandon", "abuse", "abusi", "accept", "accepta", "accepted", "accepting", "accepts", "ache", "aching", "active", "admir", "ador", "advantag", "adventur", "advers", "affection", "afraid", "aggravat", "aggress", "agitat", "agoniz", "agony", "agree", "agreeab", "agreed", "agreeing", "agreement", "agrees", "alarm", "alone", "alright", "amaz", "amor", "amus", "anger", "angr", "anguish", "annoy", "antagoni", "anxi", "aok", "apath", "appall", "appreciat", "apprehens", "argh", "argu", "arrogan", "asham", "assault", "asshole", "assur", "attachment", "attack", "attract", "aversi", "avoid", "award", "awesome", "awful", "awkward", "bad", "bashful", "bastard", "battl", "beaten", "beaut", "beloved", "benefic", "benefit", "benefits", "benefitt", "benevolen", "benign", "best", "better", "bitch", "bitter", "blam", "bless", "bold", "bonus", "bore", "boring", "bother", "brave", "bright", "brillian", "broke", "brutal", "burden", "calm", "care", "cared", "carefree", "careful", "careless", "cares", "caring", "casual", "casually", "certain", "challeng", "champ", "charit", "charm", "cheat", "cheer", "cherish", "chuckl", "clever", "comed", "comfort", "commitment", "compassion", "complain", "compliment", "concerned", "confidence", "confident", "confidently", "confront", "confus", "considerate", "contempt", "contented", "contentment", "contradic", "convinc", "cool", "courag", "crap", "crappy", "craz", "create", "creati", "credit", "cried", "cries", "critical", "critici", "crude", "cruel", "crushed", "cry", "crying", "cunt", "cut", "cute", "cutie", "cynic", "damag", "damn", "danger", "daring", "darlin", "daze", "dear", "decay", "defeat", "defect", "defenc", "defens", "definite", "definitely", "degrad", "delectabl", "delicate", "delicious", "deligh", "depress", "depriv", "despair", "desperat", "despis", "destroy", "destruct", "determina", "determined", "devastat", "devil", "devot", "difficult", "digni", "disadvantage", "disagree", "disappoint", "disaster", "discomfort", "discourag", "disgust", "dishearten", "disillusion", "dislike", "disliked", "dislikes", "disliking", "dismay", "dissatisf", "distract", "distraught", "distress", "distrust", "disturb", "divin", "domina", "doom", "dork", "doubt", "dread", "dull", "dumb", "dump", "dwell", "dynam", "eager", "ease", "easie", "easily", "easiness", "easing", "easy", "ecsta", "efficien", "egotis", "elegan", "embarrass", "emotion", "emotion", "emotional", "empt", "encourag", "enemie", "enemy", "energ", "engag", "enjoy", "enrag", "entertain", "enthus", "envie", "envious", "envy", "evil", "excel", "excit", "excruciat", "exhaust", "fab", "fabulous", "fail", "faith", "fake", "fantastic", "fatal", "fatigu", "fault", "favor", "favour", "fear", "feared", "fearful", "fearing", "fearless", "fears", "feroc", "festiv", "feud", "fiery", "fiesta", "fight", "fine", "fired", "flatter", "flawless", "flexib", "flirt", "flunk", "foe", "fond", "fondly", "fondness", "fool", "forbid", "forgave", "forgiv", "fought", "frantic", "freak", "free", "freeb", "freed", "freeing", "freely", "freeness", "freer", "frees", "friend", "fright", "frustrat", "fuck", "fucked", "fucker", "fuckin", "fucks", "fume", "fuming", "fun", "funn", "furious", "fury", "geek", "genero", "gentle", "gentler", "gentlest", "gently", "giggl", "giver", "giving", "glad", "gladly", "glamor", "glamour", "gloom", "glori", "glory", "goddam", "good", "goodness", "gorgeous", "gossip", "grace", "graced", "graceful", "graces", "graci", "grand", "grande", "gratef", "grati", "grave", "great", "greed", "grief", "griev", "grim", "grin", "grinn", "grins", "gross", "grouch", "grr", "guilt", "ha", "haha", "handsom", "happi", "happy", "harass", "harm", "harmed", "harmful", "harming", "harmless", "harmon", "harms", "hate", "hated", "hateful", "hater", "hates", "hating", "hatred", "hazy", "heartbreak", "heartbroke", "heartfelt", "heartless", "heartwarm", "heaven", "heh", "hell", "hellish", "helper", "helpful", "helping", "helpless", "helps", "hero", "hesita", "hilarious", "hoho", "homesick", "honest", "honor", "honour", "hope", "hoped", "hopeful", "hopefully", "hopefulness", "hopeless", "hopes", "hoping", "horr", "hostil", "hug", "hugg", "hugs", "humiliat", "humor", "humour", "hurra", "hurt", "ideal", "idiot", "ignor", "immoral", "impatien", "impersonal", "impolite", "importan", "impress", "improve", "improving", "inadequa", "incentive", "indecis", "ineffect", "inferior ", "inhib", "innocen", "insecur", "insincer", "inspir", "insult", "intell", "interest", "interrup", "intimidat", "invigor", "irrational", "irrita", "isolat", "jaded", "jealous", "jerk", "jerked", "jerks", "joke", "joking", "joll", "joy", "keen", "kidding", "kill", "kind", "kindly", "kindn", "kiss", "laidback", "lame", "laugh", "lazie", "lazy", "liabilit", "liar", "libert", "lied", "lies", "like", "likeab", "liked", "likes", "liking", "livel", "LMAO", "LOL", "lone", "longing", "lose", "loser", "loses", "losing", "loss", "lost", "lous", "love", "loved", "lovely", "lover", "loves", "loving", "low", "loyal", "luck", "lucked", "lucki", "luckless", "lucks", "lucky", "ludicrous", "lying", "mad", "maddening", "madder", "maddest", "madly", "magnific", "maniac", "masochis", "melanchol", "merit", "merr", "mess", "messy", "miser", "miss", "missed", "misses", "missing", "mistak", "mock", "mocked", "mocker", "mocking", "mocks", "molest", "mooch", "mood", "moodi", "moods", "moody", "moron", "mourn", "murder", "nag", "nast", "neat", "needy", "neglect", "nerd", "nervous", "neurotic", "nice", "numb", "nurtur", "obnoxious", "obsess", "offence", "offend", "offens", "ok", "okay", "okays", "oks", "openminded", "openness", "opportun", "optimal", "optimi", "original", "outgoing", "outrag", "overwhelm", "pain", "pained", "painf", "paining", "painl", "pains", "palatabl", "panic", "paradise", "paranoi", "partie", "party", "passion", "pathetic", "peace", "peculiar", "perfect", "personal", "perver", "pessimis", "petrif", "pettie", "petty", "phobi", "piss", "piti", "pity ", "play", "played", "playful", "playing", "plays", "pleasant", "please", "pleasing", "pleasur", "poison", "popular", "positiv", "prais", "precious", "prejudic", "pressur", "prettie", "pretty", "prick", "pride", "privileg", "prize", "problem", "profit", "promis", "protest", "protested", "protesting", "proud", "puk", "punish", "radian", "rage", "raging", "rancid", "rape", "raping", "rapist", "readiness", "ready", "reassur", "rebel", "reek", "regret", "reject", "relax", "relief", "reliev", "reluctan", "remorse", "repress", "resent", "resign", "resolv", "respect ", "restless", "revenge", "revigor", "reward", "rich", "ridicul", "rigid", "risk", "ROFL", "romanc", "romantic", "rotten", "rude", "ruin", "sad", "sadde", "sadly", "sadness", "safe", "sarcas", "satisf", "savage", "save", "scare", "scaring", "scary", "sceptic", "scream", "screw", "secur", "selfish", "sentimental", "serious", "seriously", "seriousness", "severe", "shake", "shaki", "shaky", "shame", "share", "shared", "shares", "sharing", "shit", "shock", "shook", "shy", "sicken", "sigh", "sighed", "sighing", "sighs", "silli", "silly", "sin", "sincer", "sinister", "sins", "skeptic", "slut", "smart", "smil", "smother", "smug", "snob", "sob", "sobbed", "sobbing", "sobs", "sociab", "solemn", "sorrow", "sorry", "soulmate", "special", "spite", "splend", "stammer", "stank", "startl", "steal", "stench", "stink", "strain", "strange", "strength", "stress", "strong", "struggl", "stubborn", "stunk", "stunned", "stuns", "stupid", "stutter", "submissive", "succeed", "success", "suck", "sucked", "sucker", "sucks", "sucky", "suffer", "suffered", "sufferer", "suffering", "suffers", "sunnier", "sunniest", "sunny", "sunshin", "super", "superior", "support", "supported", "supporter", "supporting", "supportive", "supports", "suprem", "sure", "surpris", "suspicio", "sweet", "sweetheart", "sweetie", "sweetly", "sweetness", "sweets", "talent", "tantrum", "tears", "teas", "tehe", "temper", "tempers", "tender", "tense", "tensing", "tension", "terribl", "terrific", "terrified", "terrifies", "terrify ", "terrifying", "terror", "thank", "thanked", "thankf", "thanks", "thief", "thieve", "thoughtful", "threat", "thrill", "ticked", "timid", "toleran", "tortur", "tough", "traged", "tragic ", "tranquil", "trauma", "treasur", "treat", "trembl", "trick", "trite", "triumph", "trivi", "troubl", "true ", "trueness", "truer", "truest", "truly", "trust", "truth", "turmoil", "ugh", "ugl", "unattractive", "uncertain", "uncomfortabl", "uncontrol", "uneas", "unfortunate", "unfriendly", "ungrateful", "unhapp", "unimportant", "unimpress", "unkind", "unlov", "unpleasant", "unprotected", "unsavo", "unsuccessful", "unsure", "unwelcom", "upset", "uptight", "useful", "useless ", "vain", "valuabl", "value", "valued", "values", "valuing", "vanity", "vicious", "victim", "vigor", "vigour", "vile", "villain", "violat", "violent", "virtue", "virtuo", "vital", "vulnerab", "vulture", "war", "warfare", "warm", "warred", "warring", "wars", "weak", "wealth", "weapon", "weep", "weird", "welcom", "well", "wept", "whine", "whining", "whore", "wicked", "willing", "wimp", "win", "winn", "wins", "wisdom", "wise", "witch", "woe", "won", "wonderf", "worr", "worse", "worship", "worst", "worthless ", "worthwhile", "wow", "wrong", "yay", "yays", "yearn"],


    'Posemo' : ["accept", "accepta", "accepted", "accepting", "accepts", "active", "admir", "ador", "advantag", "adventur", "affection", "agree", "agreeab", "agreed", "agreeing", "agreement", "agrees", "alright", "amaz", "amor", "amus", "aok", "appreciat", "assur", "attachment", "attract", "award", "awesome", "beaut", "beloved", "benefic", "benefit", "benefits", "benefitt", "benevolen", "benign", "best", "better", "bless", "bold", "bonus", "brave", "bright", "brillian", "calm", "care", "cared", "carefree", "careful", "cares", "caring", "casual", "casually", "certain", "challeng", "champ", "charit", "charm", "cheer", "cherish", "chuckl", "clever", "comed", "comfort", "commitment", "compassion", "compliment", "confidence", "confident", "confidently", "considerate", "contented", "contentment", "convinc", "cool", "courag", "create", "creati", "credit", "cute", "cutie", "daring", "darlin", "dear", "definite", "definitely", "delectabl", "delicate", "delicious", "deligh", "determina", "determined", "devot", "digni", "divin", "dynam", "eager", "ease", "easie", "easily", "easiness", "easing", "easy", "ecsta", "efficien", "elegan", "encourag", "energ", "engag", "enjoy", "entertain", "enthus", "excel", "excit", "fab", "fabulous", "faith", "fantastic", "favor", "favour", "fearless", "festiv", "fiesta", "fine", "flatter", "flawless", "flexib", "flirt", "fond", "fondly", "fondness", "forgave", "forgiv", "free", "free", "freeb", "freed", "freeing", "freely", "freeness", "freer", "frees", "friend", "fun", "funn", "genero", "gentle", "gentler", "gentlest", "gently", "giggl", "giver", "giving", "glad", "gladly", "glamor", "glamour", "glori", "glory", "good", "goodness", "gorgeous", "grace", "graced", "graceful", "graces", "graci", "grand", "grande", "gratef", "grati", "great", "grin", "grinn", "grins", "ha", "haha", "handsom", "happi", "happy", "harmless", "harmon", "heartfelt", "heartwarm", "heaven", "heh", "helper", "helpful", "helping", "helps", "hero", "hilarious", "hoho", "honest", "honor", "honour", "hope", "hoped", "hopeful", "hopefully", "hopefulness", "hopes", "hoping", "hug ", "hugg", "hugs", "humor", "humour", "hurra", "ideal", "importan", "impress", "improve", "improving", "incentive", "innocen", "inspir", "intell", "interest", "invigor", "joke", "joking", "joll", "joy", "keen", "kidding", "kind", "kindly", "kindn", "kiss", "laidback", "laugh", "libert", "like", "likeab", "liked", "likes", "liking", "livel", "LMAO", "LOL", "love", "loved", "lovely", "lover", "loves", "loving", "loyal", "luck", "lucked", "lucki", "lucks", "lucky", "madly", "magnific", "merit", "merr", "neat", "nice", "nurtur", "ok", "okay", "okays", "oks", "openminded", "openness", "opportun", "optimal", "optimi", "original", "outgoing", "painl", "palatabl", "paradise", "partie", "party", "passion", "peace", "perfect", "play", "played", "playful", "playing", "plays", "pleasant", "please", "pleasing", "pleasur", "popular", "positiv", "prais", "precious", "prettie", "pretty", "pride", "privileg", "prize", "profit", "promis", "proud", "radian", "readiness", "ready", "reassur", "relax", "relief", "reliev", "resolv", "respect ", "revigor", "reward", "rich", "ROFL", "romanc", "romantic", "safe", "satisf", "save", "scrumptious", "secur", "sentimental", "share", "shared", "shares", "sharing", "silli", "silly", "sincer", "smart", "smil", "sociab", "soulmate", "special", "splend", "strength", "strong", "succeed", "success", "sunnier", "sunniest", "sunny", "sunshin", "super", "superior", "support", "supported", "supporter", "supporting", "supportive", "supports", "suprem", "sure", "surpris", "sweet", "sweetheart", "sweetie", "sweetly", "sweetness", "sweets", "talent", "tehe", "tender", "terrific", "thank", "thanked", "thankf", "thanks", "thoughtful", "thrill", "toleran", "tranquil", "treasur", "treat", "triumph", "true ", "trueness", "truer", "truest", "truly", "trust", "truth", "useful", "valuabl", "value", "valued", "values", "valuing", "vigor", "vigour", "virtue", "virtuo", "vital", "warm", "wealth", "welcom", "well", "win", "winn", "wins", "wisdom", "wise", "won", "wonderf", "worship", "worthwhile", "wow", "yay", "yays"],


    'Negemo' : ["abandon", "abuse", "abusi", "ache", "aching", "advers", "afraid", "aggravat", "aggress", "agitat", "agoniz", "agony", "alarm", "alone", "anger", "angr", "anguish", "annoy", "antagoni", "anxi", "apath", "appall", "apprehens", "argh", "argu", "arrogan", "asham", "assault", "asshole", "attack", "aversi", "avoid", "awful", "awkward", "bad", "bashful", "bastard", "battl", "beaten", "bitch", "bitter", "blam", "bore", "boring", "bother", "broke", "brutal", "burden", "careless", "cheat", "complain", "confront", "confus", "contempt", "contradic", "crap", "crappy", "craz", "cried", "cries", "critical", "critici", "crude", "cruel", "crushed", "cry", "crying", "cunt", "cut", "cynic", "damag", "damn", "danger", "daze", "decay", "defeat", "defect", "defenc", "defens", "degrad", "depress", "depriv", "despair", "desperat", "despis", "destroy", "destruct", "devastat", "devil", "difficult", "disadvantage", "disagree", "disappoint", "disaster", "discomfort", "discourag", "disgust", "dishearten", "disillusion", "dislike", "disliked", "dislikes", "disliking", "dismay", "dissatisf", "distract", "distraught", "distress", "distrust", "disturb", "domina", "doom", "dork", "doubt", "dread", "dull", "dumb", "dump", "dwell", "egotis", "embarrass", "emotional", "empt", "enemie", "enemy", "enrag", "envie", "envious", "envy", "evil", "excruciat", "exhaust", "fail", "fake", "fatal", "fatigu", "fault", "fear", "feared", "fearful", "fearing", "fears", "feroc", "feud", "fiery", "fight", "fired", "flunk", "foe", "fool", "forbid", "fought", "frantic", "freak", "fright", "frustrat", "fuck", "fucked", "fucker", "fuckin", "fucks", "fume", "fuming", "furious", "fury", "geek", "gloom", "goddam", "gossip", "grave", "greed", "grief", "griev", "grim", "gross", "grouch", "grr", "guilt", "harass", "harm", "harmed", "harmful", "harming", "harms", "hate", "hated", "hateful", "hater", "hates", "hating", "hatred", "heartbreak", "heartbroke", "heartless", "hell", "hellish", "helpless", "hesita", "homesick", "hopeless", "horr", "hostil", "humiliat", "hurt", "idiot", "ignor", "immoral", "impatien", "impersonal", "impolite", "inadequa", "indecis", "ineffect", "inferior ", "inhib", "insecur", "insincer", "insult", "interrup", "intimidat", "irrational", "irrita", "isolat", "jaded", "jealous", "jerk", "jerked", "jerks", "kill", "lame", "lazie", "lazy", "liabilit", "liar", "lied", "lies", "lone", "longing", "lose", "loser", "loses", "losing", "loss", "lost", "lous", "low", "luckless", "ludicrous", "lying", "mad", "maddening", "madder", "maddest", "maniac", "masochis", "melanchol", "mess", "messy", "miser", "miss", "missed", "misses", "missing", "mistak", "mock", "mocked", "mocker", "mocking", "mocks", "molest", "mooch", "moodi", "moody", "moron", "mourn", "murder", "nag", "nast", "needy", "neglect", "nerd", "nervous", "neurotic", "numb", "obnoxious", "obsess", "offence", "offend", "offens", "outrag", "overwhelm", "pain", "pained", "painf", "paining", "pains", "panic", "paranoi", "pathetic", "peculiar", "perver", "pessimis", "petrif", "pettie", "petty", "phobi", "piss", "piti", "pity ", "poison", "prejudic", "pressur", "prick", "problem", "protest", "protested", "protesting", "puk", "punish", "rage", "raging", "rancid", "rape", "raping", "rapist", "rebel", "reek", "regret", "reject", "reluctan", "remorse", "repress", "resent", "resign", "restless", "revenge", "ridicul", "rigid", "risk", "rotten", "rude", "ruin", "sad", "sadde", "sadly", "sadness", "sarcas", "savage", "scare", "scaring", "scary", "sceptic", "scream", "screw", "selfish", "serious", "seriously", "seriousness", "severe", "shake", "shaki", "shaky", "shame", "shit", "shock", "shook", "shy", "sicken", "sin", "sinister", "sins", "skeptic", "slut", "smother", "smug", "snob", "sob", "sobbed", "sobbing", "sobs", "solemn", "sorrow", "sorry", "spite", "stammer", "stank", "startl", "steal", "stench", "stink", "strain", "strange", "stress", "struggl", "stubborn", "stunk", "stunned", "stuns", "stupid", "stutter", "submissive", "suck", "sucked", "sucker", "sucks", "sucky", "suffer", "suffered", "sufferer", "suffering", "suffers", "suspicio", "tantrum", "tears", "teas", "temper", "tempers", "tense", "tensing", "tension", "terribl", "terrified", "terrifies", "terrify", "terrifying", "terror", "thief", "thieve", "threat", "ticked", "timid", "tortur", "tough", "traged", "tragic ", "trauma", "trembl", "trick", "trite", "trivi", "troubl", "turmoil", "ugh", "ugl", "unattractive", "uncertain", "uncomfortabl", "uncontrol", "uneas", "unfortunate", "unfriendly", "ungrateful", "unhapp", "unimportant", "unimpress", "unkind", "unlov", "unpleasant", "unprotected", "unsavo", "unsuccessful", "unsure", "unwelcom", "upset", "uptight", "useless ", "vain", "vanity", "vicious", "victim", "vile", "villain", "violat", "violent", "vulnerab", "vulture", "war", "warfare", "warred", "warring", "wars", "weak", "weapon", "weep", "weird", "wept", "whine", "whining", "whore", "wicked", "wimp", "witch", "woe", "worr", "worse", "worst", "worthless ", "wrong", "yearn"],


    'Anx' : ["afraid", "alarm", "anguish", "anxi", "apprehens", "asham", "aversi", "avoid", "awkward", "confus", "craz", "desperat", "discomfort", "distract", "distraught", "distress", "disturb", "doubt", "dread", "dwell", "embarrass", "emotional", "fear", "feared", "fearful", "fearing", "fears", "frantic", "fright", "guilt", "hesita", "horr", "humiliat", "impatien", "inadequa", "indecis", "inhib", "insecur", "irrational", "irrita", "miser", "nervous", "neurotic", "obsess", "overwhelm", "panic", "petrif", "phobi", "pressur", "reluctan", "repress", "restless", "rigid", "risk", "scare", "scaring", "scary", "shake", "shaki", "shaky", "shame", "shook", "shy", "sicken", "startl", "strain", "stress", "struggl", "stunned", "stuns", "suspicio", "tense", "tensing", "tension", "terrified", "terrifies", "terrify ", "terrifying", "terror", "timid", "trembl", "turmoil", "uncertain", "uncomfortabl", "uncontrol", "uneas", "unsure", "upset", "uptight", "vulnerab", "worr"],


    'Anger' : ["abuse", "abusi", "aggravat", "aggress", "agitat", "anger", "angr", "annoy", "antagoni", "argh", "argu", "arrogan", "assault", "asshole", "attack", "bastard", "battl", "beaten", "bitch", "bitter", "blam", "bother", "brutal", "cheat", "confront", "contempt", "contradic", "crap", "crappy", "critical", "critici", "crude", "cruel", "cunt", "cut", "cynic", "damn", "danger", "defenc", "defens", "despis", "destroy", "destruct", "disgust", "distrust", "domina", "dumb", "dump", "enemie", "enemy", "enrag", "envie", "envious", "envy", "evil", "feroc", "feud", "fiery", "fight", "foe", "fought", "frustrat", "fuck", "fucked", "fucker", "fuckin", "fucks", "fume", "fuming", "furious", "fury", "goddam", "greed", "grouch", "grr", "harass", "hate", "hated", "hateful", "hater", "hates", "hating", "hatred", "heartless", "hell", "hellish", "hostil", "humiliat", "idiot", "insult", "interrup", "intimidat", "jealous", "jerk", "jerked", "jerks", "kill", "liar", "lied", "lies", "lous", "ludicrous", "lying", "mad", "maddening", "madder", "maddest", "maniac", "mock", "mocked", "mocker", "mocking", "mocks", "molest", "moron", "murder", "nag", "nast", "obnoxious", "offence", "offend", "offens", "outrag", "paranoi", "pettie", "petty", "piss", "poison", "prejudic", "prick", "protest", "protested", "protesting", "punish", "rage", "raging", "rape", "raping", "rapist", "rebel", "resent", "revenge", "ridicul", "rude", "sarcas", "savage", "sceptic", "screw", "shit", "sinister", "skeptic", "smother", "snob", "spite", "stubborn", "stupid", "suck", "sucked", "sucker", "sucks", "sucky", "tantrum", "teas", "temper", "tempers", "terrify", "threat", "ticked", "tortur", "trick", "ugl", "vicious", "victim", "vile", "villain", "violat", "violent", "war", "warfare", "warred", "warring", "wars", "weapon", "wicked"],


    'Sad' : ["abandon", "ache", "aching", "agoniz", "agony", "alone", "broke", "cried", "cries", "crushed", "cry", "crying", "damag", "defeat", "depress", "depriv", "despair", "devastat", "disadvantage", "disappoint", "discourag", "dishearten", "disillusion", "dissatisf", "doom", "dull", "empt", "fail", "fatigu", "flunk", "gloom", "grave", "grief", "griev", "grim", "heartbreak", "heartbroke", "helpless", "homesick", "hopeless", "hurt", "inadequa", "inferior ", "isolat", "lame", "lone", "longing", "lose", "loser", "loses", "losing", "loss", "lost", "low", "melanchol", "miser", "miss", "missed", "misses", "missing", "mourn", "neglect", "overwhelm", "pathetic", "pessimis", "piti", "pity ", "regret", "reject", "remorse", "resign", "ruin", "sad", "sadde", "sadly", "sadness", "sob", "sobbed", "sobbing", "sobs", "solemn", "sorrow", "suffer", "suffered", "sufferer", "suffering", "suffers", "tears", "traged", "tragic ", "unhapp", "unimportant", "unsuccessful", "useless ", "weep", "wept", "whine", "whining", "woe", "worthless ", "yearn"],

    'CogMech' : ["abandon", "absolute", "absolutely", "abstain", "accept", "accepta", "accepted", "accepting", "accepts", "accura", "acknowledg", "activat", "add", "addit", "adjust", "admit", "admits", "admitted", "admitting", "affect", "affected", "affecting", "affects", "afterthought", "aggravat", "all", "allot", "allow", "almost", "along", "alot", "altogether", "always", "ambigu", "anal", "analy", "and", "answer", "any", "anybod", "anyhow", "anyone", "anything", "anytime", "anywhere", "apparent", "apparently", "appear", "appeared", "appearing", "appears", "appreciat", "approximat", "arbitrar", "around", "assum", "assur", "attent", "attribut", "avert", "avoid", "aware", "ban", "banned", "banning", "bans", "barely", "barrier", "based", "bases", "basis", "became", "because", "become", "becomes", "becoming", "belief", "believe", "believed", "believes", "believing", "besides", "bet", "bets", "betting", "binding", "blatant", "block", "blocked", "blocker", "blocking", "blocks", "blur", "borderline", "boss", "both", "bound", "brake", "bridle", "but", "came", "careful", "categor", "caus", "caut", "cease", "ceasing", "certain", "chance", "change", "changed", "changes", "changing", "choice", "choos", "clarif", "clear", "clearly", "close", "closure", "cohere", "come", "commit", "commitment", "commits", "committ", "compel", "complete", "completed", "completely", "completes", "complex", "compliance", "complica", "complie", "comply", "compreh", "compulsiv", "concentrat", "conclud", "conclus", "confess", "confidence", "confident", "confidently", "confin", "conflict", "confus", "conscious", "consequen", "conserv", "consider", "considered", "considering", "considers", "constrain", "constrict", "contain", "contemplat", "contingen", "contradic", "control", "correct", "correlat", "cos", "could", "couldnt", "couldn't", "couldve", "could've", "coz", "create", "creati", "curb", "curio", "curtail", "cuz", "decid", "decis", "deduc", "defenc", "defens", "define", "defined", "defines", "defining", "definite", "definitely", "definitive", "delay", "denia", "denie", "deny", "depend", "depended", "depending", "depends", "desir", "determina", "determine", "determined", "determines", "determining", "difference", "differentiat", "directly", "discern", "disciplin", "disclo", "discourag", "discover", "disorient", "disregard", "distinct", "distinguish", "doubt", "dubious", "dunno", "duti", "duty", "each", "effect", "either", "elicit", "enclos", "enlighten", "entire", "essential", "evaluat", "ever", "every", "everybod", "everything", "evident", "exact", "examin", "except", "exclu", "expect", "experiment", "explain", "explained", "explaining", "explains", "explanat", "explicit", "explor", "extremely", "fact", "facts", "factual", "fairly", "feel", "feeling", "feels", "felt", "fenc", "figur", "find", "forbid", "force", "forever", "forgave", "forget", "forgiv", "forgot", "found", "foundation", "founded", "founder", "frankly", "fundamental", "fundamentalis", "fundamentally", "fundamentals", "fuzz", "general", "generally", "generate", "generating", "generator", "grasp", "guarant", "guard", "guess", "guessed", "guesses", "guessing", "halfass", "halt", "hangup", "hardly", "harness", "hazie", "hazy", "held", "hence", "hesita", "hold", "hope", "hoped", "hopeful", "hopefully", "hopefulness", "hopes", "hoping", "how", "hows", "how's", "hypothes", "hypothetic", "idea", "ideal", "ideas", "identif", "if", "ignit", "ignor", "imagin", "implica", "implicit", "implie", "imply", "impossib", "inact", "inadequa", "inclu", "incomplet", "indecis", "indeed", "indefinit", "independ", "indetermin", "indirect", "induc", "inevitab", "infallib", "infer", "inferr", "infers", "influenc", "info", "inform", "information", "informative", "informed", "informing", "informs", "inhib", "inquir", "inside", "insight", "inspir", "intend", "intent", "interfer", "interpret", "into", "invariab", "irrational", "irrefu", "issue", "just", "justif", "keep", "keeping", "keeps", "kept", "kind (of)", "kinda", "kindof", "knew", "know", "knowab", "knower", "knowing", "knowledg", "known", "knows", "lack", "launch", "law", "lead", "learn", "led", "lesson", "liabilit", "likel", "limit", "link", "logic", "lot", "lotof", "lots ", "lotsa", "lotta", "luck", "lucked", "lucki", "luckless", "lucks", "lucky", "made", "mainly", "make ", "maker", "makes", "making", "manipul", "marginal", "matter", "may", "maybe", "mean", "meaning", "means", "meant", "memor", "might", "mightve", "might've", "mind", "misle", "mistak", "misunder", "most", "mostly", "motiv", "must", "mustnt", "must'nt", "mustn't", "mustve", "must've", "myster", "name", "nearly", "necessar", "need", "needed", "needing", "neednt", "need'nt", "needn't", "needs", "neglect", "never", "news", "normal", "not", "notice", "noticing", "obedien", "obey", "obscur", "obstac", "obvious", "occasional", "often", "open", "opinion", "oppos", "option", "or", "organiz", "origin", "originat", "origins", "ought", "oughta", "oughtnt", "ought'nt", "oughtn't", "oughtve", "ought've", "out", "outcome", "outstanding", "overall", "partly", "perceiv", "percept", "perfect", "perhaps", "permit", "pick ", "plus", "ponder", "positiv", "possib", "practically", "precis", "prefer", "presum", "pretty", "prevent", "probable", "probablistic", "probably", "problem", "produc", "prohib", "proof", "prove", "proving", "provoc", "provok", "prude", "prudes", "prudish", "pure", "purpose", "puzzl", "quer", "question", "quite", "random", "rather", "rational", "react", "real ", "reality", "realiz", "really", "rearrang", "reason", "recall", "reckon", "recogni", "recollect", "reconcil", "reconsider", "reconstruct", "reevaluat", "refer", "reflect", "refrain", "refus", "regardless", "regret", "rein", "relate", "relating", "relation", "reluctan", "rememb", "reorgani", "repress", "requir", "reserved", "resolu", "resolv", "response", "responsib", "restrain", "restrict", "restructur", "result", "retain", "rethink", "reveal", "revelat", "rigid", "root", "safe", "same", "save", "secret", "secrets", "seem", "seemed", "seeming", "seems", "sense", "sensed", "senses", "sensing", "shaki", "shaky", "should", "shouldnt", "should'nt", "shouldn't", "shoulds", "shouldve", "should've", "sign", "since", "solution", "solve", "solved", "solves", "solving", "some", "somebod", "somehow", "someone", "something", "sometime", "sometimes", "somewhat", "somewhere", "sort", "sorta", "sortof", "sorts", "sortsa", "source", "spose", "standard", "statement", "stiff", "stimul", "stop", "stopped", "stopper", "stopping", "stops", "stories", "story", "stubborn", "subdue", "suppose", "supposed", "supposes", "supposing", "supposition", "suppress", "sure", "suspect", "suspicio", "taboo", "tempora", "tentativ", "theor", "therefor", "think", "thinker", "thinking", "thinks", "thought", "thoughts", "thus", "tidi", "tidy", "tight", "total", "totally", "trigger", "TRUE", "truest", "truly", "truth", "typically", "unaccept", "unambigu", "unaware", "uncertain", "unclear", "undecided", "undeniab", "understand", "understandab", "understanding", "understands", "understood", "undesire", "undetermin", "undo ", "undoubt", "unknow", "unless", "unlikel", "unluck", "unneccess", "unneed", "unquestion", "unrelat", "unresolv", "unsettl", "unsure", "unwant", "uptight", "use", "used", "uses", "using", "usually", "vague", "variab", "varies", "vary", "versus", "veto", "vs", "wait", "waited", "waiting", "waits", "wanna", "want", "wanted", "wanting", "wants", "we", "whether", "wholly", "why", "wish", "wished", "wishes", "wishing", "with", "withheld", "withhold", "without", "wonder", "wondered", "wondering", "wonders", "word", "would", "wouldnt", "wouldn't", "wouldve", "would've", "write", "writing", "wrote", "yearn", "yield"],

    'Insight' : ["accept", "accepta", "accepted", "accepting", "accepts", "acknowledg", "adjust", "admit", "admits", "admitted", "admitting", "afterthought", "analy", "answer", "appreciat", "assum", "attent", "aware", "became", "become", "becomes", "becoming", "belief", "believe", "believed", "believes", "believing", "categor", "choice", "choos", "clarif", "closure", "cohere", "complex", "complica", "compreh", "concentrat", "conclud", "conclus", "confess", "conscious", "consider", "considered", "considering", "considers", "contemplat", "correlat", "curio", "decid", "decis", "deduc", "define", "defines", "defining", "determina", "determine", "determined", "determines", "determining", "differentiat", "discern", "disclo", "discover", "distinguish", "effect", "enlighten", "evaluat", "examin", "explain", "explained", "explaining", "explains", "explanat", "explor", "feel", "feeling", "feels", "felt", "figur", "find", "forgave", "forgiv", "found", "grasp", "idea", "ideas", "identif", "imagin", "induc", "infer", "inferr", "infers", "info", "inform", "information", "informative", "informed", "informing", "informs", "inquir", "insight", "inspir", "interpret", "justif", "knew", "know", "knowab", "knower", "knowing", "knowledg", "known", "knows", "learn", "lesson", "link", "logic", "mean", "meaning", "means", "meant", "memor", "misunder", "motiv", "notice", "noticing", "perceiv", "percept", "ponder", "prefer", "presum", "prove", "proving", "quer", "question", "rational", "realiz", "rearrang", "reason", "recall", "reckon", "recogni", "recollect", "reconcil", "reconsider", "reconstruct", "reevaluat", "refer", "reflect", "relate", "relating", "relation", "rememb", "reorgani", "resolu", "resolv", "restructur", "rethink", "reveal", "revelat", "secret", "secrets", "seem", "seemed", "seeming", "seems", "sense", "sensed", "senses", "sensing", "solution", "solve", "solved ", "solves", "solving", "statement", "suspect", "suspicio", "think", "thinker", "thinking", "thinks", "thought", "thoughts", "unaccept", "unaware", "understand", "understandab", "understanding", "understands", "understood", "unrelat", "wonder", "wondered", "wondering", "wonders"],

    'Cause' : ["activat", "affect", "affected", "affecting", "affects", "aggravat", "allow", "attribut", "based", "bases", "basis", "because", "boss", "caus", "change", "changed", "changes", "changing", "compel", "compliance", "complie", "comply", "conclud", "consequen", "control", "cos", "coz", "create", "creati", "cuz", "deduc", "depend", "depended", "depending", "depends", "effect", "elicit", "experiment", "force", "foundation", "founded", "founder", "generate", "generating", "generator", "hence", "how", "hows", "how's", "ignit", "implica", "implie", "imply", "inact", "independ", "induc", "infer", "inferr", "infers", "influenc", "intend", "intent", "justif", "launch", "lead", "led", "made", "make", "maker", "makes", "making", "manipul", "misle", "motiv", "obedien", "obey", "origin", "originat", "origins", "outcome", "permit", "pick ", "produc", "provoc", "provok", "purpose", "rational", "react", "reason", "response", "result", "root", "since", "solution", "solve", "solved", "solves", "solving", "source", "stimul", "therefor", "thus", "trigger", "use", "used", "uses", "using", "why"],

    'Discrep' : ["besides", "could", "couldnt", "couldn't", "couldve", "could've", "desir", "expect", "hope", "hoped", "hopeful", "hopefully", "hopefulness", "hopes", "hoping", "ideal", "if", "impossib", "inadequa", "lack", "liabilit", "mistak", "must", "mustnt", "must'nt", "mustn't", "mustve", "must've", "need", "needed", "needing", "neednt", "need'nt", "needn't", "needs", "normal", "ought", "oughta", "oughtnt", "ought'nt", "oughtn't", "oughtve", "ought've", "outstanding", "prefer", "problem", "rather", "regardless", "regret", "should", "shouldnt", "should'nt", "shouldn't", "shoulds", "shouldve", "should've", "undesire", "undo", "unneccess", "unneed", "unwant", "wanna", "want", "wanted", "wanting", "wants", "wish", "wished", "wishes", "wishing", "would", "wouldnt", "wouldn't", "wouldve", "would've", "yearn"],

    'Tentat' : ["allot", "almost", "alot", "ambigu", "any", "anybod", "anyhow", "anyone", "anything", "anytime", "anywhere", "apparently", "appear", "appeared", "appearing", "appears", "approximat", "arbitrar", "assum", "barely", "bet", "bets", "betting", "blur", "borderline", "chance", "confus", "contingen", "depend", "depended", "depending", "depends", "disorient", "doubt", "dubious", "dunno", "fairly", "fuzz", "generally", "guess", "guessed", "guesses", "guessing", "halfass", "hardly", "hazie", "hazy", "hesita", "hope", "hoped", "hopeful", "hopefully", "hopefulness", "hopes", "hoping", "hypothes", "hypothetic", "if", "incomplet", "indecis", "indefinit", "indetermin", "indirect", "kind (of)", "kinda", "kindof", "likel", "lot", "lotof", "lots ", "lotsa", "lotta", "luck", "lucked", "lucki", "luckless", "lucks", "lucky", "mainly", "marginal", "may", "maybe", "might", "mightve", "might've", "most", "mostly", "myster", "nearly", "obscur", "occasional", "often", "opinion", "option", "or", "overall", "partly", "perhaps", "possib", "practically", "pretty", "probable", "probablistic", "probably", "puzzl", "question", "quite", "random", "seem", "seemed", "seeming", "seems", "shaki", "shaky", "some", "somebod", "somehow", "someone", "something", "sometime", "sometimes", "somewhat", "sort", "sorta", "sortof", "sorts", "sortsa", "spose", "suppose", "supposed", "supposes", "supposing", "supposition", "tempora", "tentativ", "theor", "typically", "uncertain", "unclear", "undecided", "undetermin", "unknow", "unlikel", "unluck", "unresolv", "unsettl", "unsure", "usually", "vague", "variab", "varies", "vary", "wonder", "wondered", "wondering", "wonders"],


    'Certain' : ["absolute", "absolutely", "accura", "all", "altogether", "always", "apparent", "assur", "blatant", "certain", "clear", "clearly", "commit", "commitment", "commits", "committ", "complete", "completed", "completely", "completes", "confidence", "confident", "confidently", "correct", "defined", "definite", "definitely", "definitive", "directly", "distinct", "entire", "essential", "ever", "every", "everybod", "everything", "evident", "exact", "explicit", "extremely", "fact", "facts", "factual", "forever", "frankly", "fundamental", "fundamentalis", "fundamentally", "fundamentals", "guarant", "implicit", "indeed", "inevitab", "infallib", "invariab", "irrefu", "must", "mustnt", "must'nt", "mustn't", "mustve", "must've", "necessar", "never", "obvious", "perfect", "positiv", "precis", "proof", "prove", "pure", "sure", "total", "totally", "TRUE", "truest", "truly", "truth", "unambigu", "undeniab", "undoubt", "unquestion", "wholly"],


    'Inhib' : ["abandon", "abstain", "anal", "avert", "avoid", "ban", "banned", "banning", "bans", "barrier", "binding", "block", "blocked", "blocker", "blocking", "blocks", "bound", "brake", "bridle", "careful", "caut", "cease", "ceasing", "compulsiv", "confin", "conflict", "conserv", "constrain", "constrict", "contain", "contradic", "control", "curb", "curtail", "defenc", "defens", "delay", "denia", "denie", "deny", "disciplin", "discourag", "disregard", "duti", "duty", "enclos", "fenc", "forbid", "forgot", "guard", "halt", "hangup", "harness", "held", "hesita", "hold", "ignor", "inhib", "interfer", "keep", "keeping", "keeps", "kept", "limit", "neglect", "obstac", "oppos", "prevent", "prohib", "protect", "prude", "prudes", "prudish", "refrain", "refus", "rein", "reluctan", "repress", "requir", "reserved", "responsib", "restrain", "restrict", "retain", "rigid", "safe", "secur", "stiff", "stop", "stopped", "stopper", "stopping", "stops", "stubborn", "subdue", "suppress", "taboo", "tidi", "tidy", "tight", "uptight", "veto", "wait", "waited", "waiting", "waits", "wariness", "wary", "withheld", "withhold", "yield"],

    'Incl' : ["add", "addit", "along", "and", "around", "both", "came", "close", "come", "each", "inclu", "inside", "into", "open", "out", "plus", "we", "with"],

    'Excl' : ["but", "either", "except", "exclu", "if", "just", "not", "or", "rather", "really", "something", "sometime", "unless", "versus", "vs", "whether", "without"],

    'Percept' : ["acid", "acrid", "aroma", "audibl", "audio", "beaut", "bitter", "black", "blacke", "blackish", "blacks", "blind", "blond", "blue", "boom", "bright", "brown", "brush", "butter", "candle", "caramel", "caress", "chocolate", "choir", "circle", "citrus", "click", "cold", "cologne", "color", "colour", "column", "concert", "cool", "cream", "deaf", "delectabl", "delicious", "deoder", "drie", "drily", "drool", "dry", "ear", "ears", "edge", "edges", "edging", "experienc", "eye", "eying", "feel", "feeling", "feels", "felt", "fetid", "finger", "fire", "fizz", "flavor", "flavour", "flexib", "fragil", "fragran", "freez", "froze", "fruit", "fuzz", "gaz", "glanc", "glow", "grab", "gray", "greas", "green", "grey", "grip", "gripp", "grips", "hair", "hand", "handful", "hands", "hard", "harde", "harmon", "hear", "heard", "hearing", "hears", "heavie", "heavy", "honey", "hot", "hott", "hush", "image", "inaudibl", "inhal", "leather", "lick", "light", "limp", "listen", "listened", "listener", "listening", "listens", "lit", "look", "looked", "looker", "looking", "looks", "loose", "loud", "mint", "musi", "nasal", "noise", "noises", "noisy", "nose", "nostril", "odor", "odour", "oil", "orange", "palatabl", "perfum", "picture", "pink", "press", "pressed", "presser", "presses", "pungen", "purpl", "quiet", "rancid", "rang", "rectang", "red", "redde", "reddish", "redness", "reds", "reek", "ring", "ringing", "rings", "rotten", "rough", "round", "rub", "rubbed", "rubbing", "rubs", "saccharine", "said", "saliv", "salt", "sampl", "sand", "sands", "sandy", "sang", "savor", "savour", "saw", "say", "scan", "scann", "scans", "scent", "scratch", "scream", "screen", "scrumptious", "see", "seeing", "seen", "seer", "sees", "sharp", "shine", "shini", "shiny", "shout", "sight", "silen", "silk", "skin", "skin'", "smell", "smooth", "sniff", "snort", "soft", "song", "sound", "sour", "soure", "souri", "sours", "soury", "speak", "speaker", "speaking", "speaks", "speech", "spice", "spiced", "spices", "spicy", "spoke", "squar", "squeez", "stank", "stare", "staring", "stench", "stink", "stroke", "stroki", "stunk", "sugar", "sumptuous", "sunli", "sunshin", "sweet", "sweetness", "sweets", "tang", "tangy", "tart", "tast", "thick", "thin", "thinn", "thunder", "tight", "tongue", "touch", "triang", "unsavo", "view", "viewer", "viewing", "views", "vivid", "voic", "waft", "warm", "watch", "weight", "weighted", "weighting", "weightless", "weightlift", "weights", "wet", "wetly", "whiff", "whisper", "white", "whitish", "yell", "yelled", "yelling", "yellow", "yells", "yum"],

    'See' : ["beaut", "black", "blacke", "blackish", "blacks", "blind", "blond", "blue", "bright", "brown", "candle", "circle", "click", "color", "colour", "column", "cream", "eye", "eying", "gaz", "glanc", "glow", "gray", "green", "grey", "image", "lit", "look", "looked", "looker", "looking", "looks", "orange", "picture", "pink", "purpl", "rectang", "red", "redde", "reddish", "redness", "reds", "round", "saw", "scan", "scann", "scans", "screen", "see", "seeing", "seen", "seer", "sees", "shine", "shini", "shiny", "sight", "squar", "stare", "staring", "sunli", "sunshin", "triang", "view", "viewer", "viewing", "views", "vivid", "watch", "white", "whitish", "yellow"],

    'Hear' : ["audibl", "audio", "boom", "choir", "concert", "deaf", "ear", "ears", "harmon", "hear", "heard", "hearing", "hears", "hush", "inaudibl", "listen", "listened", "listener", "listening", "listens", "loud", "musi", "noise", "noises", "noisy", "quiet", "rang", "ring", "ringing", "rings", "said", "sang", "say", "scream", "shout", "silen", "song", "sound", "speak", "speaker", "speaking", "speaks", "speech", "spoke", "thunder", "voic", "whisper", "yell", "yelled", "yelling", "yells"],

    'Feel' : ["brush", "caress", "cold", "cool", "drie", "drily", "dry", "edge", "edges", "edging", "feel", "feeling", "feels", "felt", "finger", "fire", "flexib", "fragil", "freez", "froze", "fuzz", "grab", "grip", "gripp", "grips", "hair", "hand", "handful", "hands", "hard", "harde", "heavie", "heavy", "hot", "hott", "leather", "limp", "loose", "press", "pressed", "presser", "presses", "rough", "round", "rub", "rubbed", "rubbing", "rubs", "sand", "sands", "sandy", "scratch", "sharp", "silk", "skin", "skin'", "smooth", "soft", "squeez", "stroke", "stroki", "thick", "thin", "thinn", "tight", "touch", "warm", "weight", "weighted", "weighting", "weightless", "weightlift", "weights", "wet", "wetly"],

    'Bio' : ["abdomen", "abortion", "abs", "ache", "aching", "acne", "addict", "advil", "aids", "alcohol", "alive", "allerg", "amput", "anal", "ankle", "anorexi", "antacid", "antidepressant", "anus", "appendic", "appendix", "appeti", "arch", "arm", "armpit", "arms", "arous", "arse", "arses", "arter", "arthr", "asleep", "aspirin", "ass", "asses", "asthma", "ate", "bake", "baking", "bald", "bandage", "bandaid", "bar", "bars", "beer", "bellies", "belly", "bi", "bicep", "binge", "binging", "bipolar", "bladder", "bleed", "blind", "blood", "bloody", "bodi", "body", "boil", "bone", "boner", "bones", "bony", "boob", "booz", "bowel", "brain", "bread", "breakfast", "breast", "breath", "bronchi", "brunch", "bulimi", "burp", "butt", "butts", "butt's", "cafeteria", "cancer", "candie", "candy", "cardia", "cardio", "checkup", "cheek", "chest", "chew", "chills", "chiropract", "chlamydia", "chok", "cholester", "chow", "chronic", "cigar", "clinic", "clothes", "cock", "cocks", "codeine", "coffee", "coke", "colon", "colono", "colons", "coma", "condom", "condoms", "congest", "constipat", "contag", "cook", "cornea", "coronar", "cough", "cramp", "crap", "crotch", "cuddl", "cyst", "deaf", "decongest", "dentist", "derma", "dessert", "detox", "diabet", "diagnos", "diarr", "dick", "dicks", "diet", "digest", "dine ", "dined", "diner", "diners", "dines", "dining", "dinner", "disease", "dish", "dishes", "dizz", "doctor", "dosage", "dose", "dosing", "dr", "drank", "drink", "drool", "drows", "drs", "drug", "drunk", "dx", "dyke", "ear", "ears", "eat", "eaten", "eating", "eats", "egg", "elbow", "emphysem", "enema", "erectile", "erection", "erotic", "espresso", "estrogen", "exercis", "exhaust", "eye", "face", "faces", "facial", "faint", "farsighted", "fat", "fatigu", "fats", "fatt", "fed", "feed", "feeder", "feeding", "feeds", "feet", "fever", "finger", "flesh", "flu", "food", "foot", "forearm", "forehead", "foreplay", "fries", "fruit", "fry", "fuck", "fucked", "fucker", "fuckin", "fucks", "gay", "gays", "genital", "gland", "glaucoma", "glutton", "gobble", "gobbling", "gonorrhea", "goosebump", "grocer", "gulp", "gums", "gut", "guts", "gynecolog", "gyno", "hair", "hallucinat", "hamstring", "hand", "hands", "hangover", "head", "headache", "heads", "heal", "healed", "healer", "healing", "heals", "health", "heart", "heartburn", "hearts", "heel", "helpings", "hemor", "herpes", "hiccup", "hip", "hips", "hiv", "ho", "homo", "homos", "homosexual", "hormone", "hornie", "horny", "hospital", "hug", "hugg", "hugs", "hump", "hunger", "hungover", "hungr", "hyperten", "hypotherm", "ibuprofen", "ICU", "ill", "illness", "immun", "incest", "indigestion", "infect", "inflam", "ingest", "injur", "insomnia", "insulin", "intestin", "intox", "itch", "iv", "jaw", "jissom", "jizz", "joints", "kidney", "kiss", "kitchen", "knee", "knuckle", "leg", "legs", "lesbian", "leuke", "libid", "life", "lip", "lips", "liquor", "liver", "living", "love", "loved", "lover", "loves", "lozenge", "lump", "lunch", "lung", "lust", "lymph", "makeout", "mammogram", "manicdep", "meal", "medic", "migrain", "milk", "miscar", "mono", "mouth", "mri ", "mucous", "muscle", "muscular", "myopi", "naked", "nasal", "nause", "nearsighted", "neck", "nerve", "neural", "neurolog", "neuron", "nipple", "nose", "nostril", "nude", "nudi", "numb", "nurse", "nutrition", "obes", "OCD", "optometr", "orgasm", "orgies", "orgy", "orthodon", "orthoped", "ovar", "overate", "overeat", "overweight", "pain", "pained", "painf", "paining", "painl", "pains", "palm", "palms", "pap", "paraly", "passion", "pasta", "patholog", "pediatr", "pee", "pelvi", "penis", "perspir", "perver", "pharmac", "phobi", "physical", "physician", "pill", "pills", "pimple", "piss", "pizza", "pms", "podiatr", "poison", "poop", "porn", "pregnan", "prescri", "prick", "prognos", "prostat", "prostitu", "prozac", "prude", "prudes", "prudish", "pubic", "puk", "pulse", "puss", "pussies", "pussy", "queas", "queer", "rape", "raping", "rapist", "rash", "rehab", "restau", "retina", "rib", "ribs", "ritalin", "rx", "salad", "saliv", "sandwich", "scab", "scalp", "schizophren", "scrape", "screw", "seduc", "seizure", "sensation", "sensations", "servings", "sex", "shirt", "shit", "shoe", "shoulder", "sick", "sickday", "sicker", "sickest", "sickleave", "sickly", "sickness", "sinus", "skelet", "skin", "skinni", "skinny", "skull", "sleep", "slender", "slept", "slut", "smok", "snack", "soda", "sore", "spat", "spinal", "spine", "spit", "spits", "spitting", "starve", "starving", "std", "stiff", "stomach", "strept", "stroke", "stud", "stuffed", "sugar", "sunburn", "supper", "surgeon", "surger", "swallow", "sweat", "swelling", "swollen", "symptom", "syndrome", "syphili", "tast", "tea", "teeth", "tender", "tendon ", "tendoni", "tendons", "testosterone", "therap", "thermometer", "thigh", "thirst", "throat", "throb", "thyroid", "tingl", "tire", "tiring", "tit", "tits", "titties", "titty", "toe", "toenail", "toes", "tongue", "tonsils", "tooth", "tox", "tricep", "tumo", "twitch", "tylenol", "ulcer", "unhealth", "urin", "uter", "vagina", "vd", "veget", "veggie", "vein", "vertigo", "viagra", "vicodin", "virgin", "vitamin", "vomit", "waist", "wake", "wart", "warts", "wash", "water", "weak", "wear", "weary", "weigh ", "weighed", "weighing", "weighs", "weight", "wheez", "whiskey", "whisky", "whore", "wine", "wines", "withdrawal", "womb", "wound", "wrist", "xanax", "xray", "yawn", "zit", "zits", "zoloft"],

    'Body' : ["abdomen", "abs", "anal", "ankle", "anus", "appendix", "arch", "arm", "armpit", "arms", "arous", "arse", "arses", "arter", "asleep", "ass", "asses", "bald", "bellies", "belly", "bicep", "bladder", "blood", "bloody", "bodi", "body", "bone", "bones", "bony", "boob", "bowel", "brain", "breast", "breath", "butt", "butts", "cheek", "chest", "clothes", "cock", "cocks", "colon", "colons", "cornea", "crap", "crotch", "dick", "dicks", "drool", "ear", "ears", "elbow", "erectile", "erection", "eye", "face", "faces", "facial", "fat", "fatt", "feet", "finger", "flesh", "foot", "forearm", "forehead", "genital", "goosebump", "gums", "gut", "guts", "hair", "hamstring", "hand", "hands", "head", "heads", "heart", "hearts", "heel", "hip", "hips", "hornie", "horny", "intestin", "itch", "jaw", "joints", "kidney", "knee", "knuckle", "leg", "legs", "lip", "lips", "liver", "lung", "mouth", "mucous", "muscle", "muscular", "naked", "nasal", "neck", "nerve", "neural", "neuron", "nipple", "nose", "nostril", "nude", "nudi", "orgasm", "ovar", "palm", "palms", "pee", "pelvi", "penis", "perspir", "piss", "poop", "prick", "prostat", "pulse", "pussies", "pussy", "rash", "retina", "rib", "ribs", "saliv", "scalp", "sensation", "sensations", "shirt", "shit", "shoe", "shoulder", "skelet", "skin", "skinni", "skull", "sleep", "slender", "slept", "spat", "spinal", "spine", "spit", "spits", "spitting", "stomach", "sweat", "teeth", "tendon ", "tendons", "thigh", "thirst", "throat", "tit", "tits", "titties", "titty", "toe", "toenail", "toes", "tongue", "tonsils", "tooth", "tricep", "urin", "uter", "vagina", "vein", "waist", "wake", "wear", "womb", "wrist"],


    'Health' : ["abortion", "ache", "aching", "acne", "addict", "advil", "aids", "alcohol", "alive", "allerg", "amput", "anorexi", "antacid", "antidepressant", "appendic", "arthr", "aspirin", "asthma", "bandage", "bandaid", "binge", "binging", "bipolar", "bleed", "blind", "bronchi", "bulimi", "burp", "cancer", "cardia", "cardio", "checkup", "chills", "chiropract", "chlamydia", "chok", "cholester", "chronic", "clinic", "codeine", "colono", "coma", "congest", "constipat", "contag", "coronar", "cough", "cramp", "cyst", "deaf", "decongest", "dentist", "derma", "detox", "diabet", "diagnos", "diarr", "digest", "disease", "dizz", "doctor", "dosage", "dose", "dosing", "dr", "drows", "drs", "drug", "dx", "emphysem", "enema", "estrogen", "exercis", "exhaust", "faint", "farsighted", "fat", "fatigu", "fats", "fatt", "fever", "flu", "gland", "glaucoma", "gynecolog", "gyno", "hallucinat", "hangover", "headache", "heal", "healed", "healer", "healing", "heals", "health", "heartburn", "hemor", "herpes", "hiccup", "hiv", "hormone", "hospital", "hungover", "hyperten", "hypotherm", "ibuprofen", "ICU", "ill", "illness", "immun", "indigestion", "infect", "inflam", "ingest", "injur", "insomnia", "insulin", "intox", "itch", "iv", "leuke", "life", "living", "lozenge", "lump", "lymph", "mammogram", "manicdep", "medic", "migrain", "miscar", "mono", "mri ", "myopi", "nause", "nearsighted", "neurolog", "numb", "nurse", "nutrition", "obes", "OCD", "optometr", "orthodon", "orthoped", "overweight", "pain", "pained", "painf", "paining", "painl", "pains", "pap", "paraly", "patholog", "pediatr", "pharmac", "phobi", "physical", "physician", "pill", "pills", "pimple", "pms", "podiatr", "poison", "pregnan", "prescri", "prognos", "prozac", "puk", "puss", "queas", "rehab", "ritalin", "rx", "scab", "schizophren", "scrape", "seizure", "sick", "sickday", "sicker", "sickest", "sickleave", "sickly", "sickness", "sinus", "sore", "std", "stiff", "strept", "stroke", "sunburn", "surgeon", "surger", "swelling", "swollen", "symptom", "syndrome", "syphili", "tender", "tendoni", "testosterone", "therap", "thermometer", "throb", "thyroid", "tingl", "tire", "tiring", "tox", "tumo", "twitch", "tylenol", "ulcer", "unhealth", "vertigo", "viagra", "vicodin", "vitamin", "vomit", "wart", "warts", "wash", "weak", "weary", "wheez", "withdrawal", "wound", "xanax", "xray", "yawn", "zit", "zits", "zoloft"],

    'Sexual' : ["abortion", "aids", "arous", "ass", "asses", "bi", "boner", "boob", "breast", "butt", "butts", "butt's", "chlamydia", "cock", "cocks", "condom", "condoms", "cuddl", "dick", "dicks", "dyke", "erectile", "erection", "erotic", "foreplay", "fuck", "fucked", "fucker", "fuckin", "fucks", "gay", "gays", "genital", "gonorrhea", "hiv", "ho", "homo", "homos", "homosexual", "hornie", "horny", "hug", "hugg", "hugs", "hump", "incest", "jissom", "jizz", "kiss", "lesbian", "libid", "love", "loved", "lover", "loves", "lust", "makeout", "naked", "nipple", "nude", "orgasm", "orgies", "orgy", "ovar", "passion", "penis", "perver", "porn", "pregnan", "prostat", "prostitu", "prude", "prudes", "prudish", "pubic", "pussy", "queer", "rape", "raping", "rapist", "screw", "seduc", "sex", "slut", "std", "stud", "syphili", "tit", "tits", "titties", "titty", "vagina", "vd", "virgin", "whore", "womb"],


    'Ingest' : ["alcohol", "anorexi", "appeti", "ate", "bake", "baking", "bar", "bars", "beer", "binge", "binging", "boil", "booz", "bread", "breakfast", "brunch", "bulimi", "cafeteria", "candie", "candy", "chew", "chow", "cigar", "cocktail", "coffee", "coke", "cook", "dessert", "diet", "digest", "dine", "dined", "diner", "diners", "dines", "dining", "dinner", "dish", "dishes", "drank", "drink", "drunk", "eat", "eaten", "eating", "eats", "egg", "espresso", "fat", "fats", "fatt", "fed", "feed", "feeder", "feeding", "feeds", "food", "fries", "fruit", "fry", "glutton", "gobble", "gobbling", "grocer", "gulp", "helpings", "hunger", "hungr", "ingest", "kitchen", "liquor", "lunch", "meal", "milk", "obes", "overate", "overeat", "overweight", "pasta", "pizza", "restau", "salad", "sandwich", "servings", "skinni", "skinny", "smok", "snack", "soda", "starve", "starving", "stuffed", "sugar", "supper", "swallow", "tast", "tea", "thirst", "veget", "veggie", "waist", "water", "weigh ", "weighed", "weighing", "weighs", "weight", "whiskey", "whisky", "wine", "wines"],

    'Relativ' : ["above", "abrupt", "across", "act", "action", "advanc", "after", "afterlife", "aftermath", "afternoon", "afterthought", "afterward", "again", "age", "aged", "ages", "aging", "ago", "ahead", "air", "already", "always", "among", "ancient", "annual", "anymore", "anytime", "anywhere", "apart", "appear", "appeared", "appearing", "appears", "approach", "april", "area", "around", "arrival", "arrive", "arrived", "arrives", "arriving", "at", "atop", "attend", "attended", "attending", "attends", "august", "autumn", "away", "awhile", "back", "backward", "before", "began", "begin", "beginn", "begins", "begun", "behavio", "below", "bend", "bending", "bends", "beneath", "bent", "beside", "beyond", "biannu", "big", "bigger", "biggest", "bimonth", "birth", "biweek", "born", "both", "bottom", "breadth", "break", "brief", "bring", "bringing", "brings", "brink", "broad", "brought", "building", "busy", "came", "capacit", "car", "carried", "carrier", "carries", "carry ", "carrying", "catch", "caught", "cease", "ceasing", "ceiling", "center", "centre", "centur", "change", "changed", "changes", "changing", "childhood", "christmas", "city", "climb", "clock", "close", "closed", "closely", "closer", "closes", "closest", "closing", "come", "comes", "coming", "common", "connection", "constant ", "constantly", "continu", "corner", "corners", "countr", "coverage", "cross", "cruis", "current", "cycle", "dail", "danc", "date", "day", "decade", "decay", "december", "deep", "delay", "deliver", "dense", "densit", "depart", "departed", "departing", "departs", "departure", "depth", "diagonal", "disappear", "distan", "down", "downward", "drift", "drive", "driven", "drives", "driving", "drop ", "drove", "due", "during", "earli", "early", "east", "edge", "edges", "edging", "empt", "enclos", "encompass", "end", "ended", "ending", "ends", "enorm", "enter", "entered", "entering", "enters", "entrance", "environment", "era", "etern", "eve", "evening", "event", "eventually", "ever", "everyday", "everywhere", "exit", "explor", "exterior", "fade", "fading", "fall", "fallen", "falling", "falls", "far", "farther", "farthest", "fast", "faster", "fastest", "february", "fell", "fill", "final ", "finally", "finish", "first", "firstly", "firsts", "fit", "fled", "flee", "flew", "flies", "floor", "flow", "fly", "flying", "follow", "followed", "following", "follows", "followup", "forever", "former", "forward", "frequent", "frequented", "frequenting", "frequently", "frequents", "Friday", "front", "full", "further", "futur", "generation", "giant", "gigantic", "ginormous", "global", "go", "goes", "going", "gone", "grew", "ground", "grow", "growing", "grown", "growth", "hall", "hang", "happening", "headed", "heading", "height", "high", "highe", "hik", "histor", "horizontal", "hour", "huge", "hurrie", "hurry", "immediate", "immediately", "immediateness", "immortal", "in", "inch", "inciden", "increas", "infinit", "initial", "initiat", "inner", "inside", "insides", "instan", "interior", "internal", "internation", "intersect", "interval", "into", "january", "jog", "journey", "july", "jump", "june", "kilometer", "km", "land", "large", "last", "late", "lately", "later", "latest", "lead", "leave", "leaves", "leaving", "led", "ledge", "ledging", "left", "level", "levels", "littl", "local", "long", "longe", "longitud", "low", "map", "mapped", "mapping", "maps", "march", "mass", "meantime", "meanwhile", "meter", "metre", "mid", "middl", "mile", "min", "minute", "modern", "moment", "monday", "month", "morning", "motion", "move", "moved", "movement", "mover", "moves", "moving", "narrow", "nation", "near", "neared", "nearer", "nearest", "nearing", "nears", "never", "new", "newer", "newest", "newly", "next", "night", "nightly", "nights", "noon", "north", "november", "now", "nowhere", "occasional", "oclock", "o'clock", "october", "off", "old", "olden", "older", "oldest", "on", "once", "onto", "open", "opened", "opening", "order", "origin", "out", "outer", "outside", "outsides", "outward", "over", "overflow", "overlap", "pass", "passed", "passes", "passing", "past", "period", "perpetual", "place", "placing", "platform", "point", "position", "post", "preced", "present", "presently", "prior", "proceed", "pull", "push", "put", "puts", "putting", "quick", "ran", "receiv", "recency", "recent", "recur", "remote", "remov", "repeat", "repetit", "replace", "replacing", "respectively", "return", "rhythm", "ridden", "ride", "rides", "riding", "right", "rise", "rising", "road", "rode", "room", "roomate", "roomed", "roomie", "rooming", "roommate", "rooms", "run", "runner", "running", "runs", "rush", "saturday", "schedul", "season", "seconds", "section", "segment", "send", "senior", "sent", "separat", "september", "sequen", "set", "shake", "shape", "shaping", "shook", "short", "shut", "side", "sides", "siding", "simultaneous", "sit", "site", "sitting", "sky", "slid", "slide", "slides", "sliding", "slow", "small", "sometime ", "sometimes", "somewhere", "soon", "soone", "south", "space", "spaced", "spaces", "spaci", "span", "spann", "sped", "speed", "spring", "stage", "stair", "stand", "start", "started", "starter", "starting", "starts", "startup", "state ", "stay", "step", "stepp", "steps", "still", "stop", "stopped", "stopper", "stopper", "stopping", "stops", "straight", "street", "stretch", "subsequen", "sudden", "summer", "sunday", "surfac", "surround", "swam", "swim", "synch", "system", "tall", "taller", "tallest", "tempora", "term ", "terminat", "territor", "then", "thick", "thin", "thinly", "thinn", "threw", "throw", "thursday", "til", "till", "time", "timing", "tinier", "tiniest", "tiny", "today", "together", "tomorrow", "tonight", "took", "top", "toward", "town", "transact", "transport", "travel", "trip", "tripped", "tripping", "trips", "tuesday", "under", "underneath", "undersid", "universe", "until", "up", "updat", "upon", "upper", "uppermost", "upstairs", "usual", "usually", "vast", "verg", "vertical", "visit", "walk", "walked", "walking", "walks", "wall", "walling", "walls", "way ", "wednesday", "week", "week'", "weekend", "weekl", "weeks", "went", "west", "when", "whenever", "where", "wheres", "where's", "wherever", "while", "whilst", "wide", "width", "winter", "within", "world", "year ", "yearly", "years", "yesterday", "yet", "young", "youth"],

    'Motion' : ["act", "action", "advanc", "appear", "appeared", "appearing", "appears", "approach", "arrival", "arrive", "arrived", "arrives", "arriving", "attend", "attended", "attending", "attends", "behavio", "brief", "bring", "bringing", "brings", "brought", "came", "car", "carried", "carrier", "carries", "carry ", "carrying", "catch", "caught", "change", "changed", "changes", "changing", "climb", "closely", "closes", "closing", "come", "comes", "coming", "cross", "cruis", "danc", "deliver", "depart", "departed", "departing", "departs", "departure", "disappear", "drift", "drive", "driven", "drives", "driving", "drove", "enter", "entered", "entering", "enters", "explor", "fall", "fallen", "falling", "falls", "fell", "fill", "fled", "flee", "flew", "flies", "flow", "fly", "flying", "follow", "followed", "following", "follows", "forward", "front", "go", "goes", "going", "gone", "grew", "grow", "growing", "grown", "growth", "hang", "headed", "heading", "hik", "increas", "jog", "journey", "jump", "lead", "leave", "leaves", "leaving", "led", "motion", "move", "moved", "movement", "mover", "moves", "moving", "pass", "passed", "passes", "passing", "pull", "push", "put", "puts", "putting", "ran", "receiv", "remov", "replace", "replacing", "ridden", "ride", "rides", "riding", "rise", "rising", "rode", "run", "runner", "running", "runs", "rush", "send", "sent", "shake", "shook", "slid", "slide", "slides", "sliding", "step", "stepp", "steps", "stopper", "swam", "swim", "threw", "throw", "took", "transact", "transport", "travel", "trip", "tripped", "tripping", "trips", "visit", "walk", "walked", "walking", "walks", "went"],


    'Space' : ["above", "across", "air", "among", "anywhere", "apart", "area", "around", "at", "atop", "away", "backward", "below", "bend", "bending", "bends", "beneath", "bent", "beside", "beyond", "big", "bigger", "biggest", "both", "bottom", "breadth", "brink", "broad", "capacit", "ceiling", "center", "centre", "city", "close", "closed", "closer", "closest", "connection", "corner", "corners", "countr", "coverage", "deep", "dense", "densit", "depth", "diagonal", "distan", "down", "downward", "east", "edge", "edges", "edging", "empt", "enclos", "encompass", "enorm", "entrance", "environment", "everywhere", "exit", "exterior", "far", "farther", "farthest", "fit", "floor", "forward", "full", "further", "giant", "gigantic", "ginormous", "global", "ground", "hall", "height", "high", "highe", "horizontal", "huge", "in", "inch", "inner", "inside", "insides", "interior", "internal", "internation", "intersect", "into", "kilometer", "km", "land", "large", "ledge", "ledging", "left", "level", "levels", "littl", "local", "longitud", "low", "map", "mapped", "mapping", "maps", "mass", "meter", "metre", "mid", "middl", "mile", "narrow", "nation", "near", "neared", "nearer", "nearest", "nearing", "nears", "north", "nowhere", "off", "on", "onto", "open", "opened", "opening", "out", "outer", "outside", "outsides", "outward", "over", "overflow", "overlap", "place", "placing", "platform", "point", "position", "remote", "right", "road", "room", "roomate", "roomed", "roomie", "rooming", "roommate", "rooms", "section", "segment", "separat", "shape", "shaping", "short", "shut", "side", "sides", "siding", "site", "sky", "small", "somewhere", "south", "space", "spaced", "spaces", "spaci", "span", "spann", "stair", "straight", "street", "stretch", "surfac", "surround", "tall", "taller", "tallest", "territor", "thick", "thin", "thinly", "thinn", "tinier", "tiniest", "tiny", "together", "top", "toward", "town", "under", "underneath", "undersid", "universe", "up", "upon", "upper", "uppermost", "upstairs", "vast", "verg", "vertical", "wall", "walling", "walls", "west", "where", "wheres", "where's", "wherever", "wide", "width", "within", "world"],

    'Time' : ["abrupt", "after", "afterlife", "aftermath", "afternoon", "afterthought", "afterward", "again", "age", "aged", "ages", "aging", "ago", "ahead", "already", "always", "ancient", "annual", "anymore", "anytime", "april", "august", "autumn", "awhile", "back", "before", "began", "begin", "beginn", "begins", "begun", "biannu", "bimonth", "birth", "biweek", "born", "busy", "bye", "cease", "ceasing", "centur", "childhood", "christmas", "clock", "common", "constant ", "constantly", "continu", "current", "cycle", "dail", "date", "day", "decade", "decay", "december", "delay", "due", "during", "earli", "early", "end", "ended", "ending", "ends", "era", "etern", "eve", "evening", "event", "eventually", "ever", "everyday", "fade", "fading", "fast", "faster", "fastest", "february", "final ", "finally", "finish", "first", "firstly", "firsts", "followup", "forever", "former", "forward", "frequent", "frequented", "frequenting", "frequently", "frequents", "Friday", "futur", "generation", "happening", "histor", "hour", "hurrie", "hurry", "immediate", "immediately", "immediateness", "immortal", "inciden", "infinit", "initial", "initiat", "instan", "interval", "january", "july", "june", "last", "late", "lately", "later", "latest", "like", "long", "longe", "march", "meantime", "meanwhile", "min", "minute", "modern", "moment", "monday", "month", "morning", "never", "new", "newer", "newest", "newly", "next", "night", "nightly", "nights", "noon", "november", "now", "occasional", "oclock", "o'clock", "october", "old", "olden", "older", "oldest", "once", "origin", "past", "period", "perpetual", "preced", "present", "presently", "prior", "proceed", "quick", "recency", "recent", "recur", "repeat", "repetit", "respectively", "return", "rhythm", "saturday", "schedul", "season", "seconds", "senior", "september", "sequen", "simultaneous", "slow", "sometime ", "sometimes", "soon", "soone", "sped", "speed", "spring", "start", "started", "starter", "starting", "starts", "startup", "still", "stop", "stopped", "stopper", "stopping", "stops", "subsequen", "sudden", "summer", "sunday", "synch", "tempora", "term ", "terminat", "then", "thursday", "til", "till", "time", "timing", "today", "tomorrow", "tonight", "tuesday", "until", "updat", "usual", "usually", "wednesday", "week", "week'", "weekend", "weekl", "weeks", "when", "whenever", "while", "whilst", "winter", "year ", "yearly", "years", "yesterday", "yet", "young", "youth"],

    'Work' : ["absent", "academ", "accomplish", "achiev", "administrat", "advertising", "advis", "agent", "agents", "ambiti", "applicant", "applicat", "apprentic", "assign", "assistan", "associat", "auditorium", "award", "beaten", "benefits", "biolog", "biz", "blackboard", "bldg", "book", "boss", "broker", "bureau", "burnout", "business", "busy", "cafeteria", "calculus", "campus", "career", "ceo", "certif", "chairm", "chalk", "challeng", "champ", "class", "classes", "classmate", "classroom", "collab", "colleague", "colleg", "com", "commerc", "commute", "commuting", "companies", "company", "comput", "conferenc", "conglom", "consult", "consumer", "contracts", "corp ", "corporat", "corps", "counc", "couns", "course", "coworker", "credential", "credit", "cubicle", "curricul", "customer", "cv", "deadline", "dean", "delegat", "demote", "department", "dept", "desk", "diplom", "director", "dissertat", "dividend", "doctor", "dorm", "dotcom", "downsiz", "dropout", "duti", "duty", "earn", "econ", "edit", "educat", "elementary", "employ", "esl", "exam", "exams", "excel", "executive", "expel", "expulsion", "factories", "factory", "facult", "fail", "fax", "feedback", "finaliz", "finals", "financ", "fired", "firing", "franchis", "frat", "fratern", "freshm", "gmat", "goal", "gov", "govern", "gpa", "grad", "grade", "grading", "graduat", "gre", "hardwork", "headhunter", "highschool", "hire", "hiring", "homework", "inc", "income", "incorp", "industr", "instruct", "interview", "inventory", "jd", "job", "junior", "keyboard", "kinderg", "labor", "labour", "laidoff", "laptop", "lawyer ", "layoff", "lead", "learn", "lectur", "legal", "librar", "lsat", "ltd", "mailroom", "majoring", "majors", "manag", "manufact", "market", "masters", "math", "mcat", "mda", "meeting", "memo", "memos", "menial", "mentor", "merger", "mfg", "mfr", "mgmt", "mgr", "midterm", "motiv", "negotiat", "ngo", "nonprofit", "occupa", "office", "org", "organiz", "outlin", "outsourc", "overpaid", "overtime", "overworked", "paper", "pay", "pc", "pen", "pencil", "pens ", "pension", "phd", "photocop", "pledg", "police", "policy", "political", "politics", "practice", "prereq", "presentation", "presiden", "procrastin", "produc", "prof", "profession", "professor", "profit", "profs", "program", "project ", "projector", "projects", "prom", "promot", "psych", "psychol", "publish", "qualifi", "quiz", "read ", "recruit", "register", "registra", "report", "requir", "research", "resource", "resources", "resourcing", "responsib", "resume", "retire", "retiring", "review", "rhetor", "salar", "scholar", "scholaring", "scholarly", "scholars", "scholarship", "scholastic", "school", "scien", "secretar", "sector", "semester", "senior", "servic", "session", "sickday", "sickleave", "sophom", "sororit", "staff", "stapl", "stipend", "stock", "stocked", "stocker", "stocks", "student", "studied", "studies", "studious", "study", "succeed", "success", "supervis", "syllabus", "taught", "tax", "taxa", "taxed", "taxes", "taxing", "teach", "team", "tenure", "test", "tested", "testing", "tests", "textbook", "theses", "thesis", "toefl", "trade", "trading", "transcript", "transfer", "tutor", "type", "typing", "undergrad", "underpaid", "unemploy", "universit", "unproduc", "upperclass", "varsit", "vita", "vitas", "vocation", "vp", "wage", "wages", "warehous", "welfare", "work ", "workabl", "worked", "worker", "working", "works", "xerox"],

    'Achiev' : ["abilit", "able", "accomplish", "ace", "achiev", "acquir", "acquisition", "adequa", "advanc", "advantag", "ahead", "ambiti", "approv", "attain", "attempt", "authorit", "award", "beat", "beaten", "best", "better", "bonus", "burnout", "capab", "celebrat", "challeng", "champ", "climb", "closure", "compet", "conclud", "conclus", "confidence", "confident", "confidently", "conquer", "conscientious", "control", "create", "creati", "crown", "defeat", "determina", "determined", "diligen", "domina", "domote", "driven", "dropout", "earn", "effect", "efficien", "effort", "elit", "enabl", "endeav", "excel", "fail", "finaliz", "first", "firsts", "founded", "founder", "founding", "fulfill", "gain", "goal", "hero", "honor", "honour", "ideal", "importan", "improve", "improving", "inadequa", "incapab", "incentive", "incompeten", "ineffect", "initiat", "irresponsible", "king", "lazie", "lazy", "lead", "lesson", "limit", "lose", "loser", "loses", "losing", "loss", "lost", "master", "mastered", "masterful", "mastering", "mastermind", "masters", "mastery", "medal", "mediocr", "motiv", "obtain", "opportun", "organiz", "originat", "outcome", "overcome", "overconfiden", "overtak", "perfect", "perform", "persever", "persist", "plan", "planned", "planner", "planning", "plans", "potential", "power", "practice", "prais", "presiden", "pride", "prize", "produc", "proficien", "progress", "promot", "proud", "purpose", "queen", "queenly", "quit", "quitt", "rank", "ranked", "ranking", "ranks", "recover", "requir", "resolv", "resourceful", "responsib", "reward", "skill", "skilled", "skills", "solution", "solve", "solved", "solves", "solving", "strateg", "strength", "striv", "strong", "succeed", "success", "super", "superb", "surviv", "team", "top", "tried", "tries", "triumph", "try", "trying", "unable", "unbeat", "unproduc", "unsuccessful", "victor", "win", "winn", "wins", "won", "work ", "workabl", "worked", "worker", "working", "works"],


    'Leisure' : ["actor", "actress", "aerobic", "amus", "apartment", "art", "artist", "arts", "athletic", "ball", "ballet", "band", "bands", "bar", "bars", "baseball", "basketball", "bath", "beach", "beer", "bicyc", "bike", "birdie", "blackjack", "blockbuster", "blog", "book", "café", "camping", "cards", "casino", "casual", "cd", "celebrat", "celebrit", "channel", "chat", "checkers", "chess", "chillin", "choir", "chorus", "cinema", "club", "coach", "cocktail", "coffee", "comed", "comic", "concert", "cook", "cruis", "danc", "decorat", "diaries", "diary", "dj", "drama", "dream", "drink", "drum", "drumm", "drums", "drunk", "dvd", "easy", "entertain", "exercis", "families", "family", "fantasi", "fantasy", "film", "fishing", "fitness", "flatscreen", "football", "frisbee", "game", "gaming", "garage", "garden", "golf", "guitar", "gym", "hangin", "hangout", "hik", "hiphop", "hobb", "hockey", "holiday", "horseback", "hotel", "hunting", "inn", "inns", "intramural", "ipod", "jazz", "jog", "joke", "joking", "karaoke", "keg", "kegger", "keggers", "kegs", "laidback", "liquor", "magazine", "mall", "malls", "marathon", "margarita", "martini", "meditat", "minesweeper", "motel", "movie", "mtv", "museum", "musi", "nap", "naps", "netflix", "nintendo", "novel", "novels", "opera", "orchestra", "parks", "partie", "party", "pitcher", "play", "played", "playful", "playing", "plays", "playstation", "poetry", "poker", "pool", "pretend", "pub", "pubs", "radio", "rap", "read ", "reading", "recording", "recreation", "reggae", "relax", "restau", "resting", "rock", "rocks", "roller", "rowing", "rugby", "rum", "runner", "running", "salsa", "sang", "scrabble", "scrapbook", "sculpt", "shop", "shopaholic", "shopp", "shops", "show", "shows", "sing", "singing", "sings", "skat", "ski", "skied", "skier", "skiing", "skis", "soaps", "soccer", "solitaire", "song", "spa", "spas", "sport", "standup", "stereo", "stereos", "stoned", "sunbath", "swim", "symphon", "tanning", "team", "techno", "television", "tennis", "tequila", "theat", "ticket", "tivo", "toy", "travel", "triathl", "tv", "unwind", "vacation", "vcr", "video", "vodka", "volleyb", "weekend", "weightlift", "weights", "whiskey", "whisky", "wine", "wines", "workout", "xbox", "yard", "yoga"],

    'Home' : ["address", "apartment", "backyard", "bake", "baking", "balcon", "bath", "bed", "bedding", "bedroom", "beds", "blender", "broom", "carpet", "chore", "clean", "closet", "closets", "condo", "condominium", "condos", "couch", "curtain", "den", "dishwasher", "doghouse", "domestic", "door", "dorm", "drape", "dresser", "driveway", "duplex", "families", "family", "fireplace", "fridge", "furniture", "futon", "garage", "garden", "gate", "home", "homes", "homesick", "homework", "house", "housing", "kitchen", "landlord", "lawn", "lease", "leasing", "loft", "lofts", "loveseat", "maid", "mailbox", "mattress", "microwave", "mop", "mortg", "neighbor", "oven", "patio", "pet", "pets", "pillow", "porch", "remodel", "renovat", "rent", "residen", "room", "roomate", "roomed", "roomie", "rooming", "roommate", "rooms", "rug", "rugs", "shower", "sofa", "stove", "studio", "studios", "sweep", "tenant", "toaster", "vacuum", "window", "yard"],


    'Money' : ["account", "atm", "atms", "auction", "audit", "audited", "auditing", "auditor", "auditors", "audits", "bank", "bargain", "beggar", "begging", "bet", "bets", "betting", "bill", "billed", "billing", "bills", "bonus", "borrow", "bought", "broker", "buck", "bucks", "budget", "business", "buy", "cash", "casino", "cent", "cents", "charit", "cheap", "check", "checking", "checks", "chequ", "coin", "coins", "compensat", "consumer", "corporat", "cost", "coupon", "credit", "currenc", "customer", "debit", "debt", "deposit", "dime", "dinar", "dinero", "discount", "dividend", "dollar", "donat", "econ", "embezzl", "euro", "euros", "exchang", "expens", "fee", "fees", "financ", "fortune", "franc", "franchis", "francs", "free", "freeb", "fund", "funded", "funding", "funds", "gambl", "greed", "income", "inexpens", "inherit", "insurance", "invest", "IRS", "jackpot", "kopek", "kron", "lease", "leasing", "lira", "loan", "lotter", "mastercard", "merchant", "money", "monopol", "mortg", "nickel", "overpaid", "overtime", "owe ", "owed", "owes", "owing", "paid", "pay", "pence", "pennies", "penny", "peso", "pesos", "poor", "portfolio", "poverty", "price", "prici", "profit", "purchas", "rebate", "recession", "refund", "reimburs", "rent", "retail", "revenue", "rich", "ruble", "rupee", "salar", "sale", "sales", "saving", "scholarship", "sell", "seller", "selling", "sells", "shilling", "shop", "shopaholic", "shopp", "shops", "sold", "spend", "spender", "spending", "spends", "spent", "stipend", "stocks", "store", "tax", "taxa", "taxed", "taxes", "taxing", "thrift", "trade", "trading", "tuition", "underpaid", "value", "visa", "wage", "wager", "wages", "wealth", "worth", "yen", "yuan"],


    'Relig' : ["afterlife", "agnost", "alla", "allah", "altar", "amen", "amish", "angel", "angelic", "angels", "baptis", "baptiz", "belief", "bible", "biblic", "bishop", "bless", "buddh", "catholic", "chapel", "chaplain", "christ", "christian", "christmas", "church", "clergy", "confess", "convent", "convents", "crucifi", "crusade", "demon", "devil", "divin", "doom", "episcopal", "evangel", "faith", "fundamentalis", "gentile", "god'", "gospel", "heaven", "hell", "hellish", "hells", "hindu", "holie", "holy", "hymn", "immoral", "immortal", "islam", "jesuit", "jesus", "jew", "jewish", "jews", "jihad", "juda", "karma", "kippur", "koran ", "kosher", "krishna", "krisna", "lord", "lutheran", "mecca", "meditat", "mercif", "mercy", "methodis", "minister", "ministr", "missionar", "mitzvah", "mohamm", "monast", "monk", "moral", "morality", "morals", "mormon", "mosque", "muhamm", "mujahid", "musl", "nun", "nuns", "orthodox", "pagan", "paradise", "passover", "pastor", "penance", "pentecost", "pew", "pews", "piety", "pilgrim", "pious", "pope", "prais", "pray", "preach", "presbyterian", "priest", "prophe", "protestant", "psalm", "purgator", "puritan", "quran", "qur'an", "rabbi", "rabbinical", "rabbis", "ramadan", "religio", "rite", "rites", "ritual", "rosaries", "rosary", "roshashan", "sabbath", "sacred", "sacrific", "saint", "salvation", "satan", "scriptur", "sect", "sectarian", "sects", "seminar", "shi'", "shiite", "shrine", "sikh", "sin", "sinn", "sins", "soul", "souls", "spirit", "sunni", "sunnis", "temple", "testament", "theolog", "torah", "vatican", "veil", "worship", "yiddish", "zen", "zion"],


    'Death' : ["autops", "alive", "bereave", "burial", "buried", "bury", "casket", "casualt", "cemet", "coffin", "coroner", "corpse", "cremat", "crypt", "dead", "death", "decease", "demise", "die", "died", "dies", "DOA", "drown", "dying", "embalm", "epidemic", "execution", "exterminat", "fatal", "funer", "genocid", "ghost", "grave", "grief", "griev", "hearse", "holocaust", "homocid", "immortal", "kill", "lethal", "lynch", "manslaughter", "massacre", "mausoleum", "morgue", "mortal", "mortician", "mourn", "murder", "obit", "od", "oded", "overdosed", "pallbearer", "plague", "reaper", "slaughter", "suicid", "tomb", "urn", "war"],

    'Assent' : ["absolutely", "agree", "ah", "alright", "aok", "aw", "awesome", "cool", "duh", "ha", "hah", "haha", "heh", "hm", "huh", "lol", "mm", "ok", "okay", "okey", "rofl", "uhhu", "uhuh", "yah", "yay", "yea", "yeah", "yep", "yes", "yup"],

    'Nonflu' : ["er", "hm", "sigh", "uh", "um", "umm", "well", "zz"],

    'Filler' : ["blah", "idontknow", "imean", "like", "ohwell", "rr", "yakno", "ykn", "youknow"]

        }


class LIWCFeature(BaseEstimator,TransformerMixin):
    """
    Emotion Feature estimator
    """
    def __init__(self, emo=None):
        if not emo:
            self.emotion_dict = emotions
        else:
            self.emotion_dict = emo


    def get_feature_names(self):
        return np.array(["Feature_Funct_emo", "Feature_Pronoun_emo", "Feature_Ppron_emo", "Feature_I_emo", "Feature_We_emo", "Feature_You_emo",
                         "Feature_SheHe_emo", "Feature_They_emo", "Feature_Ipron_emo", "Feature_Article_emo", "Feature_Verbs_emo", "Feature_AuxVb_emo",
                         "Feature_Past_emo", "Feature_Present_emo", "Feature_Future_emo", "Feature_Adverbs_emo", "Feature_Prep_emo", "Feature_Conj_emo",
                         "Feature_Negate_emo", "Feature_Quant_emo", "Feature_Numbers_emo", "Feature_Swear_emo", "Feature_Social_emo", "Feature_Family_emo",
                         "Feature_Friends_emo", "Feature_Humans_emo", "Feature_Affect_emo", "Feature_Posemo_emo", "Feature_Negemo_emo", "Feature_Anx_emo",
                         "Feature_Anger_emo", "Feature_Sad_emo", "Feature_CogMech_emo", "Feature_Insight_emo", "Feature_Cause_emo", "Feature_Discrep_emo",
                         "Feature_Tentat_emo", "Feature_Certain_emo", "Feature_Inhib_emo", "Feature_Incl_emo", "Feature_Excl_emo", "Feature_Percept_emo",
                         "Feature_See_emo", "Feature_Hear_emo", "Feature_Feel_emo", "Feature_Bio_emo", "Feature_Body_emo", "Feature_Health_emo",
                         "Feature_Sexual_emo", "Feature_Ingest_emo", "Feature_Relativ_emo", "Feature_Motion_emo", "Feature_Space_emo", "Feature_Time_emo",
                         "Feature_Work_emo", "Feature_Achiev_emo", "Feature_Leisure_emo", "Feature_Home_emo", "Feature_Money_emo", "Feature_Relig_emo",
                         "Feature_Death_emo", "Feature_Assent_emo", "Feature_Nonflu_emo", "Feature_Filler_emo"])
        #return np.array(["Feature_death_emo", "Feature_sexual_emo", "Feature_anger_emo" , "Feature_sad_emo"])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        fea_Funct_emo, fea_Pronoun_emo, fea_Ppron_emo, fea_I_emo, fea_We_emo, fea_You_emo, fea_SheHe_emo, fea_They_emo, fea_Ipron_emo, fea_Article_emo, fea_Verbs_emo, fea_AuxVb_emo, fea_Past_emo, fea_Present_emo, fea_Future_emo, fea_Adverbs_emo, fea_Prep_emo, fea_Conj_emo, fea_Negate_emo, fea_Quant_emo, fea_Numbers_emo, fea_Swear_emo, fea_Social_emo, fea_Family_emo, fea_Friends_emo, fea_Humans_emo, fea_Affect_emo, fea_Posemo_emo, fea_Negemo_emo, fea_Anx_emo, fea_Anger_emo, fea_Sad_emo, fea_CogMech_emo, fea_Insight_emo, fea_Cause_emo, fea_Discrep_emo, fea_Tentat_emo, fea_Certain_emo, fea_Inhib_emo, fea_Incl_emo, fea_Excl_emo, fea_Percept_emo, fea_See_emo, fea_Hear_emo, fea_Feel_emo, fea_Bio_emo, fea_Body_emo, fea_Health_emo, fea_Sexual_emo, fea_Ingest_emo, fea_Relativ_emo, fea_Motion_emo, fea_Space_emo, fea_Time_emo, fea_Work_emo, fea_Achiev_emo, fea_Leisure_emo, fea_Home_emo, fea_Money_emo, fea_Relig_emo, fea_Death_emo, fea_Assent_emo, fea_Nonflu_emo, fea_Filler_emo = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        for doc in documents:
            #print("here 1")
            Funct_emo = 0
            Pronoun_emo = 0
            Ppron_emo = 0
            I_emo = 0
            We_emo = 0
            You_emo = 0
            SheHe_emo = 0
            They_emo = 0
            Ipron_emo = 0
            Article_emo = 0
            Verbs_emo = 0
            AuxVb_emo = 0
            Past_emo = 0
            Present_emo = 0
            Future_emo = 0
            Adverbs_emo = 0
            Prep_emo = 0
            Conj_emo = 0
            Negate_emo = 0
            Quant_emo = 0
            Numbers_emo = 0
            Swear_emo = 0
            Social_emo = 0
            Family_emo = 0
            Friends_emo = 0
            Humans_emo = 0
            Affect_emo = 0
            Posemo_emo = 0
            Negemo_emo = 0
            Anx_emo = 0
            Anger_emo = 0
            Sad_emo = 0
            CogMech_emo = 0
            Insight_emo = 0
            Cause_emo = 0
            Discrep_emo = 0
            Tentat_emo = 0
            Certain_emo = 0
            Inhib_emo = 0
            Incl_emo = 0
            Excl_emo = 0
            Percept_emo = 0
            See_emo = 0
            Hear_emo = 0
            Feel_emo = 0
            Bio_emo = 0
            Body_emo = 0
            Health_emo = 0
            Sexual_emo = 0
            Ingest_emo = 0
            Relativ_emo = 0
            Motion_emo = 0
            Space_emo = 0
            Time_emo = 0
            Work_emo = 0
            Achiev_emo = 0
            Leisure_emo = 0
            Home_emo = 0
            Money_emo = 0
            Relig_emo = 0
            Death_emo = 0
            Assent_emo = 0
            Nonflu_emo = 0
            Filler_emo = 0
            #total = 0
            #print("here 2")
            tokens = ark_tweet_tokenizer(doc.content)
            #print("here 3")
            if len(tokens)>0:
                #print("here 4")
            #for token in tokens:
            #    flag = False
            #    if token.lower() in self.emotion_dict['death']:
            #        death_emo += 1
            #        flag = True
            #    if token.lower() in self.emotion_dict['sexual']:
            #        sexual_emo += 1
            #        flag = True
            #    if token.lower() in self.emotion_dict['anger']:
            #        anger_emo += 1
            #        flag = True
            #    if token.lower() in self.emotion_dict['sad']:
            #        sad_emo += 1
            #        flag = True
            #   if flag == True:
            #        total += 1
                i = 0
                for word in self.emotion_dict["Funct"]:
                    if word in doc.content.lower():
                        Funct_emo += 1

                for word in self.emotion_dict["Pronoun"]:
                    if word in doc.content.lower():
                        Pronoun_emo += 1


                for word in self.emotion_dict["Ppron"]:
                  if word in doc.content.lower():
                    Ppron_emo += 1

                for word in self.emotion_dict["I"]:
                  if word in doc.content.lower():
                    I_emo += 1

                for word in self.emotion_dict["We"]:
                  if word in doc.content.lower():
                    We_emo += 1

                for word in self.emotion_dict["You"]:
                  if word in doc.content.lower():
                    You_emo += 1

                for word in self.emotion_dict["SheHe"]:
                  if word in doc.content.lower():
                    SheHe_emo += 1

                for word in self.emotion_dict["They"]:
                  if word in doc.content.lower():
                    They_emo += 1

                for word in self.emotion_dict["Ipron"]:
                  if word in doc.content.lower():
                    Ipron_emo += 1

                for word in self.emotion_dict["Article"]:
                  if word in doc.content.lower():
                    Article_emo += 1

                for word in self.emotion_dict["Verbs"]:
                  if word in doc.content.lower():
                    Verbs_emo += 1

                for word in self.emotion_dict["AuxVb"]:
                  if word in doc.content.lower():
                    AuxVb_emo += 1

                for word in self.emotion_dict["Past"]:
                  if word in doc.content.lower():
                    Past_emo += 1

                for word in self.emotion_dict["Present"]:
                  if word in doc.content.lower():
                    Present_emo += 1

                for word in self.emotion_dict["Future"]:
                  if word in doc.content.lower():
                    Future_emo += 1

                for word in self.emotion_dict["Adverbs"]:
                  if word in doc.content.lower():
                    Adverbs_emo += 1

                for word in self.emotion_dict["Prep"]:
                  if word in doc.content.lower():
                    Prep_emo += 1

                for word in self.emotion_dict["Conj"]:
                  if word in doc.content.lower():
                    Conj_emo += 1

                for word in self.emotion_dict["Negate"]:
                  if word in doc.content.lower():
                    Negate_emo += 1

                for word in self.emotion_dict["Quant"]:
                  if word in doc.content.lower():
                    Quant_emo += 1

                for word in self.emotion_dict["Numbers"]:
                  if word in doc.content.lower():
                    Numbers_emo += 1

                for word in self.emotion_dict["Swear"]:
                  if word in doc.content.lower():
                    Swear_emo += 1

                for word in self.emotion_dict["Social"]:
                  if word in doc.content.lower():
                    Social_emo += 1

                for word in self.emotion_dict["Family"]:
                  if word in doc.content.lower():
                    Family_emo += 1

                for word in self.emotion_dict["Friends"]:
                  if word in doc.content.lower():
                    Friends_emo += 1

                for word in self.emotion_dict["Humans"]:
                  if word in doc.content.lower():
                    Humans_emo += 1

                for word in self.emotion_dict["Affect"]:
                  if word in doc.content.lower():
                    Affect_emo += 1

                for word in self.emotion_dict["Posemo"]:
                  if word in doc.content.lower():
                    Posemo_emo += 1

                for word in self.emotion_dict["Negemo"]:
                  if word in doc.content.lower():
                    Negemo_emo += 1

                for word in self.emotion_dict["Anx"]:
                  if word in doc.content.lower():
                    Anx_emo += 1

                for word in self.emotion_dict["Anger"]:
                  if word in doc.content.lower():
                    Anger_emo += 1

                for word in self.emotion_dict["Sad"]:
                  if word in doc.content.lower():
                    Sad_emo += 1

                for word in self.emotion_dict["CogMech"]:
                  if word in doc.content.lower():
                    CogMech_emo += 1

                for word in self.emotion_dict["Insight"]:
                  if word in doc.content.lower():
                    Insight_emo += 1

                for word in self.emotion_dict["Cause"]:
                  if word in doc.content.lower():
                    Cause_emo += 1

                for word in self.emotion_dict["Discrep"]:
                  if word in doc.content.lower():
                    Discrep_emo += 1

                for word in self.emotion_dict["Tentat"]:
                  if word in doc.content.lower():
                    Tentat_emo += 1

                for word in self.emotion_dict["Certain"]:
                  if word in doc.content.lower():
                    Certain_emo += 1

                for word in self.emotion_dict["Inhib"]:
                  if word in doc.content.lower():
                    Inhib_emo += 1

                for word in self.emotion_dict["Incl"]:
                  if word in doc.content.lower():
                    Incl_emo += 1

                for word in self.emotion_dict["Excl"]:
                  if word in doc.content.lower():
                    Excl_emo += 1

                for word in self.emotion_dict["Percept"]:
                  if word in doc.content.lower():
                    Percept_emo += 1

                for word in self.emotion_dict["See"]:
                  if word in doc.content.lower():
                    See_emo += 1

                for word in self.emotion_dict["Hear"]:
                  if word in doc.content.lower():
                    Hear_emo += 1

                for word in self.emotion_dict["Feel"]:
                  if word in doc.content.lower():
                    Feel_emo += 1

                for word in self.emotion_dict["Bio"]:
                  if word in doc.content.lower():
                    Bio_emo += 1

                for word in self.emotion_dict["Body"]:
                  if word in doc.content.lower():
                    Body_emo += 1

                for word in self.emotion_dict["Health"]:
                  if word in doc.content.lower():
                    Health_emo += 1

                for word in self.emotion_dict["Sexual"]:
                  if word in doc.content.lower():
                    Sexual_emo += 1

                for word in self.emotion_dict["Ingest"]:
                  if word in doc.content.lower():
                    Ingest_emo += 1

                for word in self.emotion_dict["Relativ"]:
                  if word in doc.content.lower():
                    Relativ_emo += 1

                for word in self.emotion_dict["Motion"]:
                  if word in doc.content.lower():
                    Motion_emo += 1

                for word in self.emotion_dict["Space"]:
                  if word in doc.content.lower():
                    Space_emo += 1

                for word in self.emotion_dict["Time"]:
                  if word in doc.content.lower():
                    Time_emo += 1

                for word in self.emotion_dict["Work"]:
                  if word in doc.content.lower():
                    Work_emo += 1

                for word in self.emotion_dict["Achiev"]:
                  if word in doc.content.lower():
                    Achiev_emo += 1

                for word in self.emotion_dict["Leisure"]:
                  if word in doc.content.lower():
                    Leisure_emo += 1

                for word in self.emotion_dict["Home"]:
                  if word in doc.content.lower():
                    Home_emo += 1

                for word in self.emotion_dict["Money"]:
                  if word in doc.content.lower():
                    Money_emo += 1

                for word in self.emotion_dict["Relig"]:
                  if word in doc.content.lower():
                    Relig_emo += 1

                for word in self.emotion_dict["Death"]:
                  if word in doc.content.lower():
                    Death_emo += 1

                for word in self.emotion_dict["Assent"]:
                  if word in doc.content.lower():
                    Assent_emo += 1

                for word in self.emotion_dict["Nonflu"]:
                  if word in doc.content.lower():
                    Nonflu_emo += 1

                for word in self.emotion_dict["Filler"]:
                  if word in doc.content.lower():
                    Filler_emo += 1

                #print("here 5")


                fea_Funct_emo.append(round(Funct_emo/len(tokens), 2))
                fea_Pronoun_emo.append(round(Pronoun_emo/len(tokens), 2))
                fea_Ppron_emo.append(round(Ppron_emo/len(tokens), 2))
                fea_I_emo.append(round(I_emo/len(tokens), 2))
                fea_We_emo.append(round(We_emo/len(tokens), 2))
                fea_You_emo.append(round(You_emo/len(tokens), 2))
                fea_SheHe_emo.append(round(SheHe_emo/len(tokens), 2))
                fea_They_emo.append(round(They_emo/len(tokens), 2))
                fea_Ipron_emo.append(round(Ipron_emo/len(tokens), 2))
                fea_Article_emo.append(round(Article_emo/len(tokens), 2))
                fea_Verbs_emo.append(round(Verbs_emo/len(tokens), 2))
                fea_AuxVb_emo.append(round(AuxVb_emo/len(tokens), 2))
                fea_Past_emo.append(round(Past_emo/len(tokens), 2))
                fea_Present_emo.append(round(Present_emo/len(tokens), 2))
                fea_Future_emo.append(round(Future_emo/len(tokens), 2))
                fea_Adverbs_emo.append(round(Adverbs_emo/len(tokens), 2))
                fea_Prep_emo.append(round(Prep_emo/len(tokens), 2))
                fea_Conj_emo.append(round(Conj_emo/len(tokens), 2))
                fea_Negate_emo.append(round(Negate_emo/len(tokens), 2))
                fea_Quant_emo.append(round(Quant_emo/len(tokens), 2))
                fea_Numbers_emo.append(round(Numbers_emo/len(tokens), 2))
                fea_Swear_emo.append(round(Swear_emo/len(tokens), 2))
                fea_Social_emo.append(round(Social_emo/len(tokens), 2))
                fea_Family_emo.append(round(Family_emo/len(tokens), 2))
                fea_Friends_emo.append(round(Friends_emo/len(tokens), 2))
                fea_Humans_emo.append(round(Humans_emo/len(tokens), 2))
                fea_Affect_emo.append(round(Affect_emo/len(tokens), 2))
                fea_Posemo_emo.append(round(Posemo_emo/len(tokens), 2))
                fea_Negemo_emo.append(round(Negemo_emo/len(tokens), 2))
                fea_Anx_emo.append(round(Anx_emo/len(tokens), 2))
                fea_Anger_emo.append(round(Anger_emo/len(tokens), 2))
                fea_Sad_emo.append(round(Sad_emo/len(tokens), 2))
                fea_CogMech_emo.append(round(CogMech_emo/len(tokens), 2))
                fea_Insight_emo.append(round(Insight_emo/len(tokens), 2))
                fea_Cause_emo.append(round(Cause_emo/len(tokens), 2))
                fea_Discrep_emo.append(round(Discrep_emo/len(tokens), 2))
                fea_Tentat_emo.append(round(Tentat_emo/len(tokens), 2))
                fea_Certain_emo.append(round(Certain_emo/len(tokens), 2))
                fea_Inhib_emo.append(round(Inhib_emo/len(tokens), 2))
                fea_Incl_emo.append(round(Incl_emo/len(tokens), 2))
                fea_Excl_emo.append(round(Excl_emo/len(tokens), 2))
                fea_Percept_emo.append(round(Percept_emo/len(tokens), 2))
                fea_See_emo.append(round(See_emo/len(tokens), 2))
                fea_Hear_emo.append(round(Hear_emo/len(tokens), 2))
                fea_Feel_emo.append(round(Feel_emo/len(tokens), 2))
                fea_Bio_emo.append(round(Bio_emo/len(tokens), 2))
                fea_Body_emo.append(round(Body_emo/len(tokens), 2))
                fea_Health_emo.append(round(Health_emo/len(tokens), 2))
                fea_Sexual_emo.append(round(Sexual_emo/len(tokens), 2))
                fea_Ingest_emo.append(round(Ingest_emo/len(tokens), 2))
                fea_Relativ_emo.append(round(Relativ_emo/len(tokens), 2))
                fea_Motion_emo.append(round(Motion_emo/len(tokens), 2))
                fea_Space_emo.append(round(Space_emo/len(tokens), 2))
                fea_Time_emo.append(round(Time_emo/len(tokens), 2))
                fea_Work_emo.append(round(Work_emo/len(tokens), 2))
                fea_Achiev_emo.append(round(Achiev_emo/len(tokens), 2))
                fea_Leisure_emo.append(round(Leisure_emo/len(tokens), 2))
                fea_Home_emo.append(round(Home_emo/len(tokens), 2))
                fea_Money_emo.append(round(Money_emo/len(tokens), 2))
                fea_Relig_emo.append(round(Relig_emo/len(tokens), 2))
                fea_Death_emo.append(round(Death_emo/len(tokens), 2))
                fea_Assent_emo.append(round(Assent_emo/len(tokens), 2))
                fea_Nonflu_emo.append(round(Nonflu_emo/len(tokens), 2))
                fea_Filler_emo.append(round(Filler_emo/len(tokens), 2))
               # print("here 6")

            else:
                #print("here 7")
                fea_Funct_emo.append(0)
                fea_Pronoun_emo.append(0)
                fea_Ppron_emo.append(0)
                fea_I_emo.append(0)
                fea_We_emo.append(0)
                fea_You_emo.append(0)
                fea_SheHe_emo.append(0)
                fea_They_emo.append(0)
                fea_Ipron_emo.append(0)
                fea_Article_emo.append(0)
                fea_Verbs_emo.append(0)
                fea_AuxVb_emo.append(0)
                fea_Past_emo.append(0)
                fea_Present_emo.append(0)
                fea_Future_emo.append(0)
                fea_Adverbs_emo.append(0)
                fea_Prep_emo.append(0)
                fea_Conj_emo.append(0)
                fea_Negate_emo.append(0)
                fea_Quant_emo.append(0)
                fea_Numbers_emo.append(0)
                fea_Swear_emo.append(0)
                fea_Social_emo.append(0)
                fea_Family_emo.append(0)
                fea_Friends_emo.append(0)
                fea_Humans_emo.append(0)
                fea_Affect_emo.append(0)
                fea_Posemo_emo.append(0)
                fea_Negemo_emo.append(0)
                fea_Anx_emo.append(0)
                fea_Anger_emo.append(0)
                fea_Sad_emo.append(0)
                fea_CogMech_emo.append(0)
                fea_Insight_emo.append(0)
                fea_Cause_emo.append(0)
                fea_Discrep_emo.append(0)
                fea_Tentat_emo.append(0)
                fea_Certain_emo.append(0)
                fea_Inhib_emo.append(0)
                fea_Incl_emo.append(0)
                fea_Excl_emo.append(0)
                fea_Percept_emo.append(0)
                fea_See_emo.append(0)
                fea_Hear_emo.append(0)
                fea_Feel_emo.append(0)
                fea_Bio_emo.append(0)
                fea_Body_emo.append(0)
                fea_Health_emo.append(0)
                fea_Sexual_emo.append(0)
                fea_Ingest_emo.append(0)
                fea_Relativ_emo.append(0)
                fea_Motion_emo.append(0)
                fea_Space_emo.append(0)
                fea_Time_emo.append(0)
                fea_Work_emo.append(0)
                fea_Achiev_emo.append(0)
                fea_Leisure_emo.append(0)
                fea_Home_emo.append(0)
                fea_Money_emo.append(0)
                fea_Relig_emo.append(0)
                fea_Death_emo.append(0)
                fea_Assent_emo.append(0)
                fea_Nonflu_emo.append(0)
                fea_Filler_emo.append(0)

        X = np.array([fea_Funct_emo, fea_Pronoun_emo, fea_Ppron_emo, fea_I_emo, fea_We_emo, fea_You_emo, fea_SheHe_emo, fea_They_emo, fea_Ipron_emo, fea_Article_emo, fea_Verbs_emo, fea_AuxVb_emo, fea_Past_emo, fea_Present_emo, fea_Future_emo, fea_Adverbs_emo, fea_Prep_emo, fea_Conj_emo, fea_Negate_emo, fea_Quant_emo, fea_Numbers_emo, fea_Swear_emo, fea_Social_emo, fea_Family_emo, fea_Friends_emo, fea_Humans_emo, fea_Affect_emo, fea_Posemo_emo, fea_Negemo_emo, fea_Anx_emo, fea_Anger_emo, fea_Sad_emo, fea_CogMech_emo, fea_Insight_emo, fea_Cause_emo, fea_Discrep_emo, fea_Tentat_emo, fea_Certain_emo, fea_Inhib_emo, fea_Incl_emo, fea_Excl_emo, fea_Percept_emo, fea_See_emo, fea_Hear_emo, fea_Feel_emo, fea_Bio_emo, fea_Body_emo, fea_Health_emo, fea_Sexual_emo, fea_Ingest_emo, fea_Relativ_emo, fea_Motion_emo, fea_Space_emo, fea_Time_emo, fea_Work_emo, fea_Achiev_emo, fea_Leisure_emo, fea_Home_emo, fea_Money_emo, fea_Relig_emo, fea_Death_emo, fea_Assent_emo, fea_Nonflu_emo, fea_Filler_emo]).T
        return X
