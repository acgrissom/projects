import itertools
from dijkstar import Graph, find_path
from collections import defaultdict
def score_sublists(list1): 
    sublist = [[]] 
    for i in range(len(list1) + 1): 
        for j in range(i + 1, len(list1) + 1): 
            sub = list1[i:j] 
            sublist.append(sub) 
    return sublist 

words = 'abcdefghijklmnopqrstuvwxyz'
actions = ['W','C']

class TreeNode:
    prev_node = None
    children = None
    def __init__(self,
                 prev_node,
                 incremental_translation,
):
        self.prev_node = prev_node
        self.children = list()

        
import PyAlgDat    
def build_tree(source_words, possible_actions):
    graph = (len(possible_actions) + 1)**len(source_words)
    for word in sentence_words:
        for action in possible_actions:
            

def add_node(last_node, possible_actions, remaining_words):
    for action in possible_actions:
        add_note
    

for i in itertools.product('WC', repeat=50):
    pass
#print(score_sublists(words))
