from collections import defaultdict
from dijkstar import Graph, find_path
import networkx as nx
import math
import matplotlib.pyplot as plt
import sys
import random

class OptimalPolicy:
    def __init__(self):
        pass

    #words list for single (possibly partial) sentence

    def translate(self, words):
        return words.split()

    def score_translation(self, translation_words):
        return 0

    def new_state(self, old_state, action):
        old_words = old_state.old_wordss
        untranslated_words = old_state.untranslated_words
        old_score = old_state.cumulative_score

        if action == 'WAIT':
            return State(list(old_words),
                         untranslated_words + new_word,
                         [],
                         self.score_translation(self.translate(untranslated_words)))
        elif action == 'COMMIT':
            return State(list(old_words),
                         [],
                         untranslated_words + new_word,
                         self.score_translation(self.translate(untranslated_words + new_word)))

    def score_sublists(list1): 
        sublist = [[]] 
        for i in range(len(list1) + 1): 
            for j in range(i + 1, len(list1) + 1): 
                sub = list1[i:j] 
                sublist.append(sub) 
        return sublist 
  
    def build_trellis(self,
                      sentence_words,
                      possible_actions,
                      max_depth=4,
                      start_index=0,
                      start_state_name=None):
        original_words = sentence_words
        sentence_words = sentence_words[start_index:]
        start_state = None

        if len(sentence_words) == 0:
            return (0,'END')
        if start_state_name is None:
            START_STATE = 'START'
        else:
            START_STATE = start_state_name
        g = nx.DiGraph()
        g.add_node(START_STATE)
        j = len(sentence_words) + 1
        #prev_layer_edges = [0]
        prev_layer = [START_STATE]
        for i in range(min(len(sentence_words), max_depth)):
            #this_layer_edges = []
            this_layer = []
            word = sentence_words[i]
            for action in possible_actions:
                for pe in prev_layer:
                    score = 0
                    if action == 'W':
                        score = 0
                    else:
                        score = 1
                    g.add_edge(pe, action + '_' + str(j), weight=score)
                    #this_layer_edges.append(j)
                    this_layer.append(action + '_' + str(j))
                    j += 1
            #prev_layer_edges = this_layer_edges
            prev_layer = this_layer

        best_path = None
        best_score = -math.inf
        for leaf in prev_layer:
            path = nx.single_source_dijkstra(g, START_STATE, leaf)
            #print(path)
            if path[0] > best_score:
                best_score = path[0]
                best_path = path
        
        #Convert best path to normal list
        print(best_path)
        sys.exit()
        best_path_cost = best_path[0] 
        best_step = best_path[1][1] #check, might need 

        all_best_steps = []

        #sys.exit()
        #print(sentence_words[max_depth-1:])
        all_best_steps += [best_step]
        all_best_steps += self.build_trellis(original_words,
                                             possible_actions,
                                             max_depth,
                                             start_index + 1,
                                             start_state_name=best_step)
        print(all_best_steps)

        return all_best_steps




        
        
    def find_optimal(self,sentence_words, possible_actions, horizon):
        #horizon should be bleu width                 
        subsentences = sublists(sentence_words)
        
        sentence_scores = list()
        temp = list()
        optimal_sequence = []
        sequences = defaultdict(dict)
        #for step in range(sentence_words):
        #find best first commit
        
        for i in range(sentence_words):                         
            word = sentence_words[i]                         
            for action in possible_actions:
                if i == 0:
                    sequences[i][action] = new_state(State([], [], word), action)
                else:
                     if action == 'WAIT':
                         last_words = sentence_words[:-1]
                         sequences[i][action] = new_state(last_words) 
                         #sequences[i][action] = new_state(sequences[i-1])
                                                          

            

            
class State:
    # old_words = []
    # untranslated_words = []
    # to_translate = []
    def __init__(self, old_words,
                 untranslated_words,
                 to_translate,
                 step_score):
        self.old_words = old_words
        self.untranslated_words = untranslated_words
        self.to_translate = to_translate
        #self.cumulative_score = cumulative_score
        self.step_score = step_score                                                


o = OptimalPolicy()
g = o.build_trellis("a b c d e".split(), ["W","C"])
sys.exit()
#cost_func = lambda u, v, e, prev_e: e['cost']
#path = find_path(g, 0, 8, cost_func=cost_func)
#print(path)

pos=nx.spring_layout(g) # positions for all nodes


# labels


from networkx.drawing.nx_agraph import write_dot
from networkx.drawing.nx_agraph import graphviz_layout

#nx.draw_networkx_labels(g,pos,font_size=20,font_family='sans-serif')
write_dot(g,'test.dot')

# same layout using matplotlib with no labels
plt.title('draw_networkx')
pos=graphviz_layout(g, prog='dot')
nx.draw(g, pos, with_labels=False, arrows=False)
plt.savefig('nx_test.png')
#nx.draw(g)
