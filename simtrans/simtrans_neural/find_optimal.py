from collections import defaultdict
from dijkstar import Graph, find_path
import networkx as nx


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
  
    def build_trellis(self, sentence_words, possible_actions):
        graph = Graph()
        g = nx.DiGraph()
        graph.add_edge(0, 1, {'cost' : 0, 'action' : 'WAIT', 'step' : 0}) # 0 -> 1
        g.add_edge('START', 1, weigh =1)
        j = len(sentence_words) + 1
        prev_layer_edges = [0]
        prev = ['START']
        for i in range(len(sentence_words)):
            this_layer_edges = []
            this = []
            word = sentence_words[i]
            for action in possible_actions:
                #for pe in prev_layer_edges:
                for pe in prev:
                    graph.add_edge(pe, j, {'cost' : 1, 'action' : action, 'step' : i + 1})                   
                    g.add_edge(pe, action + str(i))
                    this_layer_edges.append(j)
                    this.append(prev)
                    j += 1
            prev_layer_edges = this_layer_edges

        return g




        
        
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
g = o.build_trellis("a b c d".split(), ["W","C"])
#cost_func = lambda u, v, e, prev_e: e['cost']
#path = find_path(g, 0, 8, cost_func=cost_func)
#print(path)
nx.draw(g)
