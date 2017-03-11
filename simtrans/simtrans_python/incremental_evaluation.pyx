from prediction import VWBinaryVerbPredictor
from filereaders import POSCSVFileReader
import class_shrinker

cdef class IncrementalEvaluator:
    cdef:
        dict vocab_stats
        dict percent_stats
        dict distance_stats
                
        
