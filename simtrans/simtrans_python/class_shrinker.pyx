from Cython.Build import cythonize
import pyximport; pyximport.install()
import operator
import codecs
from filereaders import *

"""
This is a class template for taking data and
limiting/redistributing classes/lables  according to some strategy.
It is initialized with a FileReader object, which handles the parsing of the examples.
The default strategy cuts of after n labels.
"""

cdef class ClassShrinker(object):
    def __init__(self):
        self.class_counts = {}
        self.exclusion_tokens = {}
        self.count_threshold = 0
        self.min_percent_threshold = 0.0
        self.max_classes = 2147483647

    """
    Adds an example to the class counts.        
    """        
    cpdef add_example(self, unicode class_name):
        if class_name in self.exclusion_tokens.keys():
            return
        if not class_name in self.class_counts:
            self.class_counts[class_name] = 1
        else:
            self.class_counts[class_name] += 1
        self.total_sum += 1

    cpdef add_examples_from_file_reader(self, reader):
        for line in reader:
            self.add_example(reader.get_class(line))

    """
    Returns the number of occurrences of the specified class.        
    """
    cpdef int get_count(self, unicode class_name):
        if class_name in self.class_counts:
            return self.class_counts[class_name]
        else:
            return 0

    """
    Returns the most common class name
    """
    cpdef unicode most_common_class(self):
        return max(self.class_counts.iteritems(), key=operator.itemgetter(1))[0]

    """
    Returns the least common class count
    """
    cpdef int least_common_class_count(self):
        return min(self.class_counts.iteritems(), key=operator.itemgetter(1))[1]

    """
    Sets the minimum number of instances a class must have
    to be included in calculations.
    """
    cpdef set_min_count_threshold(self, int threshold):
        self.count_threshold = threshold

    """
    Sets the minimum percentage (excluding excluded tokens)
    that a token must compose to be included in calculations.
    This value should be in [0,1]
    """
    cpdef set_min_percentage_threshold(self, float percent):
        self.min_percent_threshold = percent

    """
    Adds a token to be excluded from calculations and output.
    """
    cpdef set_exclude_class(self, unicode _class):
        self.exclusion_tokens[_class] = 1
        if _class in self.class_counts:
            del self.class_counts[_class]

    """
    Sets the maximum number of classes allowed.
    """
    cpdef set_max_classes(self, int _max):
        self.max_classes = _max


    """
    Applies a transformation on the counts in the data.
    """
    cpdef fit_to_distribution(self):
        raise NotImplementedError("Not implemented")

    """
    Returns a sorted list of tuples with classes and their counts.
    """
    cpdef list get_shrunk_class_counts(self):
        return self.sorted_counts

    cpdef list get_shrunk_class_list(self):
        cdef list class_list = []
        cdef tuple current_class
        for current_class in self.sorted_counts:
            class_list.append(current_class[0])
        return class_list

    cpdef list export_class_list(self, out_filename):
        cdef list shrunk_list = self.get_shrunk_class_list()
        of = codecs.open(out_filename, "w", encoding="utf-8")
        for label in shrunk_list:
            of.write(label + u"\n")
        of.close()
        
            
    """
    Applies conditions set to the data that was loaded.
    """
    cpdef recalculate(self):
        #First, delete by maximum allowed classes:
        self.sorted_counts = sorted(self.class_counts.iteritems(), key=operator.itemgetter(1), reverse=True)
        del self.sorted_counts[self.max_classes:]
        #Next, delete by minimum count threshold
        # cdef int index = 0
        # cdef int new_total = 0
        # for item in self.sorted_counts:
        #     if item[1] < self.count_threshold:
        #         break
        #     index += 1
        #     new_total += item[1]
        # del self.sorted_counts[index:]
        # #Finally, delete by percentage of total
        # #not implemented
        # index = 0
        # for item in self.sorted_counts:
        #     #TODO:problem right here.  Fix.
        #     if item[1] / new_total >= self.min_percent_threshold:
        #         index += 1
        #     else:
        #         break
        # del self.sorted_counts[index:]
        #print self.sorted_counts
