from filereaders import FileReader
cdef class ClassShrinker(object):
    cpdef dict class_counts
    cpdef dict exclusion_tokens
    cdef int count_threshold
    cdef float min_percent_threshold
    cdef list sorted_counts
    cpdef int total_sum
    cpdef int max_classes
    cpdef list get_shrunk_class_counts(self)
    cpdef list get_shrunk_class_list(self)
    cpdef add_example(self, unicode class_name)
    cpdef int get_count(self, unicode class_name)
    cpdef int least_common_class_count(self)
    cpdef add_examples_from_file_reader(self, reader)
    cpdef unicode most_common_class(self)
    cpdef set_min_count_threshold(self, int threshold)
    cpdef set_min_percentage_threshold(self, float percent)
    cpdef set_exclude_class(self, unicode _class)
    cpdef set_max_classes(self, int _max)
    cpdef fit_to_distribution(self)
    cpdef recalculate(self)
    cpdef list export_class_list(self, out_filename)
