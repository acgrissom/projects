import subprocess, sys, errno, pty, os
__author__ = "Alvin Grissom II"
"""
smor_interface.pyx: This is an interface for the SMOR FST
toolkit, using for morphological analysis.
"""

cdef class SMORInterface:
    cdef unicode fst_file
    cdef int out, err
    cdef stdout, stderr
    cdef smor_process
    def __init__(self, unicode fst_file):
        self.fst_file = fst_file
        self.out, self.err = pty.openpty()

    cdef unicode get_raw_info(self, unicode text):
        raise NotImplementedError("When implemented, this class returns the raw output of the SMOR subprocess in response to text input.")

cdef class MorphistoInterface(SMORInterface):
    def __init__(self, unicode fst_file):        
        SMORInterface.__init__(self, fst_file)
        smor_command = u"fst-infl " + fst_file
        self.smor_process = subprocess.Popen(smor_command.split(),
                                              stdin=subprocess.PIPE,
                                              stdout=self.out,
                                              stderr=self.out,
                                              close_fds=True)
        self.stdout = os.fdopen(self.out)
        self.stderr = os.fdopen(self.err)
        if self.smor_process is None:
            sys.stderr.write("Failed to load " + smor_command + "\n")
            sys.exit()
        output = ""
        cdef unicode err_output = u""
        while not err_output.startswith("finished"):
            err_output = unicode(self.stderr.readline())
            #line = self.stdout.readline()
            #line = self.stdout.readline()
            #print "stderr:init",err_output
            #print "stdout:init",line
        while not output.startswith("finished"):
            #print "stdout:init",output
            output = self.stdout.readline()
            #line = self.stdout.readline()
            #line = self.stdout.readline()

        #print "exiting init"

    """
    Gets the raw output of smor.
    You shouldn't change or use this.
    """
    cdef unicode get_raw_info(self, unicode text):
        #I hacked the shit out of this interface
        cdef:
            str output = ""
            unicode err_output = u""
            str line = ""
        #print "writing", text
        #print "flushed stdin"
        #self.smor_process.stdout.flush()
        #self.smor_process.stderr.flush()
        #self.smor_process.stdin.write(u"fake_string\n")
        #self.smor_process.stdin.flush()
                #print "sent fake_string"
        #while not output.startswith("no result for fake_string"):
        #    output = self.stdout.readline()
        output = ""

        self.smor_process.stdin.write(text + u"\nfake_string2\n")
        self.smor_process.stdin.flush()


        while not line.startswith("no result for fake_string2"):
            line = self.stdout.readline()
            print "line",line
            if line.startswith("> fake_string2"):
                self.stdout.readline()
                break
            output += line
        return unicode(output, errors='ignore')

    cpdef list get_morphology_list(self, unicode word):
        raw_output = self.get_raw_info(word)
        cdef list lines = raw_output.split("\r")
        cdef int i = 0
        if lines[0] == "> ":
            del lines[0]
        lines[0] = lines[0][2:]
        if lines[1].startswith("no result"):
            lines = lines[:1]
        cdef int length = len(lines)
        if length > 0:
            for i in xrange(1,length):
                if lines[i].startswith("no result"):
                    del lines[i]
        print lines            
        return lines[:len(lines) - 1]
        
    cpdef list get_morphology_tokens(self, unicode word):
        pass

import re
split_angles = re.compile("<|>")

cdef class MorphToken:
    #cpdef public list morph_components
    cpdef public dict possible_case
    cpdef public list possible_labelings
    def __init__(self, list morph_output):
        cdef unicode tok
        self.possible_labelings = list()
        self.possible_case = dict()
        for tok in morph_output:
            self.possible_labelings.append(split_angles.sub(" ",tok).split())
        cdef list possible_labeling
        cdef unicode attribute
        for possible_labeling in self.possible_labelings:
            for attribute in possible_labeling:
                if attribute == u"Akk":
                    #should probably map to realization
                    self.possible_case[u"ACC"] = u"ACC"
                    break
                elif attribute == u"Nom":
                    self.possible_case[u"NOM"] = u"NOM"
                    break
                elif attribute == u"Dat":
                    self.possible_case[u"DAT"] = u"DAT"
    

