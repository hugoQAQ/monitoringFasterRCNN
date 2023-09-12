#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abstractions import *
import pickle
import numpy as np

class Monitor(object):

    def __init__(self, good_ref=None):
        # self.abs_type = abs_type
        self.good_ref = good_ref


    def set_reference(self, good_ref):
        self.good_ref = good_ref
    
    # def get_identity(self):
    #     print("Monitor for network:" + self.netName + "class: " + str(self.classification) + "at layer " + str(self.location))


    def make_verdicts(self, features):
        if len(self.good_ref):
            verdicts = ref_query(features, self.good_ref)
        else:
            raise RuntimeError("No reference exists!")
        return verdicts
    
    # def make_verdicts_delta(self, features, delta):
    #     if len(self.good_ref):
    #         verdicts = ref_query_delta(features, self.good_ref, delta)
    #     else:
    #         raise RuntimeError("No reference exists!")
    #     return verdicts
    

def ref_query(features, reference):
    query_results = [boxes_query(x, reference) for x in features]
    return query_results

# def ref_query_delta(features, reference, delta):
#     query_results = [boxes_query_delta(x, reference, delta) for x in features]
#     return query_results


# def query_infusion(in_good_ref, in_bad_ref):
#     if len(in_good_ref) == len(in_bad_ref): #0: acceptance (true, false), 1: rejection (false, true or false), 2: uncertainty (true, true)
#         verdicts = np.zeros(len(in_good_ref), dtype=int)
#         for i in range(len(in_good_ref)):
#             if not in_good_ref[i]:
#                 verdicts[i] = 1
#             elif in_bad_ref[i]:
#                 verdicts[i] = 2
#         return verdicts
#     else:
#         print("Error: IllegalArgument")       
