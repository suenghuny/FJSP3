import pandas as pd
import numpy as np

import math

soe = 0.01
code = 2
mode = 'training' #'training' #'test1', 'test2', 'test3'



#

process_time_list = pd.read_csv('scheduling_problems/dataset1_process_time.csv').values.tolist()
ops_name_array = pd.read_csv('scheduling_problems/dataset1_ops_name.csv').values
ops_name_array = np.reshape(ops_name_array, (1, -1))
ops_name_list = ops_name_array[0].tolist()
ops_type_list = pd.read_csv('scheduling_problems/dataset1_ops_type.csv').values.tolist()
alternative_machine_list = pd.read_csv('scheduling_problems/dataset1_alternative_machine.csv').values.tolist()
setup_list = pd.read_csv('scheduling_problems/dataset1_setup_time.csv').values.tolist()
maxlen = len(process_time_list[0])
###
for k in range(len(process_time_list)):
    for t in range(maxlen):
        if math.isnan(process_time_list[k][-1]):
            del process_time_list[k][-1]

maxlen = len(ops_type_list[0])
for k in range(len(ops_type_list)):
    for t in range(maxlen):
        if math.isnan(ops_type_list[k][-1]):
            del ops_type_list[k][-1]


maxlen = len(alternative_machine_list[0])
for k in range(len(alternative_machine_list)):
    for t in range(maxlen):
        try:
            if math.isnan(alternative_machine_list[k][-1]):
                del alternative_machine_list[k][-1]
        except TypeError as TE:pass


num_job_type = len(ops_type_list)

#num_machines = np.max([np.max([print(m) for m in machine_list]) for machine_list in alternative_machine_list])+1
num_machines = np.max([np.max([eval(m) for m in machine_list]) for machine_list in alternative_machine_list])+1


workcenter = list()
for machine_list in alternative_machine_list:
    for m in machine_list:
        if eval(m) not in workcenter:
            workcenter.append(eval(m))
workcenter_name = dict()
for i in range(len(workcenter)):
    for m in workcenter[i]:
        workcenter_name[m] = i
print(workcenter_name)
print(ops_type_list)
#sp1 = [2, 3, 2, 4, 5, 3, 2, 3, 2, 3, 4,2]
#sp1 = [2, 1, 1, 1, 3, 1, 1, 1, 4, 1, 1, 1, 1, 4, 2]
#sp1 = [8, 5, 6, 7]
#sp1 = [5, 6, 3, 3]
sp1 = [7]*12
problems = [sp1]
# tp1 = [7, 2, 13,2, 8, 10, 2, 8,3,4, 1, 2, 4,4,10]
# tp2 = [4, 8, 8, 4,3, 3, 6, 5, 6,5, 3, 12, 4, 4, 5]
# tp3 = [8,12, 6,5, 7,  9,7,15, 10, 7, 8, 8,  5,5,8]
# tp4 = [10, 5, 7, 8, 8, 9,6,7,15, 7, 8, 7, 8, 5, 10]
# tp5 = [14, 4, 15, 8, 8, 11,  14, 22, 4, 15, 5, 18, 10, 12]
# tp6 = [10, 12, 12, 15, 12, 8, 12, 5, 18, 5,  12, 8, 11, 20]
# tp7 = [10, 10, 5, 40, 20, 20, 10, 10, 20, 10, 10, 10, 15, 30, 20]
# tp8 = [15, 25, 10, 15, 10, 20, 10, 15, 10, 30, 10, 10, 10, 40, 10]
# test_problems = [tp1, tp2, tp3, tp4, tp5, tp6, tp7, tp8]
#
# if problem == 1:
#     num_job_type = 12
#     num_machines = 18
#     num_jobs = sum([10,5,10,10,10,10,10,15,5,10,10,5])
#     sp1 = [5]*12
#     problems = [sp1]
#     tp1 = [5, 10, 2, 3, 15, 2, 2, 2, 5, 4, 8, 2]
#     tp2 = [5, 5, 2, 2, 3, 2, 2, 5, 7, 10, 15, 2]
#     tp3 = [8,12, 6,5, 2, 15, 12, 13, 8,  9,15,15]
#     tp4 = [5,2, 15, 7, 12,7,10, 8,  18, 8,17, 11]
#     tp5 = [5, 18, 20, 15, 8, 8, 11, 17, 14, 18, 20, 6]
#     tp6 = [19, 19,  15, 7, 18, 11,6, 14, 20, 13, 10, 8]
#     tp7 = [10, 10, 5, 40, 20, 20, 10, 10, 20, 25, 40, 30]
#     tp8 = [40, 20, 20,10, 10, 18, 17, 5,  20, 35, 25, 20]
#     test_problems = [tp1, tp2, tp3, tp4, tp5, tp6, tp7, tp8]
#
# if problem == 0:
#     #num_job_type = 7
#     num_job_type = 3
#     num_jobs =12
#
#     if mode == 'training':
#         num_machines = 4
#
#     sp1 = [4]*num_job_type
#     problems = [sp1]