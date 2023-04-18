import simpy
import time
from copy import deepcopy
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from scheduling_problems.problem_generator import workcenter, num_machines, num_job_type, soe, process_time_list, ops_name_array, ops_name_list, alternative_machine_list, setup_list, problems, ops_type_list, workcenter_name
import random
import numpy as np


class Operation():
    global num_machines, process_time_list, alternative_machine_list
    def __init__(self, job_type, name, num_m=num_machines):
        self.job_type = job_type

        self.k = name
        self.idx = "{}_{}".format(job_type, name)
        if self.idx[2] != '_':
            a = int(self.idx[0])
            b = int(self.idx[2])
        else:
            a = int(self.idx[0:2])
            b = int(self.idx[3])
        self.ops_type = ops_type_list[a][b]
        #print(a, b)
        self.process_time = process_time_list[a][b]
        self.alternative_machine_list = eval(alternative_machine_list[a][b])


all_ops_list = list()
flatten_all_ops_list = list()
ops_length_list = list()

for j in range(len(process_time_list)):
    job_j_operation_list = list()
    ops_length_list.append(len(process_time_list[j]))
    for k in range(len(process_time_list[j])):
        o_j_k = Operation(j, k)
        job_j_operation_list.append(o_j_k)
        flatten_all_ops_list.append(o_j_k)
    all_ops_list.append(job_j_operation_list)
num_ops = len(flatten_all_ops_list)
avail_ops_by_machine = [[] for _ in range(num_machines)]
for ops in flatten_all_ops_list:
    for m in ops.alternative_machine_list:
        avail_ops_by_machine[m].append(ops.idx)
#print(avail_ops_by_machine[0])
# print("operation 개수", len(flatten_all_ops_list))
total_num_ops = sum(ops_length_list)
num_jobs = len(process_time_list)
max_ops_length = max(ops_length_list)
# Job 객체는 Operation 정보 등을 기록함

class Job():
    global all_ops_list, setup_list
    def __init__(self, job_type, name):
        self.job_type = job_type
        self.name = name
        self.operations = deepcopy(all_ops_list[job_type])
        self.current_working_operation = False
        self.apply_action_indicator = 1
        self.completed_operations = list()
        self.process_time = 0
        self.setup_time = 0
        self.ptsu = 0
        self.node_feature = np.concatenate([np.eye(num_jobs)[self.job_type], np.eye(max_ops_length)[self.operations[0].k]])

    def operation_start(self):
        self.current_working_operation = self.operations[0]

    def operation_complete(self):
        self.completed_operations.append(self.operations[0])
        self.current_working_operation = False
        del self.operations[0]
        if len(self.operations)>0:
            self.node_feature = np.concatenate([np.eye(num_jobs)[self.job_type],
                                               np.eye(max_ops_length)[self.operations[0].k]])


# Machine 객체이고 idle, setup, process 시간 등을 기록함
class Machine(simpy.Resource):
    global all_ops_list, flatten_all_ops_lis
    def __init__(self, env, name, waiting_job_store, production_list, workcenter, RL_ENV,capacity=1, home = None, ):
        super().__init__(env, capacity=capacity)
        self.env = env
        self.home = home
        self.workcenter = workcenter
        self.name = name
        self.RL_ENV = RL_ENV
        self.recent_action = 0
        self.last_setup_remain_time = None
        self.last_process_remain_time = None
        choices = list()
        for ops in flatten_all_ops_list:
            if self.name in ops.alternative_machine_list:
                choices.append(ops.idx)
        self.setup = random.choice(choices)
        self.waiting_job_store = waiting_job_store
        self.production_list = production_list
        self.machine_selection_indicator = 1
        self.count_ = 0
        #self.action_history = [0 for _ in range(num_machines + 1)]
        self.last_status = 'idle'
        self.idle_history = 0
        self.setup_history = 0
        self.process_history = 0
        self.ref_time = 0
        self.p_j_k = 0
        self.current_working_status = False
        self.state = list()
        self.action_type = False
        self.current_working_job = None
        self.last_recorded_idle = 0
        self.last_recorded_setup = None
        self.last_recorded_process = None


        self.last_recorded_idle_for_reward = 0
        self.last_recorded_setup_for_reward = 0
        self.last_recorded_process_for_reward = 0

        self.setup_start = 0
        self.process_start = 0
        self.q_value = 0
        self.idle_start_list = [0]
        #self.action_space = [False for i in range(len(ops_name_list) + 2)]
        self.status = 'idle'
        self.reward = 0
        self.reward_record = 0
        self.last_recorded_first_idle = None
        self.last_recorded_first_setup = None
        self.last_recorded_first_process = None

        self.last_setup_history = 0
        self.last_process_history =0
        self.last_idle_history=0


        self.remaining_setup = 0
        self.remaining_process_time = 0


        self.current_setup_time = None
        self.current_process_time = None
        self.current_idle_time = None
        self.current_setup_time_abs = None
        self.current_process_time_abs = None

        self.current_idle_time_abs = None
        self.complete = True

    def idle_complete_setup_start(self, job):
        self.status = 'setup'
        self.current_working_job  = job

        self.reward += -(self.env.now - self.last_recorded_idle_for_reward)

        self.last_recorded_setup = self.env.now
        self.last_recorded_setup_for_reward = self.env.now


        self.setup_start = self.env.now
        self.current_idle_time = None
        self.current_idle_time_abs = None
        self.idle_complete_setup_start_check = True
        self.complete = False




    def setup_complete_process_start(self, job):
        self.status = 'working'
        self.setup_history += self.env.now - self.last_recorded_setup
        self.last_setup_history = self.setup_history

        self.reward += -(self.env.now - self.last_recorded_setup_for_reward)
        #self.reward_record += -(self.env.now - self.last_recorded_setup)

        self.setup = job.operations[0].idx
        self.last_recorded_process = self.env.now
        self.last_recorded_process_for_reward = self.env.now
        self.last_recorded_setup = self.env.now
        self.last_recorded_setup_for_reward = self.env.now




        self.process_start = self.env.now
        self.current_setup_time = None
        self.current_setup_time_abs = None
        self.setup_complete_process_start_check = True
        self.complete = False


    def process_complete_idle_start(self):
        self.status = 'idle'



        #self.reward += -(self.env.now - self.last_recorded_process_for_reward)
        #self.reward_record += -(self.env.now - self.last_recorded_process_for_reward) # 여기 확인
        self.last_recorded_idle = self.env.now
        self.last_recorded_idle_for_reward = self.env.now


        self.process_history += self.env.now - self.last_recorded_process
        self.current_working_job = None

        self.current_process_time = None
        self.current_process_time_abs = None
        self.process_complete_idle_start_check = True
        self.complete = True






class Process:
    global num_ops, num_job_type, num_jobs, num_machines, setup_list, production_number
    def __init__(self, env, RL_ENV, mode='agent', test=False, eps=False):
        self.env = env
        self.waiting_job_store = simpy.FilterStore(env)
        self.RL_ENV = RL_ENV
        if test == False:
            selection = np.random.randint(0, len(problems))
            scheduling_problem = problems[selection]
            self.scheduling_problem = [int(p) + random.choice([-1, 0, 1]) for p in scheduling_problem]
            #self.scheduling_problem = [int(p) for p in scheduling_problem]

        print(self.scheduling_problem)
        for k in range(len(self.scheduling_problem)):
            pro = self.scheduling_problem[k]
            for j in range(pro):
                self.waiting_job_store.items.append(Job(k, 0))

        self.production_list = [0 for i in range(num_job_type)]
        self.mode = mode
        self.machine_store = simpy.FilterStore(env)
        for p in self.waiting_job_store.items:
            self.production_list[p.job_type] += 1
        self.completed_job_store = simpy.Store(env)
        self.completed_count = [0 for i in range(num_job_type)]
        self.machine_store.items = [Machine(env,
                                            i,
                                            self.waiting_job_store,
                                            self.production_list,
                                            workcenter_name[i],
                                            home = self,
                                            RL_ENV = RL_ENV)
                                    for i in range(num_machines) ]
        self.dummy_res_store = list()
        for machine in self.machine_store.items:
            self.dummy_res_store.append(machine)
        self.sorting_res_store = self.dummy_res_store[:]
        self.action = [0 for _ in range(num_machines)]
        self.change = False
        self.start = True
        self.process = self.env.process(self._execution())
        self.reward = 0
        self.decision_time_step = self.env.event()
        self.action_space = np.arange(len(ops_name_list) + 1)



    def _execution(self):
        if self.mode == 'agent':
            while True:
                yield self.env.timeout(0)
                self.change = True
                yield self.env.timeout(0)
                count = 0
                self.sorting_res_store.sort(key=lambda machine: machine.q_value, reverse = True)

                for m in self.sorting_res_store:
                    count += 1
                    if m in self.machine_store.items:
                        ops_idx = self.action[m.name]
                        if ops_idx < len(ops_name_list):
                            ops = ops_name_list[ops_idx]
                            waiting_ops = set([j.operations[0].idx for j in self.waiting_job_store.items])
                            if ops in waiting_ops:
                                job = yield self.waiting_job_store.get(lambda job: job.operations[0].idx == ops)
                                machine = yield self.machine_store.get(lambda res: res == m)
                                self.env.process(self._do_working(job, machine))
                                if machine.name not in job.operations[0].alternative_machine_list:
                                    print("?????")
                            else:
                                self.action[m.name] = self.action_space[-1]
                        else:
                            pass
                    else:
                        pass
                yield self.decision_time_step
        elif self.mode == 'spsu':
            while True:
                yield self.env.timeout(0)
                self.change = True
                yield self.env.timeout(0)
                for m_idx in range(num_machines):
                    for machine in self.sorting_res_store:
                        temp_setup_list = list()
                        for job in self.waiting_job_store.items:
                            a = ops_name_list.index(machine.setup)
                            b = ops_name_list.index(job.operations[0].idx)
                            if machine in self.machine_store.items and machine.name in job.operations[
                                0].alternative_machine_list:
                                temp_setup_list.append(setup_get(machine, job.operations[0])+job.operations[0].process_time+machine.name)

                            else:
                                temp_setup_list.append(float('inf'))
                        if len(temp_setup_list) > 0:
                            machine.shortest_setup_time = min(temp_setup_list)


                    for job in self.waiting_job_store.items:
                        temp_setup_list = list()
                        for machine in self.sorting_res_store:
                            a = ops_name_list.index(machine.setup)
                            b = ops_name_list.index(job.operations[0].idx)
                            if machine in self.machine_store.items and machine.name in job.operations[
                                0].alternative_machine_list:
                                temp_setup_list.append(setup_get(machine, job.operations[0])+job.operations[0].process_time+machine.name)
                            else:
                                temp_setup_list.append(float('inf'))
                        if len(temp_setup_list) > 0:
                            job.shortest_setup_time = min(temp_setup_list)
                    self.machine_store.items.sort(key=lambda machine: machine.shortest_setup_time)
                    self.waiting_job_store.items.sort(key=lambda job: job.shortest_setup_time)
                    # print("전", [machine.shortest_setup_time for machine in self.machine_store.items])
                    # print("후", [job.shortest_setup_time for job in self.waiting_job_store.items])
                    if len(self.waiting_job_store.items) > 0 and len(self.machine_store.items) > 0 and np.min([job.shortest_setup_time for job in self.waiting_job_store.items]) != float('inf'):
                        machine = yield self.machine_store.get()
                        job = yield self.waiting_job_store.get()
                        #print(machine.name, job.operations[0].alternative_machine_list)
                        if machine.name not in job.operations[0].alternative_machine_list:
                            print("a;a;a;?????")
                        self.env.process(self._do_working(job, machine))
                yield self.decision_time_step
        elif self.mode == 'ssu':
            while True:
                yield self.env.timeout(0)
                self.change = True
                yield self.env.timeout(0)
                for m_idx in range(num_machines):
                    for machine in self.sorting_res_store:
                        temp_setup_list = list()
                        for job in self.waiting_job_store.items:
                            if machine in self.machine_store.items and \
                                    machine.name in job.operations[0].alternative_machine_list:
                                temp_setup_list.append(setup_get(machine, job.operations[0])-machine.name)

                            else:
                                temp_setup_list.append(float('inf'))
                        if len(temp_setup_list) > 0:
                            machine.shortest_setup_time = min(temp_setup_list)


                    for job in self.waiting_job_store.items:
                        temp_setup_list = list()
                        for machine in self.sorting_res_store:
                            if machine in self.machine_store.items and machine.name in job.operations[0].alternative_machine_list:
                                temp_setup_list.append(setup_get(machine, job.operations[0])-machine.name)
                            else:
                                temp_setup_list.append(float('inf'))
                        if len(temp_setup_list) > 0:
                            job.shortest_setup_time = min(temp_setup_list)
                    self.machine_store.items.sort(key=lambda machine: machine.shortest_setup_time)
                    self.waiting_job_store.items.sort(key=lambda job: job.shortest_setup_time)
                    # print("전", [machine.shortest_setup_time for machine in self.machine_store.items])
                    # print("후", [job.shortest_setup_time for job in self.waiting_job_store.items])
                    if len(self.waiting_job_store.items) > 0 and len(self.machine_store.items) > 0 and np.min([job.shortest_setup_time for job in self.waiting_job_store.items]) != float('inf'):
                        machine = yield self.machine_store.get()
                        job = yield self.waiting_job_store.get()
                        if machine.name not in job.operations[0].alternative_machine_list:
                            print(machine.name, job.operations[0].alternative_machine_list)
                            print("?????")

                        self.env.process(self._do_working(job, machine))

                yield self.decision_time_step
    def _do_working(self, job, machine):
        with machine.request() as req:
            yield req
            job.operation_start()
            setup_time = setup_get(machine, job.operations[0])
            machine.idle_complete_setup_start(job)
            machine.est_setup = setup_time
            machine.est_process = job.operations[0].process_time
            soe = 0.1
            machine.current_setup_time = setup_time
            machine.current_setup_time_abs = self.env.now+setup_time
            process_time = job.operations[0].process_time
            machine.current_process_time = process_time
            machine.current_process_time_abs = self.env.now+setup_time+process_time
            setup_time = np.random.gamma(shape=(1 / soe) ** 2, scale=(soe ** 2) * setup_time)
            yield self.env.timeout(setup_time)#np.random.gamma(shape=(1 / soe) ** 2, scale=(soe ** 2) * setup_list[a][b]))
            machine.setup_complete_process_start(job)
            soe = 0.2
            process_time = np.random.gamma(shape=(1 / soe) ** 2, scale=(soe ** 2) * process_time)
            yield self.env.timeout(process_time)#np.random.uniform(0.8*process_time, 1.2*process_time))
            job.operation_complete()
            machine.process_complete_idle_start()
        self.machine_store.put(machine)
        if len(job.operations) == 0:
            self.completed_job_store.put(job)
            if len(self.completed_job_store.items) != sum(self.scheduling_problem):
                self.decision_time_step.succeed()
                self.decision_time_step = self.env.event()
        else:
            self.waiting_job_store.put(job)
            self.decision_time_step.succeed()
            self.decision_time_step = self.env.event()

def setup_get(machine, ops):
    if machine.workcenter ==0:#[0,1,2,3,4,5,6,7,8,9]:#[0,1,2,3,4,5,6,7]:#[0,1,2,3,4,5,6,7,8,9]:#[0,1,2,3,4,5,6,7]:#[0,1,2,3,4,5,6,7,8,9]:
        if len(machine.setup) == 4:
            machine_setup_jobtype = machine.setup[:2]
            machine_setup_type = flatten_all_ops_list[ops_name_list.index(machine.setup)].ops_type
            ops_setup_type = flatten_all_ops_list[ops_name_list.index(ops.idx)].ops_type
            if (int(machine_setup_jobtype) == int(ops.job_type)) and (machine_setup_type == ops_setup_type):

                setup = 0
            elif (int(machine_setup_jobtype) == int(ops.job_type)) and (machine_setup_type != ops_setup_type):

                setup = 30
            elif (int(machine_setup_jobtype) != int(ops.job_type)) and (machine_setup_type == ops_setup_type):

                setup = 60
            else:
                setup = 60
        else:
            machine_setup_jobtype = machine.setup[:1]
            machine_setup_type = flatten_all_ops_list[ops_name_list.index(machine.setup)].ops_type
            ops_setup_type = flatten_all_ops_list[ops_name_list.index(ops.idx)].ops_type
            if (int(machine_setup_jobtype) == int(ops.job_type)) and (machine_setup_type == ops_setup_type):
                setup = 0
            elif (int(machine_setup_jobtype) == int(ops.job_type)) and (machine_setup_type != ops_setup_type):

                setup = 30
            elif (int(machine_setup_jobtype) != int(ops.job_type)) and (machine_setup_type == ops_setup_type):

                setup = 60
            else:
                setup = 60
    else:
        if len(machine.setup) == 4:
            machine_setup_jobtype = machine.setup[:2]
            machine_setup_type = flatten_all_ops_list[ops_name_list.index(machine.setup)].ops_type
            ops_setup_type = flatten_all_ops_list[ops_name_list.index(ops.idx)].ops_type
            if (int(machine_setup_jobtype) == int(ops.job_type)) and (machine_setup_type == ops_setup_type):
                setup = 0
            elif (int(machine_setup_jobtype) == int(ops.job_type)) and (machine_setup_type != ops_setup_type):
                setup = 30
            elif (int(machine_setup_jobtype) != int(ops.job_type)) and (machine_setup_type == ops_setup_type):
                setup = 120
            else:
                setup = 120
        else:
            machine_setup_jobtype = machine.setup[:1]
            machine_setup_type = flatten_all_ops_list[ops_name_list.index(machine.setup)].ops_type
            ops_setup_type = flatten_all_ops_list[ops_name_list.index(ops.idx)].ops_type
            if (int(machine_setup_jobtype) == int(ops.job_type)) and (machine_setup_type == ops_setup_type):
                #print(int(machine_setup_jobtype) == int(ops.job_type))
                setup = 0
            elif (int(machine_setup_jobtype) == int(ops.job_type)) and (machine_setup_type != ops_setup_type):
                setup = 30
            elif (int(machine_setup_jobtype) != int(ops.job_type)) and (machine_setup_type == ops_setup_type):
                setup = 120

            else:
                setup = 120
    return setup



class RL_ENV:
    def __init__(self, mode = 'agent'):
        self.mode = mode
        self.env = simpy.Environment()
        self.reward = 0
        self.proc = Process(self.env, mode = mode, RL_ENV = self)
        self.prev_time = 0
        self.n_agents = num_machines
        self.n_actions = len(ops_name_list) + 1
        self.last_action = np.zeros((self.n_agents, self.n_actions))
        ## observation 나타내기 위함
        self.setup = np.eye(num_ops)
        self.current_working = np.eye(num_ops+1)
        self.status = np.eye(3)
        self.agent_id = np.eye(num_machines)
        self.event_log = list()

        self.last_time_step = 0

        self.action_history = [[0 for i in range(num_ops+1)] for _ in range(num_machines)]
        self.last_action = [len(ops_name_list)]*self.n_agents

        #self.last_action = [len(ops_name_list)] * self.n_agents

        self.action_encodes_identity = np.eye(self.n_actions)

        self.get_edge_index_m_to_m = [[],[]]
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                self.get_edge_index_m_to_m[0].append(i)
                self.get_edge_index_m_to_m[1].append(j)


    def get_env_info(self):
        # node_feature = np.concatenate([np.array([0, 0, 0,
        #                                          first_moment_idle,
        #                                          first_moment_setup,
        #                                          first_moment_process,
        #
        #                                          second_moment_idle,
        #                                          second_moment_setup,
        #                                          second_moment_process,
        #                                          setup_remain_time,
        #                                          process_remain_time]), setup, self.action_history[i]])


        num_agents = num_machines
        env_info = {"n_agents" : num_machines,
                    "job_feature_shape": len(ops_name_list)+len(workcenter)+len(ops_name_list)+1+6+1,  # + self.n_agents,
                    "machine_feature_shape" : 8 + num_job_type+ len(ops_name_list) + 1,
                    "n_actions": len(ops_name_list) + 1
                    }
        print(env_info)
        print(sum(ops_length_list), len(ops_name_list))

        #print(env_info['obs_shape'])
        return env_info
    def get_state(self):
        waiting_ops = [j.operations[0].idx for j in self.proc.waiting_job_store.items]
        num_waiting_operations = [waiting_ops.count(ops) / self.proc.production_list[flatten_all_ops_list[ops_name_list.index(ops)].job_type] if ops in waiting_ops else 0 for ops in ops_name_list]
        num_waiting_operations = np.reshape(num_waiting_operations, (1, -1))
        setup = [self.setup[ops_name_list.index(m.setup)] for m in self.proc.dummy_res_store]
        setup = np.reshape(setup, (1, -1))
        current_working = [self.current_working[ops_name_list.index(m.current_working_job.operations[0].idx)] if m.current_working_job != None else self.current_working[-1]  for m in self.proc.dummy_res_store]
        current_working = np.reshape(current_working, (1, -1))
        state = np.concatenate([num_waiting_operations, setup, current_working], axis = 1)#history, action_history], axis = 1)
        return state


    def _conditional(self, machine, ops):
        if machine in self.proc.machine_store.items:
            if (ops in self.waiting_ops) and (ops in avail_ops_by_machine[machine.name]):
                result = True
            else:
                result = False
        else:
            if ops == machine.current_working_job.operations[0].idx:

                result = True
            else:
                result = False


        return result


    def get_avail_actions(self):
        self.waiting_ops = [j.operations[0].idx for j in self.proc.waiting_job_store.items]
        avail_actions_by_agent = [[self._conditional(m, ops) for ops in ops_name_list] for m in self.proc.dummy_res_store]

        for avail_actions in avail_actions_by_agent:
            if True not in avail_actions:
                avail_actions.append(True)
            else:
                avail_actions.append(False)

        return avail_actions_by_agent





    def render(self):
        while self.proc.change == False:
            self.env.step()
        self.proc.change = False
    # def get_edge_index_for_job(self):
    #     for i in self.proc.dummy_res_store:
    #         for j in self.proc.waiting_job_store.items:
    def get_edge_index_job_machine(self):
        edge_index = [[],[]]
        for i in range(self.n_agents):
            machine = self.proc.dummy_res_store[i]
            for j in range(len(self.proc.waiting_job_store.items)):
                job = self.proc.waiting_job_store.items[j]
                idx = machine.setup
                if idx[2] != '_':
                    setup = int(idx[0])
                else:
                    setup = int(idx[0:2])

                if (job.job_type == setup) and (machine.status == 'idle'):
                    edge_index[0].append(i)
                    edge_index[1].append(j)
        return edge_index


    def get_edge_index_machine_machine(self, fully_connected = False):
        if fully_connected == True:
            edge_index = self.get_edge_index_m_to_m
        else:
            edge_index = [[], []]
            for i in range(self.n_agents):
                machine = self.proc.dummy_res_store[i]
                if machine.status == 'setup' or machine.status == 'working':
                    pass
                else:
                    for j in range(self.n_agents):
                        edge_index[0].append(i)
                        edge_index[1].append(j)
                        # comm_machine = self.proc.dummy_res_store[j]
                        # if comm_machine.status == 'working':
                        #
                        #     edge_index[0].append(i)
                        #     edge_index[1].append(j)
        return edge_index




    def get_node_feature_job(self):
        job_features = list()
        for i in range(self.n_agents):
            machine = self.proc.dummy_res_store[i]
            if machine.current_working_job != None:
                a = machine.current_working_job.operations[0].job_type
                b = machine.current_working_job.operations[0].k
                job_feature = np.concatenate([np.eye(num_jobs)[a], np.eye(max_ops_length)[b]])
            else:
                job_feature = np.zeros(num_jobs+max_ops_length)
            job_features.append(job_feature)


        for j in range(len(self.proc.waiting_job_store.items)):
            job = self.proc.waiting_job_store.items[j]
            job_features.append(job.node_feature)

        return job_features

    def get_node_feature_machine(self):
        status_encoding = np.eye(3)
        time_delta = (self.env.now - self.last_time_step)
        node_features = list()
        workcenter_encodes = np.eye(len(workcenter))
        self.waiting_ops = [j.operations[0].idx for j in self.proc.waiting_job_store.items]
        num_waiting_operations = [self.waiting_ops.count(ops)/self.proc.production_list[flatten_all_ops_list[ops_name_list.index(ops)].job_type]
                                    if ops in self.waiting_ops else 0 for ops in ops_name_list]
        num_waiting_operations.append(time_delta/120)
        status = [0]*self.n_agents
        waiting_ops_list = list()

        process_history_global = 0
        setup_history_global = 0
        idle_history_global = 0
        first_moment_process_global = 0
        first_moment_setup_global = 0
        first_moment_idle_global = 0



        if np.array(self.action_history).sum() > 0:
            action_history = np.array(self.action_history).sum(axis = 0)/np.array(self.action_history).sum()
        else:
            action_history = np.zeros(self.n_actions)


        for i in range(self.n_agents):
            machine = self.proc.dummy_res_store[i]
            # waiting_ops = deepcopy(num_waiting_operations)
            #
            # waiting_ops_list.append(np.concatenate([num_waiting_operations,workcenter_encodes[machine.workcenter]]))
            # waiting_ops.append(num_waiting_operations)
            # num_total_action = sum(self.action_history[i])
            last_action = self.last_action[i]
            last_action_encodes = self.action_encodes_identity[last_action]

            if machine.status == 'setup':
                k = 0
                machine.setup_history += self.env.now - machine.last_recorded_setup
                machine.last_setup_history = machine.setup_history
                if time_delta != 0:
                    first_moment_process = 0
                    first_moment_setup = (self.env.now - machine.last_recorded_setup) / time_delta
                    first_moment_idle = 0
                else:
                    first_moment_process = 0
                    first_moment_setup = 0
                    first_moment_idle = 0
                setup_remain_time = (machine.current_setup_time_abs - self.env.now)/machine.current_setup_time
                process_remain_time = 1
                machine.last_recorded_setup = self.env.now



            if machine.status == 'working':
                k = 1
                machine.process_history += self.env.now - machine.last_recorded_process
                machine.last_process_history = machine.process_history
                if time_delta != 0:
                    first_moment_process = (self.env.now - machine.last_recorded_process)/time_delta

                    first_moment_idle = 0
                    first_moment_setup =1-first_moment_process-first_moment_idle #(self.env.now - machine.last_recorded_setup) / time_delta
                else:
                    first_moment_process = 0
                    first_moment_setup = 0
                    first_moment_idle = 0

                setup_remain_time = 0
                process_remain_time = (machine.current_process_time_abs - self.env.now)/machine.current_process_time

                machine.last_recorded_process = self.env.now
            if machine.status == 'idle':
                status[i] = 1
                k = 2
                machine.idle_history += self.env.now - machine.last_recorded_idle
                machine.last_idle_history = machine.idle_history

                if time_delta != 0:
                    first_moment_process = 0
                    first_moment_setup =0
                    first_moment_idle =  (self.env.now - machine.last_recorded_idle) / time_delta
                else:
                    first_moment_process = 0
                    first_moment_setup = 0
                    first_moment_idle = 0
                # machine.last_recorded_first_setup = 0
                # machine.last_recorded_first_process = 0
                # machine.last_recorded_first_idle = self.env.now - machine.last_recorded_idle
                setup_remain_time = 0
                process_remain_time = 0
                machine.last_recorded_idle = self.env.now
            idx = machine.setup
            if time_delta != 0:
                #print(machine.process_history/time_delta+machine.setup_history/time_delta+machine.idle_history/time_delta)
                process_history_global += machine.process_history/self.env.now
                setup_history_global += machine.setup_history/self.env.now
                idle_history_global += machine.idle_history/self.env.now
                first_moment_process_global += first_moment_process/time_delta
                first_moment_setup_global += first_moment_setup/time_delta
                first_moment_idle_global += first_moment_idle/time_delta
            if idx[2] != '_':
                a = int(idx[0])
                b = int(idx[2])
            else:
                a = int(idx[0:2])
                b = int(idx[3])
            machine.last_status = machine.status
            setup = np.eye(num_jobs)[a]

            if machine.last_recorded_first_idle != None:
                second_moment_idle = first_moment_idle - machine.last_recorded_first_idle

            else:
                second_moment_idle = 0



            machine.last_recorded_first_idle = first_moment_idle

            if machine.last_recorded_first_setup != None:
                second_moment_setup = first_moment_setup - machine.last_recorded_first_setup
            else:
                second_moment_setup = 0


            # if machine.name == 5:
            #     print("전", second_moment_setup, first_moment_setup, machine.last_recorded_first_setup)
            machine.last_recorded_first_setup = first_moment_setup
            # if machine.name == 5:
            #     print("후", second_moment_setup, first_moment_setup, machine.last_recorded_first_setup)

            if machine.last_recorded_first_process != None:
                second_moment_process = first_moment_process - machine.last_recorded_first_process
            else:
                second_moment_process = 0
            # if machine.name == 5:
            #     print("후1", second_moment_setup, first_moment_setup, machine.last_recorded_first_setup)
            machine.last_recorded_first_process = first_moment_process
            # if machine.name == 5:
            #     print("후2", second_moment_setup, first_moment_setup, machine.last_recorded_first_setup)


            if machine.last_setup_remain_time != None and time_delta != 0 :
                first_moment_setup_remain_time = -(machine.last_setup_remain_time - setup_remain_time) / time_delta
            else:
                first_moment_setup_remain_time = 0

            if machine.last_process_remain_time != None and time_delta !=0:
                first_moment_process_remain_time = -(
                            machine.last_process_remain_time - process_remain_time) / time_delta

            else:
                first_moment_process_remain_time = 0

            first_moment_idle = first_moment_idle
            first_moment_setup = first_moment_setup
            first_moment_process=first_moment_process
            setup_remain_time =setup_remain_time
            process_remain_time = process_remain_time
            first_moment_process_remain_time = first_moment_process_remain_time
            first_moment_setup_remain_time = first_moment_setup_remain_time
            machine.last_setup_remain_time = setup_remain_time
            machine.last_process_remain_time = process_remain_time
            # if machine.name == 5:
            if self.env.now == 0:
                node_feature = np.concatenate([np.array([0, 0, 0,
                                                         first_moment_idle,
                                                         first_moment_setup,
                                                         first_moment_process,

                                                         setup_remain_time,
                                                         process_remain_time]), setup, last_action_encodes])

            else:
                # node_feature = np.concatenate([np.array([machine.idle_history/self.env.now,
                #                         machine.setup_history/self.env.now,
                #                                              machine.process_history/self.env.now]), setup, np.array(self.action_history[i])/num_total_action, workcenter_encodes[machine.workcenter]])
                node_feature = np.concatenate([np.array([machine.idle_history/self.env.now,
                                        machine.setup_history/self.env.now,
                                                             machine.process_history/self.env.now,
                                                             first_moment_idle,
                                                             first_moment_setup,
                                                             first_moment_process,


                                                             setup_remain_time,
                                                             process_remain_time]), setup, last_action_encodes])


            node_features.append(node_feature)
        #len(ops_name_list) + len(workcenter) + len(ops_name_list) + 1 + 6 + len(ops_name_list) + 1


        #print(process_history_global/self.n_agents, process_history_global)
        history = [process_history_global/self.n_agents, setup_history_global/self.n_agents, idle_history_global/self.n_agents, first_moment_process_global/self.n_agents, first_moment_setup_global/self.n_agents, first_moment_idle_global/self.n_agents]
        #print(process_history_global/self.n_agents+setup_history_global/self.n_agents+idle_history_global/self.n_agents)
        for i in range(self.n_agents):
            machine = self.proc.dummy_res_store[i]
            waiting_ops_list.append(np.concatenate([num_waiting_operations,
                                                    workcenter_encodes[machine.workcenter],
                                                    action_history,
                                                    history]).tolist())


        #print(len(waiting_ops_list[0], len(ops_name_list)+len(workcenter)+len(ops_name_list)+1+6+len(ops_name_list)+1)




        self.last_time_step = self.env.now
        #print(np.array(waiting_ops_list).shape)
        return node_features, waiting_ops_list, status

    def get_heterogeneous_graph(self):
        node_features, num_waiting_operations, status = self.get_node_feature_machine()
        return node_features, num_waiting_operations, self.get_edge_index_machine_machine(), status

    def reset(self):

        self.env = simpy.Environment()
        self.reward = 0
        self.proc = Process(self.env, mode = self.mode, RL_ENV = self)
        self.prev_time = 0
        self.n_agents = num_machines
        self.n_actions = len(ops_name_list) + 1
        self.last_action = np.zeros((self.n_agents, self.n_actions))
        ## observation 나타내기 위함
        self.setup = np.eye(num_ops)
        self.current_working = np.eye(num_ops+1)
        self.status = np.eye(3)
        self.agent_id = np.eye(num_machines)
        self.event_log = list()
        self.action_history = [[0 for i in range(num_ops + 1)] for _ in range(num_machines)]
        self.last_time_step = 0

        self.last_action = [len(ops_name_list)] * self.n_agents

    def step(self, actions, vdn, q_values = False):
        self.proc.action = actions


        for m in self.proc.dummy_res_store:
            m.q_value = q_values[m.name]
        done = False
        while self.proc.change == False:
            try:
                self.env.step()
                changed_actions = self.proc.action
            except simpy.core.EmptySchedule:
                #print(sum(self.proc.scheduling_problem), len(self.proc.completed_job_store.items))
                done = True
                changed_actions = self.proc.action
                break
        actions_int = [int(a) for a in changed_actions]
        #self.last_action = np.eye(self.n_actions)[np.array(actions_int)]
        self.proc.change = False
        if vdn == True:
            reward = 0
            for machine in self.proc.dummy_res_store:
                if machine.status == 'idle':
                    machine.reward += -(self.env.now - machine.last_recorded_idle_for_reward)
                    machine.last_recorded_idle_for_reward = self.env.now
                elif machine.status == 'working':
                    machine.last_recorded_process_for_reward = self.env.now
                elif machine.status == 'setup':
                    machine.reward += -(self.env.now - machine.last_recorded_setup_for_reward)
                    machine.last_recorded_setup_for_reward = self.env.now
                reward += machine.reward
                machine.reward = 0

            for m in range(len(changed_actions)):
                a = changed_actions[m]
                self.action_history[m][a] += 1
        if vdn == False:
            reward = list()
            for machine in self.proc.dummy_res_store:
                if machine.status == 'idle':
                    machine.reward += -(self.env.now - machine.last_recorded_idle_for_reward)
                    reward.append(machine.reward)
                    # machine.reward_record += -(self.env.now - machine.last_recorded_idle_for_reward)
                    machine.last_recorded_idle_for_reward = self.env.now
                elif machine.status == 'working':
                    reward.append(0)
                    machine.last_recorded_process_for_reward = self.env.now
                    # pass
                elif machine.status == 'setup':
                    machine.reward += -(self.env.now - machine.last_recorded_setup_for_reward)
                    reward.append(machine.reward)
                    # machine.reward_record += -(self.env.now - machine.last_recorded_setup_for_reward)
                    machine.last_recorded_setup_for_reward = self.env.now
                machine.reward = 0

            for m in range(len(changed_actions)):
                a = changed_actions[m]
                self.action_history[m][a] += 1

        return reward, done, changed_actions

def normalizer(input):
    if sum(input) != 0:
        return [i/sum(input) for i in input]
    else:
        return [0 for i in input]