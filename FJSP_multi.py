import simpy
import time
from copy import deepcopy
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


from scheduling_problems.problem_generator import num_machines, num_job_type,  soe, process_time_list, ops_name_array, ops_name_list, alternative_machine_list, setup_list, problems, ops_type_list
#print(alternative_machine_list[0])

import numpy as np
import pandas as pd
np.random.seed(42)




# Operation 객체는 Job이 수행해야하는 Process 정보, alternative machine 정보를 포함한다.
class Operation():
    global num_machines, process_time_list, alternative_machine_list

    def __init__(self, job_type, name, num_m=num_machines):
        self.job_type = job_type
        self.idx = "{}_{}".format(job_type, name)

        if self.idx[2] != '_':
            a = int(self.idx[0])
            b = int(self.idx[2])
        else:
            a = int(self.idx[0:2])
            b = int(self.idx[3])
        self.ops_type = ops_type_list[a][b]
        #print(self.ops_type)
        self.process_time = process_time_list[a][b]
        self.alternative_machine_list = eval(alternative_machine_list[a][b])



all_ops_list = list()
flatten_all_ops_list = list()
for j in range(len(process_time_list)):
    job_j_operation_list = list()
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
print(avail_ops_by_machine[0])

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
    def operation_start(self):
        self.current_working_operation = self.operations[0]
    def operation_complete(self):
        self.completed_operations.append(self.operations[0])
        self.current_working_operation = False
        del self.operations[0]


# Machine 객체이고 idle, setup, process 시간 등을 기록함
class Machine(simpy.Resource):
    global all_ops_list, flatten_all_ops_lis
    def __init__(self, env, name, waiting_job_store, production_list, capacity=1, home = None):
        super().__init__(env, capacity=capacity)
        self.env = env
        self.home = home
        self.name = name
        self.recent_action = 0
        choices = list()
        for ops in flatten_all_ops_list:
            if self.name in ops.alternative_machine_list:
                choices.append(ops.idx)
        self.setup = np.random.choice(choices)
        self.waiting_job_store = waiting_job_store
        self.production_list = production_list
        self.machine_selection_indicator = 1
        self.count_ = 0

        self.action_history = [0 for _ in range(num_machines + 1)]

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
        self.setup_start = 0
        self.process_start = 0
        self.q_value = 0
        self.rew = 0
        self.idle_start_list = [0]
        self.action_history = [0 for i in range(len(flatten_all_ops_list) + 1)]
        self.action_space = [False for i in range(len(flatten_all_ops_list) + 1)]
        self.status = 'idle'
    def idle_complete_setup_start(self, job):
        self.status = 'setup'
        self.current_working_job  = job
        self.idle_history += self.env.now - self.last_recorded_idle
        self.home.reward += -(self.env.now - self.last_recorded_idle)
        self.rew +=  -(self.env.now - self.last_recorded_idle)
        self.last_recorded_setup = self.env.now
        self.setup_start = self.env.now

    def setup_complete_process_start(self, job):
        self.status = 'working'
        self.setup = job.operations[0].idx
        self.setup_history += self.env.now - self.last_recorded_setup
        self.home.reward += -(self.env.now - self.last_recorded_setup)
        self.rew += -(self.env.now - self.last_recorded_setup)
        self.last_recorded_process = self.env.now
        self.process_start = self.env.now
        #print(self.name, "working 작동", self.env.now)
    def process_complete_idle_start(self):
        self.status = 'idle'
        self.current_working_job = None
        self.process_history += self.env.now - self.last_recorded_process
        self.last_recorded_idle = self.env.now


        #print(self.name, "idle 작동", self.env.now)

class Process:
    global num_ops, num_job_type, num_jobs, num_machines, setup_list, production_number
    def __init__(self, env, mode='agent', test=False, eps=False):
        self.env = env
        self.waiting_job_store = simpy.FilterStore(env)

        if test == False:
            selection = np.random.randint(0, len(problems))
            scheduling_problem = problems[selection]
            self.scheduling_problem = [int(p) + np.random.choice([-2, -1, 0, 1, 2]) for p in scheduling_problem]
            #print(self.scheduling_problem)
        # else:
        #     selection = np.random.randint(0, len(problems))
        #     scheduling_problem = test_problems[selection]
        #
        #     # print("뿅뿅")
        #     self.scheduling_problem = [int(p) + np.random.choice([-2, -1, 0, 1, 2]) for p in scheduling_problem]

        for k in range(len(self.scheduling_problem)):
            pro = self.scheduling_problem[k]
            for j in range(pro):
                self.waiting_job_store.items.append(Job(k, 0))

        print(self.scheduling_problem)
        self.production_list = [0 for i in range(num_job_type)]
        self.mode = mode

        self.machine_store = simpy.FilterStore(env)
        for p in self.waiting_job_store.items:
            self.production_list[p.job_type] += 1

        self.completed_job_store = simpy.Store(env)
        self.completed_count = [0 for i in range(num_job_type)]
        self.machine_store.items = [Machine(env, i, self.waiting_job_store, self.production_list, home = self) for i in
                                    range(num_machines)]

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
        self.action_space = np.arange(num_ops + 1)
    def _update_current_status(self):
        for m in self.dummy_res_store:
            if m.status == 'idle':
                m.idle_history += self.env.now - m.last_recorded_idle
                self.reward += -(self.env.now - m.last_recorded_idle)
                #m.rew += -(self.env.now - m.last_recorded_idle)
                m.last_recorded_idle = self.env.now
            if m.status == 'setup':
                m.setup_history += self.env.now - m.last_recorded_setup
                self.reward += -(self.env.now - m.last_recorded_setup)
                #m.rew += -(self.env.now - m.last_recorded_setup)
                m.last_recorded_setup = self.env.now
            if m.status == 'working':
                m.process_history += self.env.now - m.last_recorded_process
                m.last_recorded_process = self.env.now
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
                    m.action_history[self.action[m.name]] += 1
                    if m in self.machine_store.items:
                        ops_idx = self.action[m.name]

                        if ops_idx < len(ops_name_list):
                            ops = ops_name_list[ops_idx]
                            waiting_ops = set([j.operations[0].idx for j in self.waiting_job_store.items])
                            if ops in waiting_ops:
                                job = yield self.waiting_job_store.get(lambda job: job.operations[0].idx == ops)
                                machine = yield self.machine_store.get(lambda res: res == m)
                                self.env.process(self._do_working(job, machine))
                                #print(machine.name, ops_name_list.index(job.operations[0].idx))
                                if machine.name not in job.operations[0].alternative_machine_list:
                                    print("?????")

                                #print((machine.name, self.action[m.name]))


                            else:
                                self.action[m.name] = self.action_space[-1]
                        else:
                            pass
                    else:
                        pass
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
                            a = ops_name_list.index(machine.setup)
                            b = ops_name_list.index(job.operations[0].idx)
                            if machine in self.machine_store.items and machine.name in job.operations[
                                0].alternative_machine_list:
                                temp_setup_list.append(setup_list[a][b])
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
                                temp_setup_list.append(setup_list[a][b])
                            else:
                                temp_setup_list.append(float('inf'))
                        if len(temp_setup_list) > 0:
                            job.shortest_setup_time = min(temp_setup_list)

                    self.machine_store.items.sort(key=lambda machine: machine.shortest_setup_time)
                    #print([m.shortest_setup_time for m in self.machine_store.items])
                    self.waiting_job_store.items.sort(key=lambda job: job.shortest_setup_time)


                    if len(self.waiting_job_store.items) > 0 and len(self.machine_store.items) > 0:

                        machine = yield self.machine_store.get()
                        job = yield self.waiting_job_store.get()
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

            s = deepcopy(setup_time)
            soe = 0.1
            #setup_time = np.random.gamma(shape=(1 / soe) ** 2, scale=(soe ** 2) * setup_time)

            yield self.env.timeout(setup_time/60)#np.random.gamma(shape=(1 / soe) ** 2, scale=(soe ** 2) * setup_list[a][b]))

            machine.setup_complete_process_start(job)
            soe = 1.0
            process_time = job.operations[0].process_time
            process_time = np.random.gamma(shape=(1 / soe) ** 2, scale=(soe ** 2) * process_time)
            yield self.env.timeout(process_time/60)#np.random.uniform(0.8*process_time, 1.2*process_time))
            job.operation_complete()
            machine.process_complete_idle_start()

        self.machine_store.put(machine)
        self._update_current_status()
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
    if machine.name in [0,1,2,3,4]:#[0,1,2,3,4,5,6,7]:#[0,1,2,3,4,5,6,7,8,9]:#[0,1,2,3,4,5,6,7]:#[0,1,2,3,4,5,6,7,8,9]:
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
    def __init__(self, mode = 'agent', seed = np.random.randint(1, 10000000)):
        self.seed = seed
        np.random.seed(self.seed)
        self.env = simpy.Environment()
        self.proc = Process(self.env, mode = mode)
        self.action_space = np.arange(num_ops)
        self.prev_time = 0
        self.n_agents = num_machines
        self.n_actions = len(ops_name_list) + 1
        self.last_action = np.zeros((self.n_agents, self.n_actions))
        # observation 나타내기 위함
        self.setup = np.eye(num_ops)
        self.current_working = np.eye(num_ops+1)
        self.status = np.eye(3)
        self.agent_id = np.eye(num_machines)
    def get_env_info(self):
        num_agents = num_machines
        env_info = {"n_agents" : num_machines,
                    "obs_shape" : len(ops_name_list)*2 + 3 + len(ops_name_list) +1, # + self.n_agents,
                    "state_shape" : len(ops_name_list) + len(ops_name_list)*num_machines + (len(ops_name_list)+1)*num_machines,
                    #len(ops_name_list) + len(ops_name_list) * num_machines + 3 * num_machines + (len(ops_name_list) +1),
                    "n_actions": len(ops_name_list) + 1
                    }

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

        # try:
        #     history = [[m.idle_history/sum([m.idle_history, m.setup_history, m.process_history]),
        #                m.setup_history/sum([m.idle_history, m.setup_history, m.process_history]),
        #                m.process_history/sum([m.idle_history, m.setup_history, m.process_history])] for m in self.proc.dummy_res_store]
        # except ZeroDivisionError:
        #     history = [[0, 0, 0] for m in self.proc.dummy_res_store]
        #
        # history = np.reshape(history, (1, -1))
        #
        #
        # action_history = np.sum([m.action_history for m in self.proc.dummy_res_store], axis = 0).tolist()
        #
        # action_history = normalizer(action_history)
        # action_history = np.reshape(action_history, (1, -1))

        state = np.concatenate([num_waiting_operations, setup, current_working], axis = 1)#history, action_history], axis = 1)
        #print(state.shape)
        return state

    def get_obs(self):
        status = ['idle', 'setup', 'working']

        self.waiting_ops = [j.operations[0].idx for j in self.proc.waiting_job_store.items]
        num_waiting_operations = [[self.waiting_ops.count(ops)/self.proc.production_list[flatten_all_ops_list[ops_name_list.index(ops)].job_type]
                                    if ops in self.waiting_ops and ops in avail_ops_by_machine[m.name] else 0 for ops in ops_name_list] for m in self.proc.dummy_res_store]
        setup = [self.setup[ops_name_list.index(m.setup)] for m in self.proc.dummy_res_store]
        #status = [self.status[status.index(m.status)] for m in self.proc.dummy_res_store]
        action_history = [normalizer(m.action_history) for m in self.proc.dummy_res_store]
        try:
            history = [[m.idle_history/sum([m.idle_history, m.setup_history, m.process_history]),
                       m.setup_history/sum([m.idle_history, m.setup_history, m.process_history]),
                       m.process_history/sum([m.idle_history, m.setup_history, m.process_history])] for m in self.proc.dummy_res_store]
        except ZeroDivisionError:
            history = [[0, 0, 0] for m in self.proc.dummy_res_store]
        observations = np.concatenate([num_waiting_operations, setup, history, action_history], axis = 1)#, agents_id], axis = 1)
        return observations

    def _conditional(self, machine, ops):
        if machine in self.proc.machine_store.items:
            if ops in self.waiting_ops and ops in avail_ops_by_machine[machine.name]:
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
        # avail_actions_by_agent = [[True
        #                            if ops in self.waiting_ops and
        #                                    ops in avail_ops_by_machine[m.name] and
        #                                     m in self.proc.machine_store.items
        #                             else False
        #
        #                            for ops in ops_name_list]
        #                                                     for m in self.proc.dummy_res_store]


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

    def step(self, actions, q_values = False):
        self.proc.action = actions
        #print([pair for pair in enumerate(actions)])
        #print(self.proc.dummy_res_store[0].idle_history, self.proc.dummy_res_store[0].setup_history, "reward :", self.proc.dummy_res_store[0].rew, self.proc.dummy_res_store[0].process_history, sum([self.proc.dummy_res_store[0].idle_history, self.proc.dummy_res_store[0].setup_history, self.proc.dummy_res_store[0].process_history]), self.env.now)
        if type(q_values) == 'list':
            for m in self.proc.dummy_res_store:
                m.q_value = q_values[m.name]

        done = False
        while self.proc.change == False:
            try:
                self.env.step()
                changed_actions = self.proc.action
            except simpy.core.EmptySchedule:
                done = True
                changed_actions = self.proc.action
                break

        actions_int = [int(a) for a in changed_actions]
        self.last_action = np.eye(self.n_actions)[np.array(actions_int)]

        self.proc.change = False
        reward = self.proc.reward
        self.proc.reward = 0


        return reward, done, changed_actions

def normalizer(input):
    if sum(input) != 0:
        return [i/sum(input) for i in input]
    else:
        return [0 for i in input]