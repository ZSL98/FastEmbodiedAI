"""
Design how to group computation stages
"""
import torch
import argparse
import sys
sys.path.append("../../repos/x-transformers")
from utils.util_func import max_below_threshold
import xformers

MAX_QUERY_TS = 1000

parser = argparse.ArgumentParser()
parser.add_argument('--query_interval', default=2, help='The iterval of requests to come', type=int)
parser.add_argument('--input_seq_len', default=128, help='input sequence length', type=int)

parser.add_argument('--encoder_n', default=1, help='The number of ts that encoder takes', type=int)
parser.add_argument('--prefill_n', default=1, help='The number of ts that prefill takes', type=int)
parser.add_argument('--decode_n', default=1, help='The number of ts that decode takes', type=int)

parser.add_argument('--encoder_len', default=1, help='Number of steps for encode', type=int)
parser.add_argument('--prefill_len', default=1, help='Number of steps for prefill', type=int)
parser.add_argument('--decode_len', default=6, help='Number of steps for decoding', type=int)
parser.add_argument('--simu_ts_len', default=20, help='Total ts len for simulation', type=int)

parser.add_argument('--real_run', default=False, help='If require to import torch', type=bool)
parser.add_argument('--enable_recompute', default=False, help='If enable replace decode with prefill', type=bool)

parser.add_argument('--mode', default='profile', help='mode', type=str)
parser.add_argument('--req_interval', default=0, help='req_interval', type=float)

## LLM transformer detail
parser.add_argument('--dim', default=512, help='LLM transformer dimension', type=int)

## profile arguments
parser.add_argument('--warmup_num', default=100, help='warmup_num', type=int)
parser.add_argument('--trail_num', default=200, help='trail_num', type=int)
parser.add_argument('--profile_mode', default='base', help='profile_mode', type=str)

## others
parser.add_argument('--verbose', default=0, help='verbose level', type=int)

# parser.add_argument('--kv_len', default=)
args = parser.parse_args()

class Query:
    def __init__(self, args, query_id):
        self.query_id = query_id
        self.task_step = 0
        self.duration = 0
        self.query_latency = 0

class SimuQueryManage:
    def __init__(self, args):
        self.query_list = list()
        self.task_len = args.encoder_n * args.encoder_len + \
                        args.prefill_n * args.prefill_len + \
                        args.decode_n * args.decode_len
        self.task = ['e'] * args.encoder_len + ['p'] * args.prefill_len + ['d'] * args.decode_len
        self.task_detail = ['e'] * args.encoder_len * args.encoder_n + \
                           ['p'] * args.prefill_len * args.prefill_n + \
                           ['d'] * args.decode_len * args.decode_n

        self.query_num = 0
        self.simu_query_interval_ms = 0
        self.args = args
        self.context_update_pos = []
    
    def update(self, ts):
        for query in self.query_list:
            query.task_step = query.task_step + 1
        
        self.query_list = list(filter(lambda x: x.task_step < self.task_len, self.query_list))
        if ts % self.args.query_interval == 0:
            self.context_update_pos.append(ts)
            self.query_list.append(Query(args, self.query_num))
            self.query_num = self.query_num + 1


    def gen_plan(self):
        ts_group = list()
        for query in self.query_list:
            ts_group.append(query.task_step)
        
        ts_group = [self.task_detail[i] for i in ts_group]
        return ts_group

    def replace_task(self, ts, task_plan, enable_recompute: bool):
        # This function is no need anymore
        if not enable_recompute:
            return task_plan
        elif (ts-1) % self.args.query_interval == 0: # The context has been updated within the last ts
            for i in range(len(task_plan)):
                # Replace all decode operations to prefill operations
                if task_plan[i] == 'd':
                    task_plan[i] = 'p'
            return task_plan

    def update_task(self, enable_recompute: bool):
        if not enable_recompute:
            return
        else:
            # update self.task, task_detail, task_len
            replace_pos = [i+self.args.encoder_n for i in range(0, MAX_QUERY_TS, self.args.query_interval)]
            pointer = 0
            new_task_detail = self.task_detail.copy()
            while replace_pos[pointer] < len(new_task_detail):
                # replace decode_n ts of decoding with prefill_n ts of prefilling
                if new_task_detail[replace_pos[pointer]] == 'd':
                    new_task_detail[replace_pos[pointer] : replace_pos[pointer] + self.args.decode_n] = ['p'] * self.args.prefill_n
                pointer = pointer + 1
            self.task_detail = new_task_detail
            self.task_len = len(self.task_detail)
            print('new task detail: ', self.task_detail)

    def get_query(self, query_id):
        return self.query_list[query_id]


class SimuScheduler:
    def __init__(self, args, q):
        self.simu_len = args.simu_ts_len
        self.query_manager = q
        self.sche_record = list()
        self.args = args

    def schedule(self):
        self.query_manager.update_task(self.args.enable_recompute)
        print(self.query_manager.task_detail)
        print(self.query_manager.task_len)
        for ts in range(self.simu_len):
            self.query_manager.update(ts)
            task_plan = self.query_manager.gen_plan()
            # task_plan = self.query_manager.replace_task(ts, task_plan, self.args.enable_recompute)
            self.sche_record.append(task_plan)
        
        # for item in self.sche_record:
        #     print(item)

        # Dictionary to store counts
        count_dict = {}

        # Count occurrences of each element
        for item in self.sche_record:
            item_tuple = tuple(item)
            if item_tuple in count_dict:
                count_dict[item_tuple] += 1
            else:
                count_dict[item_tuple] = 1

        # Print the counts
        for item, count in count_dict.items():
            print(f"{item}: {count}")

        return count_dict


    def data_analyze(self, sche_plan, profile_data):
        ts_duration_all = []
        for ts in self.sche_record:
            ts_duration = profile_data[tuple(ts)]
            ts_duration_all.append(ts_duration)

        query_num = int((len(ts_duration_all) - self.query_manager.task_len) / self.args.query_interval)
        print('query num: ', query_num)

        out_seq_len = self.args.prefill_len + self.args.decode_len
        # token_gen_pos is the position of ts in which a token is generated

        token_gen_pos = []
        for index, value in enumerate(self.query_manager.task_detail):
            if value == 'p':
                token_gen_pos.append(index)
                index = index + self.args.prefill_n - 1
            elif value == 'd':
                token_gen_pos.append(index)
                index = index + self.args.decode_n - 1

        print('token_gen_pos: ', token_gen_pos)

        all_query_token_latency = 0
        query_durations = []
        for query_id in range(query_num):
            query_start_ts = query_id * self.args.query_interval
            query_end_ts = query_start_ts + self.query_manager.task_len
            query_duration = sum(ts_duration_all[query_start_ts:query_end_ts])

            query_durations.append(query_duration)

            all_token_latency = 0
            if self.args.enable_recompute:
                for relative_ts_pos in range(self.query_manager.task_len):
                    if relative_ts_pos in token_gen_pos:
                        ts_pos = relative_ts_pos + self.args.query_interval * query_id
                        last_context_update_ts = max_below_threshold(self.query_manager.context_update_pos, ts_pos)
                        # print(last_context_update_ts, ":", ts_pos)
                        token_latency = sum(ts_duration_all[last_context_update_ts:ts_pos])
                        all_token_latency = all_token_latency + token_latency
            else:
                last_context_update_ts = self.args.query_interval * query_id
                for relative_ts_pos in range(self.query_manager.task_len):
                    ts_pos = relative_ts_pos + last_context_update_ts
                    if relative_ts_pos in token_gen_pos:
                        token_latency = sum(ts_duration_all[last_context_update_ts:ts_pos])
                        all_token_latency = all_token_latency + token_latency
            # print('all token latency: ', all_token_latency)
            all_query_token_latency = all_query_token_latency + all_token_latency

        avg_query_token_latency = all_query_token_latency / query_num / out_seq_len
        avg_query_duration = sum(query_durations) / query_num
        print("Query duration: {:.2f} ms".format(avg_query_duration))
        print("Avg token latency: {:.2f} ms".format(avg_query_token_latency))
        print("frame interval: {:.2f} ms".format(sum(ts_duration_all) / self.args.simu_ts_len * self.args.query_interval))

if __name__ == "__main__":

    torch.cuda.set_device(1)

    q_manager = SimuQueryManage(args)
    s = SimuScheduler(args, q_manager)
    if args.mode != "profile":
        sche_plan = None
    else:
        sche_plan = s.schedule()

    if args.real_run:
        import x_transformers
        from mirasol_inference.inference import mirasol_run
        from kernel_profile.flashinfer.decode_test import flashinfer_decode

        # flashinfer_decode(sche_plan = sche_plan)
        # exit()

        profile_data = mirasol_run(sche_plan = sche_plan, mode = args.mode)
        if sche_plan is not None:
            s.data_analyze(sche_plan, profile_data)
