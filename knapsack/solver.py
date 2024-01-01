#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import time
from queue import Queue

Item = namedtuple("Item", ['index', 'value', 'weight'])
State = namedtuple("State", ['capacity', 'value', 'bound'])

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])
    print("Capacity: ", capacity, " Length of items: ", item_count)

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))
        # print(int(parts[0]), int(parts[1]))
        
    taken = [0 for _ in range(len(items))]
    
    DP = 1 if len(items) < 400 else 0
    BB_EX = 0
    BB_LR_DFS = 0
    BB_LR_DFS_ITER = 1 if 300 < len(items) <= 1000 else 0 
    GREEDY = 1 if 1000 < len(items) <= 10000 else 0
    
    '''Greedy Solution'''
    if GREEDY:
        # a trivial algorithm for filling the knapsack
        # it takes items in-order until the knapsack is full
        value = 0
        weight = 0
        taken = [0]*len(items)
        taken = [0 for _ in range(len(items))]

        for item in items:
            if weight + item.weight <= capacity:
                taken[item.index] = 1
                value += item.value
                weight += item.weight
        
        # prepare the solution in the specified output format
        output_data = str(value) + ' ' + str(0) + '\n'
        output_data += ' '.join(map(str, taken))
        return output_data
    
    '''Dynamic Programming Solution'''
    
    if DP:
        # capacity = 1000
        table = [[0 for _ in range(len(items) + 1)] for _ in range(capacity)]
        value_weight = [(i, (i.value/i.weight)) for i in items]
        value_weight = sorted(value_weight, key = lambda X: X[1], reverse=True)
        items = [i[0] for i in value_weight]
        
        # Build DP table
        for i in range(len(items)):
            for w in range(1, capacity):
                if items[i].weight > w:
                    table[w][i + 1] = table[w][i]
                else:
                    table[w][i + 1] = max(table[w][i], items[i].value + table[w - items[i].weight][i])
        
        print("Finished constructing the table")
        
        # Recreate Solution
        weight = len(table) - 1
        for item in range(len(items), -1, -1):
            if table[weight][item] != table[weight][item - 1]:
                taken[item - 1] = 1
                weight -= items[item - 1].weight
            else:
                continue       
            
        print("Finished constructing the solution")
        
        taken_idx = [(i.index, t) for i, t in zip(items, taken)]
        taken_idx = sorted(taken_idx, key=lambda X: X[0])
        taken = [t for _, t in taken_idx]         
        
        # prepare the solution in the specified output format
        output_data = str(table[-1][-1]) + ' ' + str(1) + '\n'
        output_data += ' '.join(map(str, taken))
        return output_data
    
    '''DFS Solution (Very Slow after 40 Items)'''  
        
    def bound_calc(items, taken, capacity, val_weight):
        if 0:
            bound = sum([i.value for i in items])
            for i in range(len(taken)):
                if taken[i] == 0:
                    bound -= items[i].value
            return bound        
        if 1:          
            t0 = time.time()
            bound = 0
            taken = taken + [1 for _ in range(len(items) - len(taken))] 
            for i in range(len(items)):
                item, vw = val_weight[i]
                if taken[i] == 1:
                    if item.weight < capacity:
                        bound += item.value
                        capacity -= item.weight
                    elif capacity > 0:
                        bound += (vw*capacity)           
                        break   
                    
            total = time.time() - t0
            return bound
    
    taken_max = [i for i in taken]
    value_max = 0
    
    def dfs(state, items, idx, taken, capacity, val_weight, count, value_max, taken_max):
        
        if idx == len(items):
            return [i for i in taken], state.value, count+1
        
        # Check value and remaining capacity after including item at idx, call dfs on next item
        if state.capacity - items[idx].weight > 0:
            new_b = bound_calc(items, taken + [1], capacity, val_weight)
            # if new_b > state.value:
            if new_b > value_max:
                new_v = state.value + items[idx].value
                new_c = state.capacity - items[idx].weight
                taken_yes, value_yes, count = \
                    dfs(State(new_c, new_v, new_b), items, idx + 1, taken + [1], capacity, val_weight, count + 1, value_max, taken_max)
            else:
                taken_yes, value_yes = [i for i in taken], state.value
        else:
            taken_yes, value_yes = [i for i in taken], state.value
        
        # Call dfs on next item
        new_b = bound_calc(items, taken + [0], capacity, val_weight)
        # if new_b > state.value:
        if new_b > value_max:
            taken_no, value_no, count = dfs(state, items, idx + 1, taken + [0], capacity, val_weight, count + 1, value_max, taken_max)
        else:
            taken_no, value_no = [i for i in taken], state.value
            
        # Expand taken array to match size of items        
        taken_yes = taken_yes + [0 for _ in range(len(items) - len(taken_yes))]
        taken_no = taken_no + [0 for _ in range(len(items) - len(taken_no))]
        
        t, v = (taken_yes, value_yes) if value_yes >= value_no else (taken_no, value_no)
        
        if v > value_max:
            value_max = v
            taken_max = [i for i in t]
        
        return t, v, count
      
    if BB_LR_DFS:
        print('len of items is: ', len(items))
        bound = sum([i.value for i in items])
        state = State(capacity, 0, bound)
        
        # Sort items by v / w
        value_weight = [(i, (i.value/i.weight)) for i in items]
        value_weight = sorted(value_weight, key = lambda X: X[1], reverse=True)
        items = [i[0] for i in value_weight]
        
        t0 = time.time()
        
        # Perform DFS
        taken, value, count = dfs(state, items, 0, [], capacity, value_weight, 1, value_max, taken_max)
        print("Number of nodes visited: ", count)

        total = time.time() - t0
        print("It Took: ", total, " Seconds")
        
        # Reconstruct taken array
        taken_idx = [(i.index, t) for i, t in zip(items, taken)]
        taken_idx = sorted(taken_idx, key=lambda X: X[0])
        taken = [t for _, t in taken_idx]
        
        print("Total Value Sum: ", sum([i.value for i in items if taken[i.index] == 1]))
        
        output_data = str(value) + ' ' + str(0) + '\n'
        output_data += ' '.join(map(str, taken))
        return output_data
    
    '''DFS Iterative Solution'''
    
    def bound_cap(taken, items, value_weight, capacity):
        bound = 0
        original_capacity = capacity
        original_taken = [i for i in taken]
        taken = taken + [1 for _ in range(len(items) - len(taken))] 
        for i in range(len(items)):
            item, vw = value_weight[i]
            if taken[i] == 1:
                if item.weight < capacity:
                    bound += item.value
                    capacity -= item.weight
                elif capacity > 0:
                    bound += (vw*capacity)           
                    break  
        cap = original_capacity - sum([items[i].weight for i in range(len(original_taken)) if original_taken[i] == 1])
        return bound, cap
    
    if BB_LR_DFS_ITER:
        
        t0 = time.time()
        
        # Sort items by v / w
        value_weight = [(i, (i.value/i.weight)) for i in items]
        value_weight = sorted(value_weight, key = lambda X: X[1], reverse=True)
        items = [i[0] for i in value_weight]
        
        taken_max = []
        max_b = 0
        
        q = Queue()
        q.put([1])
        q.put([0])
        
        count = 0
        
        # While Queue
        while not q.empty():
            count += 1
            s = q.get()
            
            # Check if state s is over capacity
            cap_s = capacity - sum([items[i].weight for i in range(len(s)) if s[i] == 1])
            if cap_s < 1:
                continue
            
            s_val = sum([items[i].value for i in range(len(s)) if s[i] == 1])
            
            # If state s better than the best so far than update the best
            if max_b < s_val:
                taken_max = [i for i in s]
                max_b = s_val
                
            # Get bound of state s
            b_yes, c_yes = bound_cap(s + [1], items, value_weight, capacity)
            b_no, c_no = bound_cap(s + [0], items, value_weight, capacity)
             
            
            # If all items not evaluated keep going
            if len(s) + 1 < len(items):
                # Include state with or without item if bound is greater than current max and there remains capacity
                if b_yes > max_b and c_yes > 0:
                        q.put(s + [1])
                if b_no > max_b and c_no > 0:
                    q.put(s + [0])

        
        # Reconstruct taken array
        taken_max = taken_max + [0 for _ in range(len(items) - len(taken_max))] 
        taken_idx = [(i.index, t) for i, t in zip(items, taken_max)]
        taken_idx = sorted(taken_idx, key=lambda X: X[0])
        taken_max = [t for _, t in taken_idx]
        
        items = sorted(items, key = lambda i: i.index)
        print("Total Value Sum: ", sum([i.value for i in items if taken_max[i.index] == 1]))
        print("Original Capacity: ", capacity,  "Capacity Left: ", capacity - sum([items[i].weight for i in range(len(taken_max)) if taken_max[i] == 1]) )
        
        total = time.time() - t0
        print("It Took: ", total, " Seconds")
        print(f'Nodes visited: {count}')
        
        taken, value = [i for i in taken_max], max_b 
        
        output_data = str(value) + ' ' + str(0) + '\n'
        output_data += ' '.join(map(str, taken))
        return output_data
    
    '''DFS Exhaustive Solution'''
    
    def dfs_ex(state, items, idx, taken, capacity):
        if idx == len(items) - 1 or len(taken) == len(items):
            return [i for i in taken], state.value
        
        new_c = state.capacity - items[idx].weight
        new_v = state.value + items[idx].value
        new_bound = state.bound
        state_yes = State(new_c, new_v, new_bound)
        
        if new_c > 0:
            taken_yes, value_yes = dfs_ex(state_yes, items, idx + 1, taken + [1], capacity)
        else:
            taken_yes, value_yes = taken, state.value
        
        new_c = state.capacity
        new_v = state.value
        new_bound = state.bound - items[idx].value
        state_no = State(new_c, new_v, new_bound)
        taken_no, value_no = dfs_ex(state_no, items, idx + 1, taken + [0], capacity)

        taken_yes = taken_yes + [0 for _ in range(len(items) - len(taken_yes))]
        taken_no = taken_no + [0 for _ in range(len(items) - len(taken_no))]
        
        if value_yes >= value_no:
            return taken_yes, value_yes
        else:
            return taken_no, value_no
        
    if BB_EX:
        bound = sum([i.value for i in items])
        state = State(capacity, 0, bound)
        taken, value = dfs_ex(state, items, 0, [], capacity) 
    
        output_data = str(value) + ' ' + str(1) + '\n'
        output_data += ' '.join(map(str, taken))
        return output_data
        
    
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

