import random
import math
import hashlib
from multiprocessing import Pool, Manager
import pandas as pd
import numpy as np
import time

FUNCTION_MAP = {
    'fibonacci': 2,
    'padovan': 3,
    'pell': 2,
    'linear_recursive': 2,
    'triangular_sequence_randomized': 2,
    'tribonacci': 5,
    'fib_minus_const': 1,
    'alternating_add_subtract': 2,
    'generate_sequence': 2,
    'arithmetic': 3,
    'step_up': 2,
    'geometric': 2,
    'coeff_recursion': 4,
    'alternating_sequence': 5
}

TIER_WEIGHTS = {
    1: 0.1, 
    2: 0.3, 
    3: 0.5559,
    4: 0.0001
}

def triangular_sequence_randomized(n=7):
    base = random.randint(1, 20)  
    sequence = [int((base + i) * (base + i + 1) / 2) for i in range(n)]
    return sequence[:-1], sequence[-1]

def alternating_add_subtract(n=7):
    start = random.randint(1, 100)
    sequence = [start]
    add_factor = random.randint(1, 10)  
    sub_factor = random.randint(1, 10)
    
    for i in range(1, n):
        if i % 2 == 0:
            sequence.append(sequence[-1] + add_factor)
        else:
            sequence.append(sequence[-1] - sub_factor)
    
    return sequence[:-1], sequence[-1]

def generate_sequence(n=7):
    sequence = [random.randint(1, 20) for _ in range(3)]
    for i in range(3, n):
        next_value = sequence[i-3] + sequence[i-1]
        sequence.append(next_value)
    
    return sequence[:-1], sequence[-1]

def arithmetic(n=7):
    base = random.randint(0, 100)
    diff = random.randint(-50, 50)
    sequence = [base + i * diff for i in range(n)]
    return sequence[:-1], sequence[-1]

def step_up(n=6):
    start = random.randint(1, 75)
    step = random.randint(1, 5)
    seq = [start]
    for i in range(n):
        seq.append(seq[-1] + (step * (i+1)))
    return seq[:-1], seq[-1]

def geometric(n=7):
    base = random.randint(1, 50)  
    ratio = random.randint(-5, 5)  

    sequence = [base * (ratio ** i) for i in range(n)]  
    # In this form, geometric might produce 0 or negative values if ratio is 0 or negative.
    # As implemented, it returns the full sequence, so you may want to handle this carefully.
    return sequence

def fibonacci(n=6):
    seq = [random.randint(1, 40), random.randint(1, 40)]
    while len(seq) < n + 1:
        seq.append(seq[-1] + seq[-2])
    return seq[:-1], seq[-1]

def padovan(n=6):
    seq = [random.randint(1, 20), random.randint(1, 20), random.randint(1, 20)]
    while len(seq) < n + 1:
        seq.append(seq[-2] + seq[-3])
    return seq[:-1], seq[-1]

def pell(n=6):
    seq = [random.randint(1, 50), random.randint(1, 50)]
    while len(seq) < n + 1:
        seq.append(2 * seq[-1] + seq[-2])
    return seq[:-1], seq[-1]

def linear_recursive(n=6):
    a = random.randint(2, 5)
    b = random.randint(1, 20)
    x0 = random.randint(1, 50)
    
    seq = [x0]
    for _ in range(n):
        seq.append(a * seq[-1] + b)
    return seq[:-1], seq[-1]

def tribonacci(n=6):
    a, b, c = random.randint(1,50), random.randint(1,50), random.randint(1,50)
    seq = [a, b, c]
    
    for _ in range(n - 2):
        seq.append(seq[-1] + seq[-2] + seq[-3])

    return seq[:-1], seq[-1]

def coeff_recursion(n=6):
    alpha = random.randint(2, 5)
    beta = random.randint(1, 3)
    a, b = random.randint(1,10), random.randint(1,10)
    
    seq = [a, b]
    for _ in range(n-1):
        seq.append(alpha*seq[-1] + beta*seq[-2])
    return seq[:-1], seq[-1]

def fib_minus_const(n=6):
    c = random.randint(1, 5)
    a = random.randint(1, 10)
    b = random.randint(1, 10)
    seq = [a, b]
    while len(seq) < n + 1:
        seq.append(seq[-1] + seq[-2] - c)
    return seq[:-1], seq[-1]

def alternating_sequence(n=7):
    base = random.randint(1, 50)
    add_factor = random.randint(1, 20)
    mult_factor = random.randint(2, 5)

    sequence = [base]
    for i in range(1, n):
        if i % 2 == 1:
            sequence.append(sequence[-1] + add_factor)
        else:
            sequence.append(sequence[-1] * mult_factor)
    
    return sequence[:-1], sequence[-1]

def get_function_weights():
    functions = list(FUNCTION_MAP.keys())
    tiers = [FUNCTION_MAP[f] for f in functions]
    weights = np.array([TIER_WEIGHTS[t] for t in tiers])
    return weights / weights.sum()

def generate_sample(_):
    func_name = np.random.choice(
        list(FUNCTION_MAP.keys()),
        p=get_function_weights()
    )
    func = globals()[func_name]
    
    try:
        seq_and_target = func()
        
        # Some generators (like 'geometric') might return just a list,
        # so we handle that by ensuring we can unpack consistently.
        if isinstance(seq_and_target, tuple):
            seq, target = seq_and_target
        else:
            # If a function returns a single sequence,
            # pick the last item for the target
            seq = seq_and_target[:-1]
            target = seq_and_target[-1]
        
        seq_hash = hashlib.sha256((str(seq) + str(target)).encode()).hexdigest()
        return (seq, target, func_name, seq_hash)
    except:
        return None

def generate_dataset(target_size=50000):
    manager = Manager()
    seen_hashes = manager.dict()
    dataset = manager.list()
    
    def collect_results(result):
        # result: (seq, target, func_name, seq_hash)
        if result and result[-1] not in seen_hashes:
            seen_hashes[result[-1]] = True
            dataset.append({
                'sequence': result[0],
                'target': result[1],
                'function': result[2],
                'tier': FUNCTION_MAP[result[2]]  # <-- include tier here
            })

    with Pool(processes=8) as pool:
        tasks_submitted = 0
        batch_size = 1000
        
        while len(dataset) < target_size:
            remaining = target_size - len(dataset)
            current_batch = min(batch_size, remaining)
            
            for _ in range(current_batch):
                pool.apply_async(
                    generate_sample,
                    args=(None,),
                    callback=collect_results
                )
                tasks_submitted += 1
                
            print(f"Submitted {tasks_submitted} tasks, collected {len(dataset)} samples")
            time.sleep(0.5)
            
    df = pd.DataFrame(list(dataset))
    df.to_csv('sequence_dataset_curriculum.csv', index=False)
    return df

if __name__ == "__main__":
    dataset = generate_dataset()
    print(f"Final dataset size: {len(dataset)}")
