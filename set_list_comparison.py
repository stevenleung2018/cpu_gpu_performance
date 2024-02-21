# Import the time and random modules
import time
import random
import cupy as np
import multiprocessing as mp
from multiprocessing import Manager



def worker(args):
    target, my_worker_list = args
    return target, target in my_worker_list


def main():

    # Generate a set and a list of 1000000 unique random positive integers
    my_array = np.random.choice(np.arange(1, 100000001), size=1000000, replace=False)
    my_set = set(np.asnumpy(my_array))
    my_list = list(np.asnumpy(my_array))

    # Choose a random number to look for
    n_iter = 10000
    targets_np = np.random.randint(low=1, high=100000001, size=n_iter)
    targets = list(np.asnumpy(targets_np))

    # Repeat the "in" operation 100000 times for both the set and the list
    set_true_count = 0

    start = time.time()

    for target in targets:        # Measure the time for the set
        if target in my_set:
            set_true_count += 1

    end = time.time()
    set_time = end - start

    print(f"Set time: {set_time:.4f} seconds. {set_true_count} numbers are found.")

    list_true_count = 0
    # Measure the time for the list
    start = time.time()
    for target in targets:
        if target in my_list:
            list_true_count += 1

    end = time.time()
    list_time = end - start

    print(f"List time: {list_time:.4f} seconds. {list_true_count} numbers are found.")

    # Repeat the "in" operation 100000 times for the numpy array
    array_true_count = 0
    # array_matched_list = []
    # Measure the time for the numpy array
    start = time.time()
    for target in targets_np:
        if np.isin(target, my_array):
            # array_matched_list.append(target)
            array_true_count += 1

    end = time.time()
    array_time = end - start

    print(f"Array time: {array_time:.4f} seconds. {array_true_count} numbers are found.")

    # Numpy array vectorized
    start = time.time()
    mask = np.isin(targets_np, my_array)
    array_vec_true_count = np.count_nonzero(mask)
    end = time.time()
    array_vec_time = end - start
    # print("len(mask):", len(mask))
    array_vec_matched_list = targets_np[mask]

    print(f"Array vectorized time: {array_vec_time:.4f} seconds. {array_vec_true_count} numbers are found.")

    print(f"Matched numbers: {array_vec_matched_list}")

    pool = mp.Pool(mp.cpu_count() - 1)

    manager = Manager()
    my_worker_list = manager.list(my_list)

    start = time.time()
    results = pool.map(worker, [(target, my_worker_list) for target in targets])
    end = time.time()
    mp_list_time = end - start
    # print(f"results: {results}")
    mp_matched_list = [result[0] for result in results if result[1]]
    print(f"mp_matched_list: {mp_matched_list}")
    mp_list_true_count = sum([result[1] for result in results])

    # Compare the execution times



    print(f"Multiprocessing list time: {mp_list_time:.4f} seconds. {mp_list_true_count} numbers are found.")
    if set_time == 0:
        print("Set is much faster than list and array.")
    else:
        print(f"Set is {list_time / set_time:.2f} times faster than list.")
        print(f"Set is {array_time / set_time:.2f} times faster than array.")
        print(f"Set is {mp_list_time / set_time:.2f} times faster than multiprocessing list.")


if __name__ == '__main__':
    # mp.freeze_support()
    main()
