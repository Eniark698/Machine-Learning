import time

def cpu_bound_task():
    # Example task: sum of squares
    sum = 0
    for i in range(1000000000):
        sum += i * i

start = time.time()

cpu_bound_task()

end = time.time()
print(f"Python Elapsed time: {end - start} seconds")
