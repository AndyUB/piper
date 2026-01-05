import ray
import threading

@ray.remote
class ConcurrentActor:
    def __init__(self):
        print(f"Thread {threading.get_native_id()} initializing actor")
        self.counter = 0

    def increment(self):
        print(f"Thread {threading.get_native_id()} incrementing actor")
        self.counter += 1
        return self.counter
    
    def increment2(self):
        print(f"Thread {threading.get_native_id()} incrementing actor")
        self.counter += 1
        return self.counter

    def get_counter(self):
        return self.counter

actor = ConcurrentActor.options(max_concurrency=100).remote()

def increment_actor(actor):
    print(f"Thread {threading.get_native_id()} invoking increment: {ray.get(actor.increment.remote())}")

def increment_actor2(actor):
    print(f"Thread {threading.get_native_id()} invoking increment2: {ray.get(actor.increment2.remote())}")

print(f"Thread {threading.get_native_id()} laucnhing threads: {ray.get(actor.get_counter.remote())}")

threads = []
for i in range(100):
    if i % 2 == 0:
        thread = threading.Thread(target=increment_actor, args=(actor,))
    else:
        thread = threading.Thread(target=increment_actor2, args=(actor,))
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()

print(f"Thread {threading.get_native_id()} final counter: {ray.get(actor.get_counter.remote())}")