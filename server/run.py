from dianpin import Dianpin
import time

start_time = time.time()
test = Dianpin()
test.model_built()
print(test.final_predict())

print(time.time()-start_time)