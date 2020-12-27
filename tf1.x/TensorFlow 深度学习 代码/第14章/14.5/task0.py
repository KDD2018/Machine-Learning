import tensorflow as tf

c=tf.constant(1)

# 创建一个包含两个任务的集群，其中一个任务运行在本地2222端口，另一个运行在
# 本地2223端口，，ClusterSpec类构造函数原型__init__(self,cluster)
cluster = tf.train.ClusterSpec({"local":["localhost:2222","localhost:2223"]})

# 通过cluster使用Server类生成server
# Server类构造函数__init__(self,server_or_cluster_def,job_name,task_index,protocol,config,start)
server = tf.train.Server(cluster, job_name="local",task_index=0)

# 将server.target传递给Session可以配置会话使用TensorFlow集群中的资源
with tf.Session(server.target,config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(c))
    server.join()


'''打印的部分内容
I tensorflow/core/common_runtime/gpu/gpu_device.cc:906] DMA: 0 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:916] 0:   Y 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] 
Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 850M, 
pci bus id: 0000:01:00.0)
I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:200] 
Initialize GrpcChannelCache for job local -> {0 -> localhost:2222, 1 -> localhost:2223}
I tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc:221] Started server with target: 
grpc://localhost:2222
I tensorflow/core/distributed_runtime/master.cc:193] CreateSession still waiting for response 
from worker: /job:local/replica:0/task:1
I tensorflow/core/distributed_runtime/master.cc:193] CreateSession still waiting for response 
from worker: /job:local/replica:0/task:1
...
'''