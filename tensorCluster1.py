import tensorflow as tf

cluster = tf.train.ClusterSpec({
    "worker": [
        "localhost:12345",  # job:worker/task:0
    ],
    "ps": [
        "localhost:12346", # job:ps/task:0
    ]
})

server = tf.train.Server(cluster, job_name="worker", task_index=0)

server.join()