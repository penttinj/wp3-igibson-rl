from envs.wp3_test_env import Wp3TestEnv
import gym_gibson
import tensorflow as tf

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
print('optimizer=', optimizer)