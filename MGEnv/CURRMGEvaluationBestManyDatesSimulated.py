import math
import numpy as np
import tensorflow as tf
import copy
import gym
import pandas as pd
import matplotlib.pyplot as plt
from MGEnvOneDevicesHourly import MGEnv
from MGEnvTwoDevicesHourly import MGEnvTwo
from tempfile import TemporaryFile

NUMBER_EPISODES = 3000 #3000
GAMMA = 0.95
EPOCS = 5
INITIAL_TRESHOLD = 150
NUMBER_OF_SOURCETASKS = 1
THREE_COLUMNS = False
FOUR_COLUMNS = True
NUMBER_COLUMS = 3
env = MGEnv("Data/9982HourlyDevUSEGEN-0117.xlsx")
STATE_SPAC = env.state_space
ACTION_SPAC = env.action_space

class ACModel:
    def __init__(self,scope:str,state_space,action_space, hidden_layers = 200, temp = 1):
       
        with tf.variable_scope(scope):

            self.states = tf.placeholder(dtype=tf.float32,shape=[None,state_space],name = "States")
            #COLUMN 1
            with tf.variable_scope('Policy_Estimator'):
                l11 = tf.layers.dense(inputs=self.states,units = hidden_layers, activation = None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="PolicyC1_L1")
                out11 = tf.nn.relu(l11)
                l21 = tf.layers.dense(inputs=out11,units = hidden_layers, activation = None, kernel_initializer=tf.contrib.layers.xavier_initializer(),name="PolicyC1_L2")
                out21 = tf.nn.relu(l21)

                l31 = tf.layers.dense(inputs=tf.divide(out21,temp), units=action_space, kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=None, name="ActionProbsC1_")

                self.action_probs1 = tf.nn.softmax(l31)
                
            with tf.variable_scope('Value_Estimator'):
                l11 = tf.layers.dense(inputs=self.states, units = hidden_layers, activation=None, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),name="ValueC1_L1")
                out11 = tf.nn.relu(l11)
                l21 = tf.layers.dense(inputs = out11, units = hidden_layers, activation=None,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),name="ValueC1_L2")
                out21 = tf.nn.relu(l21)

                l31 = tf.layers.dense(inputs = out21, units = 1, activation = None, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name="ValueC1_L3")
                self.value1 = l31
            #COLUMN 2
            with tf.variable_scope('Policy_Estimator'):
                l12 = tf.layers.dense(inputs=self.states,units = hidden_layers, activation = None, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),name="PolicyC2_L1")
                out12 = tf.nn.relu(l12)

                l22 = tf.layers.dense(inputs=out12,units = hidden_layers, activation = None,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),name="PolicyC2_L2")
                
                U12 = tf.layers.dense(inputs=out11, units = hidden_layers, activation= None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="UPC2_L2", bias_initializer=tf.contrib.layers.xavier_initializer())

                out22 = tf.nn.relu(tf.add(l22,U12))
                
                l32 = tf.layers.dense(inputs=tf.divide(out22,temp), units=2, activation=None, name="ActionProbsC2_", kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

                U32 = tf.layers.dense(inputs=tf.divide(out21,temp), units=2, activation=None, name="UPC2_L3",kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.contrib.layers.xavier_initializer())
                
                self.action_probs2 = tf.nn.softmax(tf.add(l32,U32))
                
            with tf.variable_scope('Value_Estimator'):
                l12 = tf.layers.dense(inputs=self.states, units = hidden_layers, activation=None,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),name="ValueC2_L1")
                out12 = tf.nn.relu(l12)

                l22 = tf.layers.dense(inputs=out12, units = hidden_layers, activation=None,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),name="ValueC2_L2")
                
                U22 = tf.layers.dense(inputs=out11, units = hidden_layers, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(),name="UVC2_L2",bias_initializer=tf.contrib.layers.xavier_initializer())

                out22 = tf.nn.relu(tf.add(l22,U22))

                l32 = tf.layers.dense(inputs = out22, units = 1, activation = None, name="ValueC2_L3",kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

                U32 = tf.layers.dense(inputs = out21, units = 1, activation = None, name="UC2_L3",kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.contrib.layers.xavier_initializer())
                

                self.value2 = (tf.add(l32,U32))
            #COLUMN 3
            with tf.variable_scope('Policy_Estimator'):
                l13 = tf.layers.dense(inputs=self.states,units = hidden_layers, activation = None, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),name="PolicyC3_L1")
                out13 = tf.nn.relu(l13)

                l23 = tf.layers.dense(inputs=out13,units = hidden_layers, activation = None,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),name="PolicyC3_L2")
                
                U231 = tf.layers.dense(inputs=out11, units = hidden_layers, activation= None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="UPC3_L2-1", bias_initializer=tf.contrib.layers.xavier_initializer())
                U232 = tf.layers.dense(inputs=out12, units = hidden_layers, activation= None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="UPC3_L2-2", bias_initializer=tf.contrib.layers.xavier_initializer())
                
                out23 = tf.nn.relu(tf.add(l23,tf.add(U231,U232)))
                
                l33 = tf.layers.dense(inputs=tf.divide(out23,temp), units=2, activation=None, name="ActionProbsC3_", kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
                
                U331 = tf.layers.dense(inputs=tf.divide(out21,temp), units=2, activation=None, name="UPC3_L3-1",kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.contrib.layers.xavier_initializer())
                U332 = tf.layers.dense(inputs=tf.divide(out22,temp), units=2, activation=None, name="UPC3_L3-2",kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.contrib.layers.xavier_initializer())
                
                self.action_probs3 = tf.nn.softmax(tf.add(l33,tf.add(U331,U332)))
                
            with tf.variable_scope('Value_Estimator'):
                l13 = tf.layers.dense(inputs=self.states, units = hidden_layers, activation=None,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),name="ValueC3_L1")
                out13 = tf.nn.relu(l13)

                l23 = tf.layers.dense(inputs=out13, units = hidden_layers, activation=None,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),name="ValueC3_L2")
                
                U231 = tf.layers.dense(inputs=out11, units = hidden_layers, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(),name="UVC3_L2-1",bias_initializer=tf.contrib.layers.xavier_initializer())
                U232 = tf.layers.dense(inputs=out12, units = hidden_layers, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(),name="UVC3_L2-2",bias_initializer=tf.contrib.layers.xavier_initializer())
                
                out23 = tf.nn.relu(tf.add(l23,tf.add(U231,U232)))

                l33 = tf.layers.dense(inputs = out23, units = 1, activation = None, name="ValueC3_L3",kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

                U331 = tf.layers.dense(inputs = out21, units = 1, activation = None, name="UC3_L3-1",kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.contrib.layers.xavier_initializer())
                U332 = tf.layers.dense(inputs = out22, units = 1, activation = None, name="UC3_L3-2",kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.contrib.layers.xavier_initializer())

                self.value3 = (tf.add(l33,tf.add(U331,U332)))
            #COLUMN 4
            with tf.variable_scope('Policy_Estimator'):
                l14 = tf.layers.dense(inputs=self.states,units = hidden_layers, activation = None, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),name="PolicyC4_L1")
                out14 = tf.nn.relu(l14)

                l24 = tf.layers.dense(inputs=out14,units = hidden_layers, activation = None,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),name="PolicyC4_L2")
                
                U241 = tf.layers.dense(inputs=out11, units = hidden_layers, activation= None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="UPC4_L2-1", bias_initializer=tf.contrib.layers.xavier_initializer())
                U242 = tf.layers.dense(inputs=out12, units = hidden_layers, activation= None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="UPC4_L2-2", bias_initializer=tf.contrib.layers.xavier_initializer())
                U243 = tf.layers.dense(inputs=out13, units = hidden_layers, activation= None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="UPC4_L2-3", bias_initializer=tf.contrib.layers.xavier_initializer())
                
                out24 = tf.nn.relu(tf.add(l24,tf.add_n([U241,U242,U243])))
                
                l34 = tf.layers.dense(inputs=tf.divide(out24,temp), units=2, activation=None, name="ActionProbsC4_", kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
                
                U341 = tf.layers.dense(inputs=tf.divide(out21,temp), units=2, activation=None, name="UPC4_L3-1",kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.contrib.layers.xavier_initializer())
                U342 = tf.layers.dense(inputs=tf.divide(out22,temp), units=2, activation=None, name="UPC4_L3-2",kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.contrib.layers.xavier_initializer())
                U343 = tf.layers.dense(inputs=tf.divide(out23,temp), units=2, activation=None, name="UPC4_L3-3",kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.contrib.layers.xavier_initializer())
                
                self.action_probs4 = tf.nn.softmax(tf.add(l34,tf.add_n([U341,U342,U343])))
                
            with tf.variable_scope('Value_Estimator'):
                l14 = tf.layers.dense(inputs=self.states, units = hidden_layers, activation=None,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),name="ValueC4_L1")
                out14 = tf.nn.relu(l14)

                l24 = tf.layers.dense(inputs=out14, units = hidden_layers, activation=None,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),name="ValueC4_L2")
                
                U241 = tf.layers.dense(inputs=out11, units = hidden_layers, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(),name="UVC4_L2-1",bias_initializer=tf.contrib.layers.xavier_initializer())
                U242 = tf.layers.dense(inputs=out12, units = hidden_layers, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(),name="UVC4_L2-2",bias_initializer=tf.contrib.layers.xavier_initializer())
                U243 = tf.layers.dense(inputs=out13, units = hidden_layers, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(),name="UVC4_L2-3",bias_initializer=tf.contrib.layers.xavier_initializer())
                
                out24 = tf.nn.relu(tf.add(l24,tf.add_n([U241,U242,U243])))

                l34 = tf.layers.dense(inputs = out24, units = 1, activation = None, name="ValueC4_L3",kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

                U341 = tf.layers.dense(inputs = out21, units = 1, activation = None, name="UC4_L3-1",kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.contrib.layers.xavier_initializer())
                U342 = tf.layers.dense(inputs = out22, units = 1, activation = None, name="UC4_L3-2",kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.contrib.layers.xavier_initializer())
                U343 = tf.layers.dense(inputs = out23, units = 1, activation = None, name="UC4_L3-3",kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.contrib.layers.xavier_initializer())

                self.value4 = (tf.add(l34,tf.add_n([U341,U342,U343])))
            #COLUMN 5
            with tf.variable_scope('Policy_Estimator'):
                l15 = tf.layers.dense(inputs=self.states,units = hidden_layers, activation = None, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),name="PolicyC5_L1")
                out15 = tf.nn.relu(l15)

                l25 = tf.layers.dense(inputs=out15,units = hidden_layers, activation = None,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),name="PolicyC5_L2")
                
                U251 = tf.layers.dense(inputs=out11, units = hidden_layers, activation= None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="UPC5_L2-1", bias_initializer=tf.contrib.layers.xavier_initializer())
                U252 = tf.layers.dense(inputs=out12, units = hidden_layers, activation= None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="UPC5_L2-2", bias_initializer=tf.contrib.layers.xavier_initializer())
                U253 = tf.layers.dense(inputs=out13, units = hidden_layers, activation= None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="UPC5_L2-3", bias_initializer=tf.contrib.layers.xavier_initializer())
                U254 = tf.layers.dense(inputs=out14, units = hidden_layers, activation= None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="UPC5_L2-4", bias_initializer=tf.contrib.layers.xavier_initializer())
                
                out25 = tf.nn.relu(tf.add(l25,tf.add_n([U251,U252,U253,U254])))
                
                l35 = tf.layers.dense(inputs=tf.divide(out25,temp), units=2, activation=None, name="ActionProbsC5_", kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
                
                U351 = tf.layers.dense(inputs=tf.divide(out21,temp), units=2, activation=None, name="UPC5_L3-1",kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.contrib.layers.xavier_initializer())
                U352 = tf.layers.dense(inputs=tf.divide(out22,temp), units=2, activation=None, name="UPC5_L3-2",kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.contrib.layers.xavier_initializer())
                U353 = tf.layers.dense(inputs=tf.divide(out23,temp), units=2, activation=None, name="UPC5_L3-3",kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.contrib.layers.xavier_initializer())
                U354 = tf.layers.dense(inputs=tf.divide(out24,temp), units=2, activation=None, name="UPC5_L3-4",kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.contrib.layers.xavier_initializer())
                
                self.action_probs5 = tf.nn.softmax(tf.add(l35,tf.add_n([U351,U352,U353,U354])))
                
            with tf.variable_scope('Value_Estimator'):
                l15 = tf.layers.dense(inputs=self.states, units = hidden_layers, activation=None,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),name="ValueC5_L1")
                out15 = tf.nn.relu(l15)

                l25 = tf.layers.dense(inputs=out15, units = hidden_layers, activation=None,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),name="ValueC5_L2")
                
                U251 = tf.layers.dense(inputs=out11, units = hidden_layers, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(),name="UVC5_L2-1",bias_initializer=tf.contrib.layers.xavier_initializer())
                U252 = tf.layers.dense(inputs=out12, units = hidden_layers, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(),name="UVC5_L2-2",bias_initializer=tf.contrib.layers.xavier_initializer())
                U253 = tf.layers.dense(inputs=out13, units = hidden_layers, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(),name="UVC5_L2-3",bias_initializer=tf.contrib.layers.xavier_initializer())
                U254 = tf.layers.dense(inputs=out14, units = hidden_layers, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(),name="UVC5_L2-4",bias_initializer=tf.contrib.layers.xavier_initializer())

                out25 = tf.nn.relu(tf.add(l25,tf.add_n([U251,U252,U253,U254])))

                l35 = tf.layers.dense(inputs = out25, units = 1, activation = None, name="ValueC5_L3",kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

                U351 = tf.layers.dense(inputs = out21, units = 1, activation = None, name="UC5_L3-1",kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.contrib.layers.xavier_initializer())
                U352 = tf.layers.dense(inputs = out22, units = 1, activation = None, name="UC5_L3-2",kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.contrib.layers.xavier_initializer())
                U353 = tf.layers.dense(inputs = out23, units = 1, activation = None, name="UC5_L3-3",kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.contrib.layers.xavier_initializer())
                U354 = tf.layers.dense(inputs = out24, units = 1, activation = None, name="UC5_L3-4",kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.contrib.layers.xavier_initializer())

                self.value5 = (tf.add(l35,tf.add_n([U351,U352,U353,U354])))
            
            self.act_stochastic1 = tf.multinomial(tf.log(self.action_probs1), num_samples = 1)
            self.act_stochastic1 = tf.reshape(self.act_stochastic1, shape=[-1])

            self.act_deterministic1 = tf.argmax(self.action_probs1, axis = 1)


            self.act_stochastic2 = tf.multinomial(tf.log(self.action_probs2), num_samples = 1)
            self.act_stochastic2 = tf.reshape(self.act_stochastic2, shape=[-1])

            self.act_deterministic2 = tf.argmax(self.action_probs2, axis = 1)


            self.act_stochastic3 = tf.multinomial(tf.log(self.action_probs3), num_samples = 1)
            self.act_stochastic3 = tf.reshape(self.act_stochastic3, shape=[-1])
            
            self.act_deterministic3 = tf.argmax(self.action_probs3, axis = 1)


            self.act_stochastic4 = tf.multinomial(tf.log(self.action_probs4), num_samples = 1)
            self.act_stochastic4 = tf.reshape(self.act_stochastic4, shape=[-1])
            self.act_deterministic4 = tf.argmax(self.action_probs4, axis = 1)

            self.act_stochastic5 = tf.multinomial(tf.log(self.action_probs5), num_samples = 1)
            self.act_stochastic5 = tf.reshape(self.act_stochastic5, shape=[-1])
            self.act_deterministic5 = tf.argmax(self.action_probs5, axis = 1)
            
            self.scope = tf.get_variable_scope().name

            
    def act(self, states, column = 1, stochastic = True):
        sess = tf.get_default_session()
        if column == 1: 
            if stochastic:
                return sess.run([self.act_stochastic1,self.value1], feed_dict={self.states:states})
            else:
                return sess.run([self.act_deterministic1,self.value1], feed_dict={self.states:states})
        elif column == 2: 
            if stochastic:
                return sess.run([self.act_stochastic2,self.value2], feed_dict={self.states:states})
            else:
                return sess.run([self.act_deterministic2,self.value2], feed_dict={self.states:states})
        elif column == 3: 
            if stochastic:
                return sess.run([self.act_stochastic3,self.value3], feed_dict={self.states:states})
            else:
                return sess.run([self.act_deterministic3,self.value3], feed_dict={self.states:states})
        elif column == 4: 
            if stochastic:
                return sess.run([self.act_stochastic4,self.value4], feed_dict={self.states:states})
            else:
                return sess.run([self.act_deterministic4,self.value4], feed_dict={self.states:states})
        elif column == 5: 
            if stochastic:
                return sess.run([self.act_stochastic5,self.value5], feed_dict={self.states:states})
            else:
                return sess.run([self.act_deterministic5,self.value5], feed_dict={self.states:states})

    def get_action_probs(self,states, column = 1):
        sess = tf.get_default_session()
        if column == 1:
            return sess.run(self.action_probs1, feed_dict = {self.states:states})
        elif column == 2:
            return sess.run(self.action_probs2, feed_dict = {self.states:states})
        elif column == 3:
            return sess.run(self.action_probs3, feed_dict = {self.states:states})
        elif column == 4:
            return sess.run(self.action_probs4, feed_dict = {self.states:states})
        elif column == 5:
            return sess.run(self.action_probs5, feed_dict = {self.states:states})
    
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

           
 
        

class PPOTrain:
    def __init__(self,Policy, Old_Policy, column = 1, gamma = 0.95, clip_value = 0.2, c_1 = 1, c_2=0.01):
        self.Policy = Policy
        self.Old_Policy = Old_Policy
        self.gamma = gamma

        columnstr = "C"+str(column)+"_"


        trainable = self.Policy.get_trainable_variables()
        pi_trainable = [var for var in trainable if columnstr in var.name]

        old_trainable = self.Old_Policy.get_trainable_variables()
        old_pi_trainable = [var for var in old_trainable if columnstr in var.name]
        
        with tf.variable_scope('Assign_Operator'):
            self.assign_ops = []
            for v_old, v in zip(old_pi_trainable, pi_trainable):
                self.assign_ops.append(tf.assign(v_old,v))
        
        with tf.variable_scope('Inputs_Train'):
            self.actions = tf.placeholder(tf.int32, [None], "Actions")
            self.rewards = tf.placeholder(tf.float32, [None], "Rewards")
            self.v_next = tf.placeholder(tf.float32, [None], "Value_Next")
            self.gaes = tf.placeholder(tf.float32, [None], "GAES")

        if column == 1:
            action_probs = self.Policy.action_probs1
            action_probs_old = self.Old_Policy.action_probs1
        elif column == 2:
            action_probs = self.Policy.action_probs2
            action_probs_old = self.Old_Policy.action_probs2
        elif column == 3:
            action_probs = self.Policy.action_probs3
            action_probs_old = self.Old_Policy.action_probs3
        elif column == 4:
            action_probs = self.Policy.action_probs4
            action_probs_old = self.Old_Policy.action_probs4
        elif column == 5:
            action_probs = self.Policy.action_probs5
            action_probs_old = self.Old_Policy.action_probs5

        #Probabilities of actions took with policy
        action_probs = action_probs * tf.one_hot(indices = self.actions, depth = action_probs.shape[1])
        action_probs = tf.reduce_sum(action_probs, axis = 1)

        #Probabilites of action took with old policy
        action_probs_old = action_probs_old * tf.one_hot(indices = self.actions, depth = action_probs_old.shape[1])
        action_probs_old = tf.reduce_sum(action_probs_old, axis = 1)

	    #Clipped Policy
        with tf.variable_scope('Loss/Clip'):
            ratios = tf.exp(tf.log(action_probs)-tf.log(action_probs_old))
            clipped_ratios = tf.clip_by_value(ratios, clip_value_min = 1 - clip_value, clip_value_max = 1 + clip_value)
            loss_clip = tf.minimum(tf.multiply(self.gaes,ratios),tf.multiply(self.gaes,clipped_ratios))
            loss_clip = tf.reduce_mean(loss_clip)
            tf.summary.scalar('loss_clip', loss_clip)

        #Value Function
        with tf.variable_scope('Loss/VF'):
            if column == 1:
                v_preds = self.Policy.value1
            elif column == 2:
                v_preds = self.Policy.value2
            elif column == 3:
                v_preds = self.Policy.value3
            elif column == 4:
                v_preds = self.Policy.value4
            elif column == 5:
                v_preds = self.Policy.value5
            loss_vf = tf.squared_difference(self.rewards + self.gamma * self.v_next, v_preds)
            loss_vf = tf.reduce_mean(loss_vf)
            tf.summary.scalar('loss_vf',loss_vf)
        
        #Entropy
        with tf.variable_scope('Loss/Entropy'):
            if column == 1:
                a_probs = self.Policy.action_probs1
            elif column == 2:
                a_probs = self.Policy.action_probs2
            elif column == 3:
                a_probs = self.Policy.action_probs3
            elif column == 4:
                a_probs = self.Policy.action_probs4
            elif column == 5:
                a_probs = self.Policy.action_probs5
            entropy = - tf.reduce_sum(a_probs * tf.log(tf.clip_by_value(a_probs, 1e-10, 1.0)), axis = 1)
            entropy = tf.reduce_mean(entropy, axis = 0)
            tf.summary.scalar("Entropy",entropy)
        
        with tf.variable_scope('loss'):
            loss = loss_clip - c_1 * loss_vf + c_2 * entropy
            loss = -loss
            tf.summary.scalar('Loss',loss)
        
        self.merged = tf.summary.merge_all()
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001, epsilon = 1e-3)
        self.train_op = optimizer.minimize(loss, var_list = pi_trainable)
    
    def train(self, states, actions, rewards, v_next, gaes):
        sess = tf.get_default_session()
        sess.run([self.train_op], feed_dict={
            self.Policy.states:states,
            self.Old_Policy.states:states,
            self.actions: actions,
            self.rewards: rewards,
            self.v_next: v_next,
            self.gaes: gaes
        })
    def get_summary(self, states, actions, rewards, v_next, gaes):
        sess = tf.get_default_session()
        return sess.run([self.merged], feed_dict = {
            self.Policy.states: states,
            self.Old_Policy.states: states,
            self.actions: actions,
            self.rewards: rewards,
            self.v_next: v_next,
            self.gaes: gaes
        })
    def assign_policy_parameters(self):
        sess = tf.get_default_session()
        sess.run(self.assign_ops)
    #Generative Advantage Estimador
    def get_gaes(self,rewards, values, v_next):
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, v_next, values)]
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes)-1)):
            gaes[t] = gaes[t] + self.gamma * gaes[t+1]
        return gaes
    def print_weight(self, name):
        sess = tf.get_default_session()
        variable = sess.run([self.Policy.fc1_var])
        print(variable)
        

def main():
   
    totalit = 0
    totalit2 = 0
    f_i = 1
    
    evaluationSteps = math.floor(NUMBER_EPISODES/10)
    
    sourceTask = []
    targetReal = []
    targetSimulated = []

    "TARGET REAL"
    targetReal.append(MGEnv("Data/9737HourlyDevUSEGEN-0117.xlsx"))

    "TARGET SIMULATED"
    targetSimulated.append(MGEnv("Data/9631HourlyDevUSEGEN-0117.xlsx"))
    targetSimulated.append(MGEnv("Data/9939HourlyDevUSEGEN-0117.xlsx"))
    targetSimulated.append(MGEnv("Data/9982HourlyDevUSEGEN-0117.xlsx"))
    targetSimulated.append(MGEnv("Data/9121HourlyDevUSEGEN-0117.xlsx"))
    targetSimulated.append(MGEnv("Data/9052HourlyDevUSEGEN-0117.xlsx"))

    
    "TRIPLETS"
    sourceTask.append(MGEnv("Data/9939HourlyDevUSEGEN-0117.xlsx")) ##TARGET SIMULATED
    sourceTask.append(MGEnv("Data/9982HourlyDevUSEGEN-0317.xlsx"))
    sourceTask.append(MGEnv("Data/9942HourlyDevUSEGEN-0218.xlsx"))
    sourceTask.append(MGEnv("Data/9982HourlyDevUSEGEN-0116.xlsx"))

    maxRegret = []
    maxIndex = []
    regrets = []

    previousRegret = 0
    maximumCurriculumRegret = 0

    evaluationTransfer = []
    evaluationTS = []
    evaluationNormal = []
    evaluationNormal2 = []
    evaluationNormal3 = []
    evaluationNormal4 = []
    evaluationNormal5 = []
    evaluationTR = []


    averageRewardsTransfer = []
    averageRewardsTS = []
    averageRewardsNormal = []
    averageRewardsNormal2 = []
    averageRewardsNormal3 = []
    averageRewardsNormal4 = []
    averageRewardsNormal5 = []
    averageRewardsTR = []


    for _ in range(len(targetReal)):
        averageRewardsTransfer.append(np.zeros(NUMBER_EPISODES))
        averageRewardsTS.append(np.zeros(NUMBER_EPISODES))
        averageRewardsNormal.append(np.zeros(NUMBER_EPISODES))
        averageRewardsNormal2.append(np.zeros(NUMBER_EPISODES))
        averageRewardsNormal3.append(np.zeros(NUMBER_EPISODES))
        averageRewardsNormal4.append(np.zeros(NUMBER_EPISODES))
        averageRewardsNormal5.append(np.zeros(NUMBER_EPISODES))
        averageRewardsTR.append(np.zeros(NUMBER_EPISODES))

        evaluationTransfer.append(np.zeros(evaluationSteps))
        evaluationTS.append(np.zeros(evaluationSteps))
        evaluationNormal.append(np.zeros(evaluationSteps))
        evaluationNormal2.append(np.zeros(evaluationSteps))
        evaluationNormal3.append(np.zeros(evaluationSteps))
        evaluationNormal4.append(np.zeros(evaluationSteps))
        evaluationNormal5.append(np.zeros(evaluationSteps))
        evaluationTR.append(np.zeros(evaluationSteps))
        

 
    global THREE_COLUMNS
    global FOUR_COLUMNS
    
    for j in range(EPOCS):
        tf.reset_default_graph()
        with tf.Session() as sess:
            env = targetReal[0]
            state_space = env.state_space
            action_size = env.action_space
            Policy = None
            Old_Policy = None
            Policy = ACModel('Policy', state_space, action_size)
            Old_Policy = ACModel('Old_Policy', state_space, action_size)
            PPO = PPOTrain(Policy,Old_Policy,1,GAMMA)
            sess.run(tf.global_variables_initializer())

            lastReward = 0
            reward = 0
            succes_num = 0
            itera = 0
            reach = False
            reachpoint = 0
            for iteration in range(NUMBER_EPISODES):

                states = []
                actions = []
                values = []
                rewards = []
                itera += 1
                t = 0
                state = env.reset()
                done = False
                while not done:
                    t += 1
                    state = np.stack([state]).astype(dtype=np.float32)
                    action, value = Policy.act(state, 1, True)

                    action = np.asscalar(action)
                    value = np.asscalar(value)
                    next_state, reward, done = env.step(action)

                    states.append(state)
                    actions.append(action)
                    values.append(value)
                    
                    
                    if done:
                        v_next = values[1:] + [0]
                    
                    rewards.append(reward)
                    state = next_state

                
                
                gaes = PPO.get_gaes(rewards, values, v_next)

                states = np.reshape(states, newshape = [-1,state_space])
                actions = np.array(actions).astype(dtype=np.int32)
                rewards = np.array(rewards).astype(dtype=np.float32)
                v_next = np.array(v_next).astype(dtype=np.float32)
                gaes = np.array(gaes).astype(dtype=np.float32)
                gaes = (gaes - gaes.mean()) / gaes.std()

                PPO.assign_policy_parameters()
                inp = [states, actions, rewards, v_next, gaes]

                PPO.train(inp[0],inp[1],inp[2],inp[3],inp[4])
                
                lastReward = sum(rewards)
                averageRewardsTR[0][iteration] += lastReward
                print("Episode: {}/{} Reward:{}".format(iteration,NUMBER_EPISODES,sum(rewards)))
                if (iteration+1) % 10 == 0 and iteration != 0:
                    state = env.reset()
                    evaluationReward = 0
                    done= False
                    while not done:
                        state = np.stack([state]).astype(dtype=np.float32)
                        action, value = Policy.act(state, 1, False)
                        
                        action = np.asscalar(action)
                        next_state, reward, done = env.step(action)

                        evaluationReward+= reward
                        state = next_state

                    
                    index = (iteration+1)/10
                    evaluationTR[0][math.floor(index-1)]+= evaluationReward
                    
            totalit2 += itera
            avg = totalit2/(j+1)
            PPO = None
            sess.close()

    averageRewardsTR = np.true_divide(averageRewardsTR, EPOCS)
    evaluationTR = np.true_divide(evaluationTR, EPOCS)
    
    maximunFinalReward = -10000000

    for j in range(EPOCS):

        tf.reset_default_graph()
        with tf.Session() as sess:
            
            env = targetSimulated[0]
            state_space = env.state_space
            action_size = env.action_space
            Policy = None
            Old_Policy = None
            Policy = ACModel('Policy', state_space, action_size)
            Old_Policy = ACModel('Old_Policy', state_space, action_size)
            PPO = PPOTrain(Policy,Old_Policy,1,GAMMA)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            finalReward = 0

            COLUMN = 1
            lastReward = 0
            reward = 0
            succes_num = 0
            itera = 0
            reach = False
            reachpoint = 0
            for iteration in range(NUMBER_EPISODES):

                states = []
                actions = []
                values = []
                rewards = []
                itera += 1
                t = 0
                state = env.reset()
                done = False
                while not done:
                   
                    t += 1
                    state = np.stack([state]).astype(dtype=np.float32)
                    action, value = Policy.act(state, 1, True)

                    action = np.asscalar(action)
                    value = np.asscalar(value)
                    next_state, reward, done = env.step(action)

                    states.append(state)
                    actions.append(action)
                    values.append(value)
                    
                    if done:
                        v_next = values[1:] + [0]
                    
                    rewards.append(reward)
                    
                    state = next_state

                if sum(rewards) >= 40:
                    if not reach:
                        reachpoint = itera
                        reach = True
               
                
                gaes = PPO.get_gaes(rewards, values, v_next)

                states = np.reshape(states, newshape = [-1,state_space])
                actions = np.array(actions).astype(dtype=np.int32)
                rewards = np.array(rewards).astype(dtype=np.float32)
                v_next = np.array(v_next).astype(dtype=np.float32)
                gaes = np.array(gaes).astype(dtype=np.float32)
                gaes = (gaes - gaes.mean()) / gaes.std()

                PPO.assign_policy_parameters()
                inp = [states, actions, rewards, v_next, gaes]

                PPO.train(inp[0],inp[1],inp[2],inp[3],inp[4])
                
                lastReward = sum(rewards)
                finalReward += lastReward
                
                    
            totalit2 += itera
            avg = totalit2/(j+1)
            if finalReward > maximunFinalReward:
                maximunFinalReward = finalReward
                
                save_path = saver.save(sess, "/tmp/model2.ckpt")

            if j < (EPOCS - 1):
                continue

            
            saver.restore(sess, "/tmp/model2.ckpt")


            COLUMN += 1
            ###################THIRD COLUMN####################
            PPO = PPOTrain(Policy,Old_Policy,COLUMN,GAMMA)
           
                
            global_vars          = tf.global_variables()
            is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

            
            if len(not_initialized_vars):
                sess.run(tf.variables_initializer(not_initialized_vars))
            
            save_path = saver.save(sess, "/tmp/model2.ckpt")
            
            for l in range(EPOCS):
                saver.restore(sess, "/tmp/model2.ckpt")
                env = targetReal[0]
                state_space = env.state_space
                action_size = env.action_space
                lastReward = 0
                reward = 0
                succes_num = 0
                itera = 0
                reach = False
                reachpoint = 0
                totalit2 = 0
                for iteration in range(NUMBER_EPISODES):
                    states = []
                    actions = []
                    values = []
                    rewards = []
                    itera += 1
                    t = 0
                    state = env.reset()
                    done = False
                    while not done:
                        t += 1
                        state = np.stack([state]).astype(dtype=np.float32)


                        action, value = Policy.act(state, COLUMN, True)

                        action = np.asscalar(action)
                        value = np.asscalar(value)
                        next_state, reward, done = env.step(action)

                        states.append(state)
                        actions.append(action)
                        values.append(value)
                        
                        if done:
                            v_next = values[1:] + [0]
                        
                        rewards.append(reward)
                        state = next_state

                    if sum(rewards) >= 40:
                        if not reach:
                            reachpoint = itera
                            reach = True
                    
                    gaes = PPO.get_gaes(rewards, values, v_next)

                    states = np.reshape(states, newshape = [-1,state_space])
                    actions = np.array(actions).astype(dtype=np.int32)
                    rewards = np.array(rewards).astype(dtype=np.float32)
                    v_next = np.array(v_next).astype(dtype=np.float32)
                    gaes = np.array(gaes).astype(dtype=np.float32)
                    gaes = (gaes - gaes.mean()) / gaes.std()

                    PPO.assign_policy_parameters()
                    inp = [states, actions, rewards, v_next, gaes]

                    PPO.train(inp[0],inp[1],inp[2],inp[3],inp[4])
                    
                    lastReward = sum(rewards)
                    averageRewardsNormal[0][iteration] += lastReward
                    if (iteration+1) % 10 == 0 and iteration != 0:
                        state = env.reset()
                        evaluationReward = 0
                        done = False
                        while not done:
                            state = np.stack([state]).astype(dtype=np.float32)
                            action, value = Policy.act(state, COLUMN, False)

                            action = np.asscalar(action)
                            next_state, reward, done = env.step(action)

                            
                            evaluationReward+= reward
                            state = next_state
                        
                        index = (iteration+1)/10
                        evaluationNormal[0][math.floor(index-1)]+= evaluationReward
                totalit2 += itera
                avg = totalit2/(j+1)
            
            PPO = None
            sess.close()

    averageRewardsNormal = np.true_divide(averageRewardsNormal, EPOCS)
    evaluationNormal = np.true_divide(evaluationNormal, EPOCS)

    
    maximunFinalReward = -10000000

    for j in range(EPOCS):

        tf.reset_default_graph()
        with tf.Session() as sess:
            
            env = targetSimulated[1]
            state_space = env.state_space
            action_size = env.action_space
            Policy = None
            Old_Policy = None
            Policy = ACModel('Policy', state_space, action_size)
            Old_Policy = ACModel('Old_Policy', state_space, action_size)
            PPO = PPOTrain(Policy,Old_Policy,1,GAMMA)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            finalReward = 0

            COLUMN = 1
            lastReward = 0
            reward = 0
            succes_num = 0
            itera = 0
            reach = False
            reachpoint = 0
            for iteration in range(NUMBER_EPISODES):

                states = []
                actions = []
                values = []
                rewards = []
                itera += 1
                t = 0
                state = env.reset()
                done = False
                while not done:

                    t += 1
                    state = np.stack([state]).astype(dtype=np.float32)
                    action, value = Policy.act(state, 1, True)

                    action = np.asscalar(action)
                    value = np.asscalar(value)
                    next_state, reward, done = env.step(action)

                    states.append(state)
                    actions.append(action)
                    values.append(value)
                    
                   
                    if done:
                        v_next = values[1:] + [0]
                    
                    rewards.append(reward)
                    
                    state = next_state

                if sum(rewards) >= 40:
                    if not reach:
                        reachpoint = itera
                        reach = True
                
                
                gaes = PPO.get_gaes(rewards, values, v_next)

                states = np.reshape(states, newshape = [-1,state_space])
                actions = np.array(actions).astype(dtype=np.int32)
                rewards = np.array(rewards).astype(dtype=np.float32)
                v_next = np.array(v_next).astype(dtype=np.float32)
                gaes = np.array(gaes).astype(dtype=np.float32)
                gaes = (gaes - gaes.mean()) / gaes.std()

                PPO.assign_policy_parameters()
                inp = [states, actions, rewards, v_next, gaes]

                PPO.train(inp[0],inp[1],inp[2],inp[3],inp[4])
                
                lastReward = sum(rewards)
                finalReward += lastReward
                
                    
            totalit2 += itera
            avg = totalit2/(j+1)
            if finalReward > maximunFinalReward:
                maximunFinalReward = finalReward
                
                save_path = saver.save(sess, "/tmp/model5.ckpt")

            if j < (EPOCS - 1):
                continue

            
            saver.restore(sess, "/tmp/model5.ckpt")


            COLUMN += 1
            ###################THIRD COLUMN####################
            PPO = PPOTrain(Policy,Old_Policy,COLUMN,GAMMA)
           
                
            global_vars          = tf.global_variables()
            is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

            
            if len(not_initialized_vars):
                sess.run(tf.variables_initializer(not_initialized_vars))
            
            save_path = saver.save(sess, "/tmp/model5.ckpt")
            
            for l in range(EPOCS):
                saver.restore(sess, "/tmp/model5.ckpt")
                env = targetReal[0]
                state_space = env.state_space
                action_size = env.action_space
                lastReward = 0
                reward = 0
                succes_num = 0
                itera = 0
                reach = False
                reachpoint = 0
                totalit2 = 0
                for iteration in range(NUMBER_EPISODES):
                    states = []
                    actions = []
                    values = []
                    rewards = []
                    itera += 1
                    t = 0
                    state = env.reset()
                    done = False
                    while not done:
                       
                        t += 1
                        state = np.stack([state]).astype(dtype=np.float32)


                        action, value = Policy.act(state, COLUMN, True)

                        action = np.asscalar(action)
                        value = np.asscalar(value)
                        next_state, reward, done = env.step(action)

                        states.append(state)
                        actions.append(action)
                        values.append(value)
                        
                        
                        if done:
                            v_next = values[1:] + [0]
                        
                        rewards.append(reward)
                        
                        state = next_state

                    if sum(rewards) >= 40:
                        if not reach:
                            reachpoint = itera
                            reach = True
                    
                    gaes = PPO.get_gaes(rewards, values, v_next)

                    states = np.reshape(states, newshape = [-1,state_space])
                    actions = np.array(actions).astype(dtype=np.int32)
                    rewards = np.array(rewards).astype(dtype=np.float32)
                    v_next = np.array(v_next).astype(dtype=np.float32)
                    gaes = np.array(gaes).astype(dtype=np.float32)
                    gaes = (gaes - gaes.mean()) / gaes.std()

                    PPO.assign_policy_parameters()
                    inp = [states, actions, rewards, v_next, gaes]

                    PPO.train(inp[0],inp[1],inp[2],inp[3],inp[4])
                    
                    
                    lastReward = sum(rewards)
                    averageRewardsNormal2[0][iteration] += lastReward
                    if (iteration+1) % 10 == 0 and iteration != 0:
                        state = env.reset()
                        evaluationReward = 0
                        done = False
                        while not done:
                            state = np.stack([state]).astype(dtype=np.float32)
                            action, value = Policy.act(state, COLUMN, False)

                            action = np.asscalar(action)
                            next_state, reward, done = env.step(action)

                            
                            evaluationReward+= reward
                            state = next_state
                        
                        index = (iteration+1)/10
                        
                        evaluationNormal2[0][math.floor(index-1)]+= evaluationReward
                totalit2 += itera
                avg = totalit2/(j+1)
                
            
            PPO = None
            sess.close()

    averageRewardsNormal2 = np.true_divide(averageRewardsNormal2, EPOCS)
    evaluationNormal2 = np.true_divide(evaluationNormal2, EPOCS)


    maximunFinalReward = -10000000
    for j in range(EPOCS):

        tf.reset_default_graph()
        with tf.Session() as sess:
            
            env = targetSimulated[2]
            state_space = env.state_space
            action_size = env.action_space
            Policy = None
            Old_Policy = None
            Policy = ACModel('Policy', state_space, action_size)
            Old_Policy = ACModel('Old_Policy', state_space, action_size)
            PPO = PPOTrain(Policy,Old_Policy,1,GAMMA)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            finalReward = 0

            COLUMN = 1
            lastReward = 0
            reward = 0
            succes_num = 0
            itera = 0
            reach = False
            reachpoint = 0
            for iteration in range(NUMBER_EPISODES):

                states = []
                actions = []
                values = []
                rewards = []
                itera += 1
                t = 0
                state = env.reset()
                done = False
                while not done:
                    t += 1
                    state = np.stack([state]).astype(dtype=np.float32)
                    action, value = Policy.act(state, 1, True)

                    action = np.asscalar(action)
                    value = np.asscalar(value)
                    next_state, reward, done = env.step(action)

                    states.append(state)
                    actions.append(action)
                    values.append(value)
                    
                    if done:
                        v_next = values[1:] + [0]
                    
                    rewards.append(reward)
                    state = next_state

                if sum(rewards) >= 40:
                    if not reach:
                        reachpoint = itera
                        reach = True
                
                gaes = PPO.get_gaes(rewards, values, v_next)

                states = np.reshape(states, newshape = [-1,state_space])
                actions = np.array(actions).astype(dtype=np.int32)
                rewards = np.array(rewards).astype(dtype=np.float32)
                v_next = np.array(v_next).astype(dtype=np.float32)
                gaes = np.array(gaes).astype(dtype=np.float32)
                gaes = (gaes - gaes.mean()) / gaes.std()

                PPO.assign_policy_parameters()
                inp = [states, actions, rewards, v_next, gaes]

                PPO.train(inp[0],inp[1],inp[2],inp[3],inp[4])
                
                lastReward = sum(rewards)
                finalReward += lastReward
                
                    
            totalit2 += itera
            avg = totalit2/(j+1)
            
            if finalReward > maximunFinalReward:
                maximunFinalReward = finalReward
                
                save_path = saver.save(sess, "/tmp/model6.ckpt")

            if j < (EPOCS - 1):
                continue

            
            saver.restore(sess, "/tmp/model6.ckpt")


            COLUMN += 1
                ###################THIRD COLUMN####################
            PPO = PPOTrain(Policy,Old_Policy,COLUMN,GAMMA)
           
                
            global_vars          = tf.global_variables()
            is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

          
            if len(not_initialized_vars):
                sess.run(tf.variables_initializer(not_initialized_vars))
            
            save_path = saver.save(sess, "/tmp/model6.ckpt")
            
            for l in range(EPOCS):
                saver.restore(sess, "/tmp/model6.ckpt")
                env = targetReal[0]
                state_space = env.state_space
                action_size = env.action_space
                lastReward = 0
                reward = 0
                succes_num = 0
                itera = 0
                reach = False
                reachpoint = 0
                totalit2 = 0
                for iteration in range(NUMBER_EPISODES):
                    states = []
                    actions = []
                    values = []
                    rewards = []
                    itera += 1
                    t = 0
                    state = env.reset()
                    done = False
                    while not done:
                        
                        t += 1
                        state = np.stack([state]).astype(dtype=np.float32)


                        action, value = Policy.act(state, COLUMN, True)

                        action = np.asscalar(action)
                        value = np.asscalar(value)
                        next_state, reward, done = env.step(action)

                        states.append(state)
                        actions.append(action)
                        values.append(value)
                        
                        
                        if done:
                            v_next = values[1:] + [0]
                        
                        rewards.append(reward)
                     
                        state = next_state

                    if sum(rewards) >= 40:
                        if not reach:
                            reachpoint = itera
                            reach = True
                    
                    gaes = PPO.get_gaes(rewards, values, v_next)

                    states = np.reshape(states, newshape = [-1,state_space])
                    actions = np.array(actions).astype(dtype=np.int32)
                    rewards = np.array(rewards).astype(dtype=np.float32)
                    v_next = np.array(v_next).astype(dtype=np.float32)
                    gaes = np.array(gaes).astype(dtype=np.float32)
                    gaes = (gaes - gaes.mean()) / gaes.std()

                    PPO.assign_policy_parameters()
                    inp = [states, actions, rewards, v_next, gaes]

                    PPO.train(inp[0],inp[1],inp[2],inp[3],inp[4])
                    
                    lastReward = sum(rewards)
                    averageRewardsNormal3[0][iteration] += lastReward
                    if (iteration+1) % 10 == 0 and iteration != 0:
                        state = env.reset()
                        evaluationReward = 0
                        done = False
                        while not done:
                            state = np.stack([state]).astype(dtype=np.float32)
                            action, value = Policy.act(state, COLUMN, False)

                            action = np.asscalar(action)
                            next_state, reward, done = env.step(action)

                            
                            evaluationReward+= reward

                            state = next_state
                        
                        index = (iteration+1)/10

                        evaluationNormal3[0][math.floor(index-1)]+= evaluationReward

                totalit2 += itera
                avg = totalit2/(j+1)

            
            PPO = None
            sess.close()

    averageRewardsNormal3 = np.true_divide(averageRewardsNormal3, EPOCS)
    evaluationNormal3 = np.true_divide(evaluationNormal3, EPOCS)
  
    
    

    maximunFinalReward = -10000000
    for j in range(EPOCS):

        tf.reset_default_graph()
        with tf.Session() as sess:
            
            env = targetSimulated[3]

            state_space = env.state_space
            action_size = env.action_space
            Policy = None
            Old_Policy = None
            Policy = ACModel('Policy', state_space, action_size)
            Old_Policy = ACModel('Old_Policy', state_space, action_size)
            PPO = PPOTrain(Policy,Old_Policy,1,GAMMA)

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            finalReward = 0

            COLUMN = 1
            lastReward = 0
            reward = 0
            succes_num = 0
            itera = 0
            reach = False
            reachpoint = 0
            for iteration in range(NUMBER_EPISODES):

                states = []
                actions = []
                values = []
                rewards = []
                itera += 1
                t = 0
                state = env.reset()
                done = False
                while not done:
                    t += 1
                    state = np.stack([state]).astype(dtype=np.float32)
                    action, value = Policy.act(state, 1, True)

                    action = np.asscalar(action)
                    value = np.asscalar(value)
                    next_state, reward, done = env.step(action)

                    states.append(state)
                    actions.append(action)
                    values.append(value)

                    if done:
                        v_next = values[1:] + [0]
                    
                    rewards.append(reward)

                    state = next_state

                if sum(rewards) >= 40:
                    if not reach:
                        reachpoint = itera
                        reach = True
                
                
                gaes = PPO.get_gaes(rewards, values, v_next)

                states = np.reshape(states, newshape = [-1,state_space])
                actions = np.array(actions).astype(dtype=np.int32)
                rewards = np.array(rewards).astype(dtype=np.float32)
                v_next = np.array(v_next).astype(dtype=np.float32)
                gaes = np.array(gaes).astype(dtype=np.float32)
                gaes = (gaes - gaes.mean()) / gaes.std()

                PPO.assign_policy_parameters()
                inp = [states, actions, rewards, v_next, gaes]

                PPO.train(inp[0],inp[1],inp[2],inp[3],inp[4])

                
                lastReward = sum(rewards)
                finalReward += lastReward
                
                    
            totalit2 += itera
            avg = totalit2/(j+1)

            if finalReward > maximunFinalReward:
                maximunFinalReward = finalReward
                
                save_path = saver.save(sess, "/tmp/model7.ckpt")

            if j < (EPOCS - 1):
                continue

            
            saver.restore(sess, "/tmp/model7.ckpt")


            COLUMN += 1
            ###################THIRD COLUMN####################
            PPO = PPOTrain(Policy,Old_Policy,COLUMN,GAMMA)
           
                
            global_vars          = tf.global_variables()
            is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]


            if len(not_initialized_vars):
                sess.run(tf.variables_initializer(not_initialized_vars))
            
            save_path = saver.save(sess, "/tmp/model7.ckpt")
            
            for l in range(EPOCS):
                saver.restore(sess, "/tmp/model7.ckpt")
                env = targetReal[0]
                state_space = env.state_space
                action_size = env.action_space
                lastReward = 0
                reward = 0
                succes_num = 0
                itera = 0
                reach = False
                reachpoint = 0
                totalit2 = 0
                for iteration in range(NUMBER_EPISODES):
                    states = []
                    actions = []
                    values = []
                    rewards = []
                    itera += 1
                    t = 0
                    state = env.reset()
                    done = False
                    while not done:

                        t += 1
                        state = np.stack([state]).astype(dtype=np.float32)


                        action, value = Policy.act(state, COLUMN, True)

                        action = np.asscalar(action)
                        value = np.asscalar(value)
                        next_state, reward, done = env.step(action)

                        states.append(state)
                        actions.append(action)
                        values.append(value)
                        
         
                        if done:
                            v_next = values[1:] + [0]
                        
                        rewards.append(reward)
   
                        state = next_state

                    if sum(rewards) >= 40:
                        if not reach:
                            reachpoint = itera
                            reach = True
                    
                    gaes = PPO.get_gaes(rewards, values, v_next)

                    states = np.reshape(states, newshape = [-1,state_space])
                    actions = np.array(actions).astype(dtype=np.int32)
                    rewards = np.array(rewards).astype(dtype=np.float32)
                    v_next = np.array(v_next).astype(dtype=np.float32)
                    gaes = np.array(gaes).astype(dtype=np.float32)
                    gaes = (gaes - gaes.mean()) / gaes.std()

                    PPO.assign_policy_parameters()
                    inp = [states, actions, rewards, v_next, gaes]

                    PPO.train(inp[0],inp[1],inp[2],inp[3],inp[4])
                    

                    lastReward = sum(rewards)
                    averageRewardsNormal4[0][iteration] += lastReward

                    if (iteration+1) % 10 == 0 and iteration != 0:
                        state = env.reset()
                        evaluationReward = 0
                        done = False
                        while not done:
                            state = np.stack([state]).astype(dtype=np.float32)
                            action, value = Policy.act(state, COLUMN, False)

                            action = np.asscalar(action)
                            next_state, reward, done = env.step(action)

                            
               
                            evaluationReward+= reward

                            state = next_state
                        
                        index = (iteration+1)/10

                        evaluationNormal4[0][math.floor(index-1)]+= evaluationReward

                totalit2 += itera
                avg = totalit2/(j+1)

            
            PPO = None
            sess.close()

    averageRewardsNormal4 = np.true_divide(averageRewardsNormal4, EPOCS)
    evaluationNormal4 = np.true_divide(evaluationNormal4, EPOCS)
    maximunFinalReward = -10000000

    for j in range(EPOCS):

        tf.reset_default_graph()
        with tf.Session() as sess:
            
            env = targetSimulated[4]

            state_space = env.state_space
            action_size = env.action_space
            Policy = None
            Old_Policy = None
            Policy = ACModel('Policy', state_space, action_size)
            Old_Policy = ACModel('Old_Policy', state_space, action_size)
            PPO = PPOTrain(Policy,Old_Policy,1,GAMMA)

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            finalReward = 0

            COLUMN = 1
            lastReward = 0
            reward = 0
            succes_num = 0
            itera = 0
            reach = False
            reachpoint = 0
            for iteration in range(NUMBER_EPISODES):

                states = []
                actions = []
                values = []
                rewards = []
                itera += 1
                t = 0
                state = env.reset()
                done = False
                while not done:

                    t += 1
                    state = np.stack([state]).astype(dtype=np.float32)
                    action, value = Policy.act(state, 1, True)

                    action = np.asscalar(action)
                    value = np.asscalar(value)
                    next_state, reward, done = env.step(action)

                    states.append(state)
                    actions.append(action)
                    values.append(value)
                    

                    if done:
                        v_next = values[1:] + [0]
                    
                    rewards.append(reward)

                    state = next_state

                if sum(rewards) >= 40:
                    if not reach:
                        reachpoint = itera
                        reach = True
    
            
                
                gaes = PPO.get_gaes(rewards, values, v_next)

                states = np.reshape(states, newshape = [-1,state_space])
                actions = np.array(actions).astype(dtype=np.int32)
                rewards = np.array(rewards).astype(dtype=np.float32)
                v_next = np.array(v_next).astype(dtype=np.float32)
                gaes = np.array(gaes).astype(dtype=np.float32)
                gaes = (gaes - gaes.mean()) / gaes.std()

                PPO.assign_policy_parameters()
                inp = [states, actions, rewards, v_next, gaes]


                PPO.train(inp[0],inp[1],inp[2],inp[3],inp[4])

                
                lastReward = sum(rewards)
                finalReward += lastReward
                
                    
 
            totalit2 += itera
            avg = totalit2/(j+1)

            if finalReward > maximunFinalReward:
                maximunFinalReward = finalReward
                
                save_path = saver.save(sess, "/tmp/model8.ckpt")

            if j < (EPOCS - 1):
                continue

            
            saver.restore(sess, "/tmp/model8.ckpt")


            COLUMN += 1
            ###################THIRD COLUMN####################
            PPO = PPOTrain(Policy,Old_Policy,COLUMN,GAMMA)
           
                
            global_vars          = tf.global_variables()
            is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]


            if len(not_initialized_vars):
                sess.run(tf.variables_initializer(not_initialized_vars))
            
            save_path = saver.save(sess, "/tmp/model8.ckpt")
            
            for l in range(EPOCS):
                saver.restore(sess, "/tmp/model8.ckpt")
                env = targetReal[0]
                state_space = env.state_space
                action_size = env.action_space
                lastReward = 0
                reward = 0
                succes_num = 0
                itera = 0
                reach = False
                reachpoint = 0
                totalit2 = 0
                for iteration in range(NUMBER_EPISODES):
                    states = []
                    actions = []
                    values = []
                    rewards = []
                    itera += 1
                    t = 0
                    state = env.reset()
                    done = False
                    while not done:

                        t += 1
                        state = np.stack([state]).astype(dtype=np.float32)


                        action, value = Policy.act(state, COLUMN, True)

                        action = np.asscalar(action)
                        value = np.asscalar(value)
                        next_state, reward, done = env.step(action)

                        states.append(state)
                        actions.append(action)
                        values.append(value)
                        
 
                        if done:
                            v_next = values[1:] + [0]
                        
                        rewards.append(reward)
     
                        state = next_state


                    if sum(rewards) >= 40:
                        if not reach:
                            reachpoint = itera
                            reach = True
                    
                    gaes = PPO.get_gaes(rewards, values, v_next)

                    states = np.reshape(states, newshape = [-1,state_space])
                    actions = np.array(actions).astype(dtype=np.int32)
                    rewards = np.array(rewards).astype(dtype=np.float32)
                    v_next = np.array(v_next).astype(dtype=np.float32)
                    gaes = np.array(gaes).astype(dtype=np.float32)
                    gaes = (gaes - gaes.mean()) / gaes.std()

                    PPO.assign_policy_parameters()
                    inp = [states, actions, rewards, v_next, gaes]

                    PPO.train(inp[0],inp[1],inp[2],inp[3],inp[4])
                    
                    lastReward = sum(rewards)
                    averageRewardsNormal5[0][iteration] += lastReward

                    if (iteration+1) % 10 == 0 and iteration != 0:
                        state = env.reset()
                        evaluationReward = 0
                        done = False
                        while not done:
                            state = np.stack([state]).astype(dtype=np.float32)
                            action, value = Policy.act(state, COLUMN, False)

                            action = np.asscalar(action)
                            next_state, reward, done = env.step(action)

                            
                            evaluationReward+= reward
                            
                            state = next_state
                        
                        index = (iteration+1)/10
                        
                        evaluationNormal5[0][math.floor(index-1)]+= evaluationReward

                totalit2 += itera
                avg = totalit2/(j+1)
            
            PPO = None
            sess.close()

    averageRewardsNormal5 = np.true_divide(averageRewardsNormal5, EPOCS)
    evaluationNormal5 = np.true_divide(evaluationNormal5, EPOCS)

    maximunFinalReward = -100000000
    for j in range(EPOCS):
        break
        tf.reset_default_graph()
        COLUMN = 0
        with tf.Session() as sess:
            

            state_space = STATE_SPAC
            action_size = ACTION_SPAC
            
        
            Policy = None
            Old_Policy = None

            Policy = ACModel('Policy', state_space, action_size)
            Old_Policy = ACModel('Old_Policy', state_space, action_size)
            
 
            COLUMN += 1
            env = sourceTask[1]
            
            PPO = PPOTrain(Policy,Old_Policy,COLUMN,GAMMA)
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            finalReward = 0
            lastReward = 0
            reward = 0
            succes_num = 0
            itera = 0
            reach = False
            reachpoint = 0
            for iteration in range(NUMBER_EPISODES):
                states = []
                actions = []
                values = []
                rewards = []
                t = 0
                state = env.reset()
                done = False
                itera += 1
                while not done:
                    t += 1
                    state = np.stack([state]).astype(dtype=np.float32)

                    
                    action, value = Policy.act(state, COLUMN, True)

                    action = np.asscalar(action)
                    value = np.asscalar(value)
                    next_state, reward, done = env.step(action)

                    states.append(state)
                    actions.append(action)
                    values.append(value)
                    
                    if done:
                        v_next = values[1:] + [0]
                    
                    rewards.append(reward)
                    state = next_state

                if sum(rewards) >= 40:
                    if not reach:
                        reachpoint = itera
                        reach = True
                
                gaes = PPO.get_gaes(rewards, values, v_next)

                states = np.reshape(states, newshape = [-1,state_space])
                actions = np.array(actions).astype(dtype=np.int32)
                rewards = np.array(rewards).astype(dtype=np.float32)
                v_next = np.array(v_next).astype(dtype=np.float32)
                gaes = np.array(gaes).astype(dtype=np.float32)
                gaes = (gaes - gaes.mean()) / gaes.std()

                PPO.assign_policy_parameters()
                inp = [states, actions, rewards, v_next, gaes]
                PPO.train(inp[0],inp[1],inp[2], inp[3], inp[4])
                lastReward = sum(rewards)
            totalit += itera
            avg = totalit/(j+1)



            COLUMN += 1
            env = sourceTask[2]
            PPO = PPOTrain(Policy,Old_Policy,COLUMN,GAMMA)
        

            global_vars          = tf.global_variables()
            is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

            if len(not_initialized_vars):
                sess.run(tf.variables_initializer(not_initialized_vars))

            lastReward = 0
            reward = 0
            succes_num = 0
            itera = 0
            reach = False
            reachpoint = 0
            for iteration in range(NUMBER_EPISODES):
                states = []
                actions = []
                values = []
                rewards = []
                t = 0
                state = env.reset()
                done = False
                itera += 1
                while not done:
                    t += 1
                    state = np.stack([state]).astype(dtype=np.float32)

                    
                    action, value = Policy.act(state, COLUMN, True)

                    action = np.asscalar(action)
                    value = np.asscalar(value)
                    next_state, reward, done = env.step(action)

                    states.append(state)
                    actions.append(action)
                    values.append(value)
                    
                    if done:
                        v_next = values[1:] + [0]
                    
                    rewards.append(reward)
                    state = next_state

                if sum(rewards) >= 40:
                    if not reach:
                        reachpoint = itera
                        reach = True
                
                gaes = PPO.get_gaes(rewards, values, v_next)

                states = np.reshape(states, newshape = [-1,state_space])
                actions = np.array(actions).astype(dtype=np.int32)
                rewards = np.array(rewards).astype(dtype=np.float32)
                v_next = np.array(v_next).astype(dtype=np.float32)
                gaes = np.array(gaes).astype(dtype=np.float32)
                gaes = (gaes - gaes.mean()) / gaes.std()

                PPO.assign_policy_parameters()
                inp = [states, actions, rewards, v_next, gaes]

                PPO.train(inp[0],inp[1],inp[2], inp[3], inp[4])
                lastReward = sum(rewards)
                
            totalit += itera
            avg = totalit/(j+1)
            
            COLUMN += 1
            env = sourceTask[3]
            PPO = PPOTrain(Policy,Old_Policy,COLUMN,GAMMA)
        

            global_vars          = tf.global_variables()
            is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

            if len(not_initialized_vars):
                sess.run(tf.variables_initializer(not_initialized_vars))

            lastReward = 0
            reward = 0
            succes_num = 0
            itera = 0
            reach = False
            reachpoint = 0
            for iteration in range(NUMBER_EPISODES):
                states = []
                actions = []
                values = []
                rewards = []
                t = 0
                state = env.reset()
                done = False
                itera += 1
                while not done:
                    t += 1
                    state = np.stack([state]).astype(dtype=np.float32)

                    
                    action, value = Policy.act(state, COLUMN, True)

                    action = np.asscalar(action)
                    value = np.asscalar(value)
                    next_state, reward, done = env.step(action)

                    states.append(state)
                    actions.append(action)
                    values.append(value)
                    if done:
                        v_next = values[1:] + [0]
                    
                    rewards.append(reward)
                    state = next_state
                if sum(rewards) >= 40:
                    if not reach:
                        reachpoint = itera
                        reach = True
                
                gaes = PPO.get_gaes(rewards, values, v_next)

                states = np.reshape(states, newshape = [-1,state_space])
                actions = np.array(actions).astype(dtype=np.int32)
                rewards = np.array(rewards).astype(dtype=np.float32)
                v_next = np.array(v_next).astype(dtype=np.float32)
                gaes = np.array(gaes).astype(dtype=np.float32)
                gaes = (gaes - gaes.mean()) / gaes.std()

                PPO.assign_policy_parameters()
                inp = [states, actions, rewards, v_next, gaes]

                PPO.train(inp[0],inp[1],inp[2], inp[3], inp[4])
                lastReward = sum(rewards)

            
            COLUMN += 1
            ###################SECOND COLUMN####################
            
            PPO = PPOTrain(Policy,Old_Policy,COLUMN,GAMMA)
            
            
            global_vars          = tf.global_variables()
            is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

            if len(not_initialized_vars):
                sess.run(tf.variables_initializer(not_initialized_vars))
            

            env = MGEnv("Data/9982HourlyDevUSEGEN-0117.xlsx")
            state_space = env.state_space
            action_size = env.action_space
            lastReward = 0
            reward = 0
            succes_num = 0
            itera = 0
            reach = False
            reachpoint = 0
            totalit2 = 0
            for iteration in range(NUMBER_EPISODES):
                states = []
                actions = []
                values = []
                rewards = []
                itera += 1
                t = 0
                state = env.reset()
                done = False
                while not done:
                    t += 1
                    state = np.stack([state]).astype(dtype=np.float32)

                
                    action, value = Policy.act(state, COLUMN, True)

                    action = np.asscalar(action)
                    value = np.asscalar(value)
                    next_state, reward, done = env.step(action)

                    states.append(state)
                    actions.append(action)
                    values.append(value)
                    
                    if done:
                        v_next = values[1:] + [0]
                    
                    rewards.append(reward)
                    state = next_state

                if sum(rewards) >= 40:
                    if not reach:
                        reachpoint = itera
                        reach = True
                
                gaes = PPO.get_gaes(rewards, values, v_next)
        
                states = np.reshape(states, newshape = [-1,state_space])
                actions = np.array(actions).astype(dtype=np.int32)
                rewards = np.array(rewards).astype(dtype=np.float32)
                v_next = np.array(v_next).astype(dtype=np.float32)
                gaes = np.array(gaes).astype(dtype=np.float32)
                gaes = (gaes - gaes.mean()) / gaes.std()

                PPO.assign_policy_parameters()
                inp = [states, actions, rewards, v_next, gaes]

                PPO.train(inp[0],inp[1],inp[2],inp[3],inp[4])
                
                lastReward = sum(rewards)
                finalReward += lastReward

            totalit2 += itera
            avg = totalit2/(j+1)

            if finalReward > maximunFinalReward:
                maximunFinalReward = finalReward
                
                save_path = saver.save(sess, "/tmp/model3.ckpt")

            if j < (EPOCS - 1):
                continue

            
            saver.restore(sess, "/tmp/model3.ckpt")



            COLUMN += 1


            ###################THIRD COLUMN####################
            PPO = PPOTrain(Policy,Old_Policy,COLUMN,GAMMA)
           
                
            global_vars          = tf.global_variables()
            is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

            if len(not_initialized_vars):
                sess.run(tf.variables_initializer(not_initialized_vars))
            
            save_path = saver.save(sess, "/tmp/model3.ckpt")
            
            for l in range(EPOCS):
                saver.restore(sess, "/tmp/model3.ckpt")
                env = targetReal[0]
                state_space = env.state_space
                action_size = env.action_space
                lastReward = 0
                reward = 0
                succes_num = 0
                itera = 0
                reach = False
                reachpoint = 0
                totalit2 = 0
                for iteration in range(NUMBER_EPISODES):
                    states = []
                    actions = []
                    values = []
                    rewards = []
                    itera += 1
                    t = 0
                    state = env.reset()
                    done = False
                    while not done:
                        t += 1
                        state = np.stack([state]).astype(dtype=np.float32)


                        action, value = Policy.act(state, COLUMN, True)

                        action = np.asscalar(action)
                        value = np.asscalar(value)
                        next_state, reward, done = env.step(action)

                        states.append(state)
                        actions.append(action)
                        values.append(value)
                        
                        if done:
                            v_next = values[1:] + [0]
                        
                        rewards.append(reward)

                        state = next_state

                    if sum(rewards) >= 40:
                        if not reach:
                            reachpoint = itera
                            reach = True
                    
                    gaes = PPO.get_gaes(rewards, values, v_next)

                    states = np.reshape(states, newshape = [-1,state_space])
                    actions = np.array(actions).astype(dtype=np.int32)
                    rewards = np.array(rewards).astype(dtype=np.float32)
                    v_next = np.array(v_next).astype(dtype=np.float32)
                    gaes = np.array(gaes).astype(dtype=np.float32)
                    gaes = (gaes - gaes.mean()) / gaes.std()

                    PPO.assign_policy_parameters()
                    inp = [states, actions, rewards, v_next, gaes]


                    PPO.train(inp[0],inp[1],inp[2],inp[3],inp[4])
                    
                    lastReward = sum(rewards)
                    averageRewardsTS[0][iteration] += lastReward

                    if (iteration+1) % 10 == 0 and iteration != 0:
                        state = env.reset()
                        evaluationReward = 0
                        done = False
                        while not done:
                            state = np.stack([state]).astype(dtype=np.float32)
                            action, value = Policy.act(state, COLUMN, False)

                            action = np.asscalar(action)
                            next_state, reward, done = env.step(action)

                            
                            evaluationReward+= reward
                            
                            state = next_state
                        
                        index = (iteration+1)/10
                        evaluationTS[0][math.floor(index-1)]+= evaluationReward

                totalit2 += itera
                avg = totalit2/(j+1)
            


            PPO = None
            sess.close()
    file = open("Results/Evaluation/CURRMicroGridManyDatesBestSimulated55-8967.txt","w")

    ######################CURR############################################################################################################

    maximunFinalReward = -10000000
    for j in range(EPOCS):

        tf.reset_default_graph()
        COLUMN = 0
        with tf.Session() as sess:
            

            
            state_space = STATE_SPAC
            action_size = ACTION_SPAC
            
            finalReward = 0
        
            Policy = None
            Old_Policy = None

            Policy = ACModel('Policy', state_space, action_size)
            Old_Policy = ACModel('Old_Policy', state_space, action_size)
            
  
            COLUMN += 1
            env = sourceTask[1]
            
            PPO = PPOTrain(Policy,Old_Policy,COLUMN,GAMMA)

            sess.run(tf.global_variables_initializer())


            saver = tf.train.Saver()

            
            lastReward = 0
            reward = 0
            succes_num = 0
            itera = 0
            reach = False
            reachpoint = 0
            for iteration in range(NUMBER_EPISODES):
                states = []
                actions = []
                values = []
                rewards = []
                t = 0
                state = env.reset()
                done = False
                itera += 1
                while not done:
                    t += 1
                    state = np.stack([state]).astype(dtype=np.float32)

                    
                    action, value = Policy.act(state, COLUMN, True)

                    action = np.asscalar(action)
                    value = np.asscalar(value)
                    next_state, reward, done = env.step(action)

                    states.append(state)
                    actions.append(action)
                    values.append(value)
               
                    if done:
                        v_next = values[1:] + [0]
                    
                    rewards.append(reward)
                    
                    state = next_state

                if sum(rewards) >= 40:
                    if not reach:
                        reachpoint = itera
                        reach = True
                
                gaes = PPO.get_gaes(rewards, values, v_next)

                states = np.reshape(states, newshape = [-1,state_space])
                actions = np.array(actions).astype(dtype=np.int32)
                rewards = np.array(rewards).astype(dtype=np.float32)
                v_next = np.array(v_next).astype(dtype=np.float32)
                gaes = np.array(gaes).astype(dtype=np.float32)
                gaes = (gaes - gaes.mean()) / gaes.std()

                PPO.assign_policy_parameters()
                inp = [states, actions, rewards, v_next, gaes]

                PPO.train(inp[0],inp[1],inp[2], inp[3], inp[4])
                lastReward = sum(rewards)
            totalit += itera
            avg = totalit/(j+1)

            
            FOUR_COLUMNS = True
            if FOUR_COLUMNS:   
                COLUMN += 1
                env = sourceTask[2]
                PPO = PPOTrain(Policy,Old_Policy,COLUMN,GAMMA)
            

                global_vars          = tf.global_variables()
                is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
                not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

                if len(not_initialized_vars):
                    sess.run(tf.variables_initializer(not_initialized_vars))

                lastReward = 0
                reward = 0
                succes_num = 0
                itera = 0
                reach = False
                reachpoint = 0
                for iteration in range(NUMBER_EPISODES):
                    states = []
                    actions = []
                    values = []
                    rewards = []
                    t = 0
                    state = env.reset()
                    done = False
                    itera += 1
                    while not done:
                 
                        t += 1
                        state = np.stack([state]).astype(dtype=np.float32)

                        
                        action, value = Policy.act(state, COLUMN, True)

                        action = np.asscalar(action)
                        value = np.asscalar(value)
                        next_state, reward, done = env.step(action)

                        states.append(state)
                        actions.append(action)
                        values.append(value)
               
                        if done:
                            v_next = values[1:] + [0]
                        
                        rewards.append(reward)
                   
                        state = next_state

                    if sum(rewards) >= 40:
                        if not reach:
                            reachpoint = itera
                            reach = True
                    
                    gaes = PPO.get_gaes(rewards, values, v_next)

                    states = np.reshape(states, newshape = [-1,state_space])
                    actions = np.array(actions).astype(dtype=np.int32)
                    rewards = np.array(rewards).astype(dtype=np.float32)
                    v_next = np.array(v_next).astype(dtype=np.float32)
                    gaes = np.array(gaes).astype(dtype=np.float32)
                    gaes = (gaes - gaes.mean()) / gaes.std()

                    PPO.assign_policy_parameters()
                    inp = [states, actions, rewards, v_next, gaes]

                    PPO.train(inp[0],inp[1],inp[2], inp[3], inp[4])

                    lastReward = sum(rewards)
         
                totalit += itera
                avg = totalit/(j+1)
                

            
            COLUMN += 1
            ###################SECOND COLUMN####################
            
            PPO = PPOTrain(Policy,Old_Policy,COLUMN,GAMMA)


            
            
            global_vars          = tf.global_variables()
            is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]


            if len(not_initialized_vars):
                sess.run(tf.variables_initializer(not_initialized_vars))
            

            env = sourceTask[3]
            state_space = env.state_space
            action_size = env.action_space
            lastReward = 0
            reward = 0
            succes_num = 0
            itera = 0
            reach = False
            reachpoint = 0
            totalit2 = 0
            for iteration in range(NUMBER_EPISODES):
                states = []
                actions = []
                values = []
                rewards = []
                itera += 1
                t = 0
                state = env.reset()
                done = False
                while not done:

                    t += 1
                    state = np.stack([state]).astype(dtype=np.float32)

                
                    action, value = Policy.act(state, COLUMN, True)

                    action = np.asscalar(action)
                    value = np.asscalar(value)
                    next_state, reward, done = env.step(action)

                    states.append(state)
                    actions.append(action)
                    values.append(value)
                    

                    if done:
                        v_next = values[1:] + [0]
                    
                    rewards.append(reward)

                    state = next_state


                if sum(rewards) >= 40:
                    if not reach:
                        reachpoint = itera
                        reach = True
                
                gaes = PPO.get_gaes(rewards, values, v_next)
        
                states = np.reshape(states, newshape = [-1,state_space])
                actions = np.array(actions).astype(dtype=np.int32)
                rewards = np.array(rewards).astype(dtype=np.float32)
                v_next = np.array(v_next).astype(dtype=np.float32)
                gaes = np.array(gaes).astype(dtype=np.float32)
                gaes = (gaes - gaes.mean()) / gaes.std()

                PPO.assign_policy_parameters()
                inp = [states, actions, rewards, v_next, gaes]

                PPO.train(inp[0],inp[1],inp[2],inp[3],inp[4])
                

                lastReward = sum(rewards)
                finalReward += lastReward

  
            totalit2 += itera
            avg = totalit2/(j+1)



            
            if finalReward > maximunFinalReward:
                maximunFinalReward = finalReward
                
                save_path = saver.save(sess, "/tmp/model8967.ckpt")

            if j < (EPOCS - 1):
                continue

            
            saver.restore(sess, "/tmp/model8967.ckpt")



            COLUMN += 1
                ###################THIRD COLUMN####################
            PPO = PPOTrain(Policy,Old_Policy,COLUMN,GAMMA)

           
                
            global_vars          = tf.global_variables()
            is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]


            if len(not_initialized_vars):
                sess.run(tf.variables_initializer(not_initialized_vars))
            
            save_path = saver.save(sess, "/tmp/model8967.ckpt")
            
            for l in range(EPOCS):
                saver.restore(sess, "/tmp/model8967.ckpt")
                env = targetReal[0]
                state_space = env.state_space
                action_size = env.action_space
                lastReward = 0
                reward = 0
                succes_num = 0
                itera = 0
                reach = False
                reachpoint = 0
                totalit2 = 0
                for iteration in range(NUMBER_EPISODES):
                    states = []
                    actions = []
                    values = []
                    rewards = []
                    itera += 1
                    t = 0
                    state = env.reset()
                    done = False
                    while not done:
          
                        t += 1
                        state = np.stack([state]).astype(dtype=np.float32)


                        action, value = Policy.act(state, COLUMN, True)

                        action = np.asscalar(action)
                        value = np.asscalar(value)
                        next_state, reward, done = env.step(action)

                        states.append(state)
                        actions.append(action)
                        values.append(value)
                        
                     
                        if done:
                            v_next = values[1:] + [0]
                        
                        rewards.append(reward)
      
                        state = next_state


                    if sum(rewards) >= 40:
                        if not reach:
                            reachpoint = itera
                            reach = True
                    
                    gaes = PPO.get_gaes(rewards, values, v_next)

                    states = np.reshape(states, newshape = [-1,state_space])
                    actions = np.array(actions).astype(dtype=np.int32)
                    rewards = np.array(rewards).astype(dtype=np.float32)
                    v_next = np.array(v_next).astype(dtype=np.float32)
                    gaes = np.array(gaes).astype(dtype=np.float32)
                    gaes = (gaes - gaes.mean()) / gaes.std()

                    PPO.assign_policy_parameters()
                    inp = [states, actions, rewards, v_next, gaes]


                    PPO.train(inp[0],inp[1],inp[2],inp[3],inp[4])

                    lastReward = sum(rewards)
                    averageRewardsTransfer[0][iteration] += lastReward
         
                    if (iteration+1) % 10 == 0 and iteration != 0:
                        state = env.reset()
                        evaluationReward = 0
                        done = False
                        while not done:
                            state = np.stack([state]).astype(dtype=np.float32)
                            action, value = Policy.act(state, COLUMN, False)

                            action = np.asscalar(action)
                            next_state, reward, done = env.step(action)

                            
                      
                            evaluationReward+= reward
                            state = next_state
                        
                        index = (iteration+1)/10
               
                        evaluationTransfer[0][math.floor(index-1)]+= evaluationReward
                
                totalit2 += itera
                avg = totalit2/(j+1)
                


            PPO = None
            sess.close()

    averageRewardsTransfer = np.true_divide(averageRewardsTransfer, EPOCS)
    
    evaluationTransfer = np.true_divide(evaluationTransfer, EPOCS)

    evaluationTS = np.true_divide(evaluationTS, EPOCS)
    averageRewardsTS  = np.true_divide(averageRewardsTS, EPOCS)

    for n in range(f_i):
        maxRegret = []
        ratioDif = 0

        np.save('averageRewardTransfer-8967.npy', averageRewardsTransfer[n])
        np.save('averageRewardNormal1-8967.npy', averageRewardsNormal[n])
        np.save('averageRewardNormal2-8967.npy', averageRewardsNormal2[n])
        np.save('averageRewardNormal3-8967.npy', averageRewardsNormal3[n])
        np.save('averageRewardNormal4-8967.npy', averageRewardsNormal4[n])
        np.save('averageRewardNormal5-8967.npy', averageRewardsNormal5[n])

        print(averageRewardsTransfer[n])
        print(averageRewardsNormal[n])
        print(averageRewardsNormal2[n])
        print(averageRewardsNormal3[n])
        print(averageRewardsNormal4[n])
        print(averageRewardsNormal5[n])

        ymin = min(min(averageRewardsTransfer[n]),min(averageRewardsNormal[n]),min(averageRewardsNormal2[n]),min(averageRewardsNormal3[n]),min(averageRewardsNormal4[n]),min(averageRewardsNormal5[n]))

        dif = abs(ymin)
        dataNormal = (averageRewardsNormal[n]+dif)
        dataTransfer = (averageRewardsTransfer[n]+dif)
        dataNormal2 = (averageRewardsNormal2[n]+dif)
        dataNormal3 = (averageRewardsNormal3[n]+dif)
        dataNormal4 = (averageRewardsNormal4[n]+dif)
        dataNormal5 = (averageRewardsNormal5[n]+dif)

        ymax = max(max(dataNormal),max(dataTransfer),max(dataNormal2),max(dataNormal3),max(dataNormal4),max(dataNormal5))

        areaTotal = ymax*NUMBER_EPISODES
        areaNormal = sum(dataNormal)
        areaTransfer = sum(dataTransfer)
        areaNormal2 = sum(dataNormal2)
        areaNormal3 = sum(dataNormal3)
        areaNormal4 = sum(dataNormal4)
        areaNormal5 = sum(dataNormal5)

        print("--------------------------------------------------------------------------------------------------------------")
        print("######################### TARGET REAL: {}".format(n))
        print("Area Total: {} Area Normal: {} Area Transfer: {}".format(areaTotal,areaNormal,areaTransfer))
        file.write("--------------------------------------------------------------------------------------------------------------\n")
        file.write("######################### SOURCE TASK: {}\n".format(n))
        file.write("Area Total: {} Area Normal: {} Area Transfer: {}\n".format(areaTotal,areaNormal,areaTransfer))
        ratioNormal = areaNormal/areaTotal
        ratioTransfer = areaTransfer/areaTotal
        ratioNormal2 = areaNormal2/areaTotal
        ratioNormal3 = areaNormal3/areaTotal
        ratioNormal4 = areaNormal4/areaTotal
        ratioNormal5 = areaNormal5/areaTotal
        regrets.append(ratioTransfer)

        print("Ratio Normal: {} Ratio Normal2: {} Ratio Normal3: {} Ratio Normal4: {} Ratio Normal5: {} Ratio Transfer: {} ".format(ratioNormal,ratioNormal2,ratioNormal3,ratioNormal4,ratioNormal5,ratioTransfer))
        file.write("Ratio Normal: {} Ratio Normal2: {} Ratio Normal3: {} Ratio Normal4: {} Ratio Normal5: {} Ratio Transfer: {} \n".format(ratioNormal,ratioNormal2,ratioNormal3,ratioNormal4,ratioNormal5,ratioTransfer))
        ratioDif = ratioTransfer - ratioNormal
        if ratioDif < 0:
            ratioDif = 0
        maxRegret.append(ratioDif)

        maxRewardNormal = max(evaluationNormal[n])
        maxRewardNormal2 = max(evaluationNormal2[n])
        maxRewardNormal3 = max(evaluationNormal3[n])
        maxRewardNormal4 = max(evaluationNormal4[n])
        maxRewardNormal5 = max(evaluationNormal5[n])
        maxRewardTransfer = max(evaluationTransfer[n])

        print("Max Reward Normal: {} Max Reward Normal2: {} Max Reward Normal3: {} Max Reward Normal4: {} Max Reward Normal5: {} Max Reward Transfer: {}".format(maxRewardNormal,maxRewardNormal2,maxRewardNormal3,maxRewardNormal4,maxRewardNormal5,maxRewardTransfer))
        file.write("Max Reward Normal: {} Max Reward Normal2: {} Max Reward Normal3: {} Max Reward Normal4: {} Max Reward Normal5: {} Max Reward Transfer: {}\n".format(maxRewardNormal,maxRewardNormal2,maxRewardNormal3,maxRewardNormal4,maxRewardNormal5,maxRewardTransfer))

        indsNormal = np.argwhere(evaluationNormal[n] > INITIAL_TRESHOLD)
        indsTransfer = np.argwhere(evaluationTransfer[n] > INITIAL_TRESHOLD)
        indsNormal2 = np.argwhere(evaluationNormal2[n] > INITIAL_TRESHOLD)
        indsNormal3 = np.argwhere(evaluationNormal3[n] > INITIAL_TRESHOLD)
        indsNormal4 = np.argwhere(evaluationNormal4[n] > INITIAL_TRESHOLD)
        indsNormal5 = np.argwhere(evaluationNormal5[n] > INITIAL_TRESHOLD)
        eptotresholdNormal = 0
        eptotresholdTransfer = 0
        eptotresholdNormal2 = 0
        eptotresholdNormal3 = 0
        eptotresholdNormal4 = 0
        eptotresholdNormal5 = 0
        if len(indsNormal) > 0:
            eptotresholdNormal = min(indsNormal)
        if len(indsNormal2) > 0:
            eptotresholdNormal2 = min(indsNormal2)
        if len(indsNormal3) > 0:
            eptotresholdNormal3 = min(indsNormal3)
        if len(indsNormal4) > 0:
            eptotresholdNormal4 = min(indsNormal4)
        if len(indsNormal5) > 0:
            eptotresholdNormal5 = min(indsNormal5)
        if len(indsTransfer) > 0:
            eptotresholdTransfer = min(indsTransfer)
        print("TTT Normal: {} TTT Normal2: {} TTT Normal3: {} TTT Normal4: {} TTT Normal5: {} TTT Transfer: {}".format(eptotresholdNormal,eptotresholdNormal2,eptotresholdNormal3,eptotresholdNormal4,eptotresholdNormal5,eptotresholdTransfer))
        file.write("TTT Normal: {} TTT Normal2: {} TTT Normal3: {} TTT Normal4: {} TTT Normal5: {} TTT Transfer: {}".format(eptotresholdNormal,eptotresholdNormal2,eptotresholdNormal3,eptotresholdNormal4,eptotresholdNormal5,eptotresholdTransfer))

        

        colum = str(0+1) + str(1)
        columnum = int(colum)

        plt.figure(columnum)


        time_series_df = pd.DataFrame(averageRewardsNormal[n])
        smooth_path    = time_series_df.rolling(20).mean()
        path_deviation = 2 * time_series_df.rolling(20).std()
        plt.ion()
        plt.show()
        plt.plot(smooth_path, linewidth=2, label="S1", color = 'b')
        plt.fill_between(path_deviation.index, (smooth_path-path_deviation)[0], (smooth_path+path_deviation)[0], color='b', alpha=.1)
        jumpstart_normal = smooth_path[0][19]
        
        ########################TS######
        time_series_df = pd.DataFrame(averageRewardsNormal2[n])
        smooth_path    = time_series_df.rolling(20).mean()
        path_deviation = 2 * time_series_df.rolling(20).std()
        plt.plot(smooth_path, linewidth=2, color = 'g', label = "S2")
        plt.fill_between(path_deviation.index, (smooth_path-path_deviation)[0], (smooth_path+path_deviation)[0], color='g', alpha=.1)
        jumpstart_normal2 = smooth_path[0][19]

        time_series_df = pd.DataFrame(averageRewardsNormal3[n])
        smooth_path    = time_series_df.rolling(20).mean()
        path_deviation = 2 * time_series_df.rolling(20).std()
        plt.plot(smooth_path, linewidth=2, color = 'y', label= "S3")
        plt.fill_between(path_deviation.index, (smooth_path-path_deviation)[0], (smooth_path+path_deviation)[0], color='y', alpha=.1)
        jumpstart_normal3 = smooth_path[0][19]

        time_series_df = pd.DataFrame(averageRewardsNormal4[n])
        smooth_path    = time_series_df.rolling(20).mean()
        path_deviation = 2 * time_series_df.rolling(20).std()
        plt.plot(smooth_path, linewidth=2, color = 'k', label= "S4")
        plt.fill_between(path_deviation.index, (smooth_path-path_deviation)[0], (smooth_path+path_deviation)[0], color='k', alpha=.1)
        jumpstart_normal4 = smooth_path[0][19]

        time_series_df = pd.DataFrame(averageRewardsNormal5[n])
        smooth_path    = time_series_df.rolling(20).mean()
        path_deviation = 2 * time_series_df.rolling(20).std()
        plt.plot(smooth_path, linewidth=2, color = 'm', label= "S5")
        plt.fill_between(path_deviation.index, (smooth_path-path_deviation)[0], (smooth_path+path_deviation)[0], color='m', alpha=.1)
        jumpstart_normal5 = smooth_path[0][19]

        time_series_df = pd.DataFrame(averageRewardsTransfer[n])
        smooth_path    = time_series_df.rolling(20).mean()
        path_deviation = 2 * time_series_df.rolling(20).std()
        plt.plot(smooth_path, linewidth=2, color = 'r', label="CURR")
        plt.fill_between(path_deviation.index, (smooth_path-path_deviation)[0], (smooth_path+path_deviation)[0], color='r', alpha=.1)
        jumpstart_transfer = smooth_path[0][19]

        plt.legend()
        print("Jumpstart.  Normal: {} Normal2: {} Normal3: {} Normal4: {} Normal5: {} Transfer: {}".format(jumpstart_normal,jumpstart_normal2,jumpstart_normal3,jumpstart_normal4,jumpstart_normal5,jumpstart_transfer))
        file.write("Jumpstart.  Normal: {} Normal2: {} Normal3: {} Normal4: {} Normal5: {} Transfer: {}\n".format(jumpstart_normal,jumpstart_normal2,jumpstart_normal3,jumpstart_normal4,jumpstart_normal5,jumpstart_transfer))
        plt.xlabel('Iterations')
        plt.ylabel('Reward')
        

        plt.draw()
        plt.pause(0.001)
        plt.savefig("Results/Evaluation/MICROGRID_MANY_DATES_BEST_SIMULATED-55: {}".format(1))

        plt.figure((columnum+10))
        time_series_df = pd.DataFrame(evaluationNormal[n])
        smooth_path    = time_series_df.rolling(2).mean()
        path_deviation = 2 * time_series_df.rolling(2).std()
        plt.ion()
        plt.show()
        plt.plot(smooth_path, linewidth=2, label= "S1", color = 'b')
        plt.fill_between(path_deviation.index, (smooth_path-path_deviation)[0], (smooth_path+path_deviation)[0], color='b', alpha=.1)

        
        
        #########################################TR##########################
        time_series_df = pd.DataFrame(evaluationNormal2[n])
        smooth_path    = time_series_df.rolling(2).mean()
        path_deviation = 2 * time_series_df.rolling(2).std()
        plt.plot(smooth_path, linewidth=2, color = 'g', label= "S2")
        plt.fill_between(path_deviation.index, (smooth_path-path_deviation)[0], (smooth_path+path_deviation)[0], color='g', alpha=.1)
    

        time_series_df = pd.DataFrame(evaluationNormal3[n])
        smooth_path    = time_series_df.rolling(2).mean()
        path_deviation = 2 * time_series_df.rolling(2).std()
        plt.plot(smooth_path, linewidth=2, color = 'y', label= "S3")
        plt.fill_between(path_deviation.index, (smooth_path-path_deviation)[0], (smooth_path+path_deviation)[0], color='y', alpha=.1)

        time_series_df = pd.DataFrame(evaluationNormal4[n])
        smooth_path    = time_series_df.rolling(2).mean()
        path_deviation = 2 * time_series_df.rolling(2).std()
        plt.plot(smooth_path, linewidth=2, color = 'k', label= "S4")
        plt.fill_between(path_deviation.index, (smooth_path-path_deviation)[0], (smooth_path+path_deviation)[0], color='k', alpha=.1)

        time_series_df = pd.DataFrame(evaluationNormal5[n])
        smooth_path    = time_series_df.rolling(2).mean()
        path_deviation = 2 * time_series_df.rolling(2).std()
        plt.plot(smooth_path, linewidth=2, color = 'm', label= "S5")
        plt.fill_between(path_deviation.index, (smooth_path-path_deviation)[0], (smooth_path+path_deviation)[0], color='m', alpha=.1)

        time_series_df = pd.DataFrame(evaluationTransfer[n])
        smooth_path    = time_series_df.rolling(2).mean()
        path_deviation = 2 * time_series_df.rolling(2).std()
        plt.plot(smooth_path, linewidth=2, color = 'r', label= "CURR")
        plt.fill_between(path_deviation.index, (smooth_path-path_deviation)[0], (smooth_path+path_deviation)[0], color='r', alpha=.1)

        plt.legend()
        plt.xlabel('Iterations')
        plt.ylabel('Reward')
        plt.draw()
        plt.pause(0.001)
        plt.savefig("Results/Evaluation/MICROGRID_MANY_DATES_EVALUATION_BEST SIMULATED-55: {}".format(n))


        print("BREAKING CURR!")
       
        file.write("BREAKING CURR!\n")
        
  
        
    file.close()
    
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()
