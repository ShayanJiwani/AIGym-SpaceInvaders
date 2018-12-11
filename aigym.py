import gym
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
import cv2
import matplotlib.pyplot as plt
from collections import deque
import sys
import random
env = gym.make('SpaceInvaders-v0')

#
# preprocess -> send to dense, fully connected, layer specifying shape

class InvaderNN():

    def __init__(self, env):
        self.env = env
        self.state = env.observation_space
        self.discount = 0.95
        self.epsilon = .9
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.01

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights( self.model.get_weights() )

        self.history_buffer = []
        self.history = deque(maxlen=200)
        self.stack_size = 4



    def set_state(self, observation):
        self.state = observation

    def save_history(self, history_item):
        self.history.append(history_item)

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, 8, 8, subsample=(4, 4), input_shape=(84, 84, 4)))
        model.add(Conv2D(64, 4, 4, subsample=(2, 2)))
        model.add(Conv2D(64, 3, 3))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def fit(self):
        #data = list(map(np.array, list(zip(*random.sample(self.history, 32)))))
        data = random.sample(self.history,32)
        (states, actions, rewards, dones, new_states) = list(map(np.array, list(zip(*data))))
        targets = np.zeros((32, self.env.action_space.n))
        for i in range(32):
            targets[i] = self.model.predict(states[i].reshape((1,84,84,4)), batch_size=1)
            future_action = self.target_model.predict(new_states[i].reshape((1,84,84,4)), batch_size=1)
            targets[i, actions[i]] = rewards[i]
            if not dones[i]:
                targets[i, actions[i]] += .99* np.max(future_action)
        self.model.fit(np.moveaxis(np.array(states)[:4],0,-1), np.array(targets)[:4])


    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, t):
        """Trains network to fit given parameters"""
        batch_size = s_batch.shape[0]
        targets = np.zeros((batch_size, env.action_space.n))
        for i in range(batch_size):
            targets[i] = self.model.predict(s_batch[i].reshape(1, 84, 84, 4), batch_size = 1)
            fut_action = self.target_model.predict(s2_batch[i].reshape(1, 84, 84, 4), batch_size = 1)
            targets[i, a_batch[i]] = r_batch[i]
            if d_batch[i] == False:
                targets[i, a_batch[i]] += self.epsilon_decay * np.max(fut_action)

        loss = self.model.train_on_batch(np.moveaxis(s_batch,0,-1), targets)

        # Print the loss every 10 iterations.
        if t % 10 == 0:
            print("We had a loss equal to ", loss)


    def fit_target(self):
        model_weights = self.model.get_weights()
        target_model_weights = self.target_model.get_weights()
        for i in range(len(model_weights)):
            target_model_weights[i] = .1 * model_weights[i] + (.9) * target_model_weights[i]
        self.target_model.set_weights(target_model_weights)

    def get_action(self, state):
        # epsilon greedy
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            data = state.reshape(1,84,84,4)
            q_actions = self.model.predict(data, batch_size=1)

            return np.argmax(q_actions) # optimal policy

    # resize to (84,84)
    # grayscale
    # dimensions are (84,84,1)
    # remove top 26 rows as they only contain the score
    # concatenate the last four "images" of (84,84) to get 84x84*4 image as input
    def preprocess(self):
        input_buffer = []
        for instance in self.history_buffer:
            observation = cv2.resize(instance, dsize=(84, 110), interpolation=cv2.INTER_CUBIC)
            observation = observation[26::,]
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
            input_buffer.append(observation)
        return np.array(input_buffer)
        # cv2.imshow('Window',observation)
        # cv2.waitKey(0)
        # print(observation)


    def save(self):
        model_json = self.model.to_json()
        with open("model_num.json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("model_num.h5")

    def load(self):
        json_file = open('model_num.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model_num.h5")


invaderNN_model = InvaderNN(env)
scores = []
n_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 100
outfile = open("log.csv","w")
outfile.write("episode,score,mean\n")
done = False

for i_episode in range(n_episodes):
    t = 0
    score = 0
    env.reset()
    done = False
    invaderNN_model.history_buffer = [env.step(5)[0] for _ in range(4)]
    curr_state = invaderNN_model.preprocess()

    while not done:

        if len(invaderNN_model.history_buffer) == 4:
            initial_state = invaderNN_model.preprocess()
            invaderNN_model.history_buffer = []

        action = invaderNN_model.get_action(curr_state)

        for i in range(4):
            #env.render()
            temp_observation, temp_score, temp_done, temp_info = env.step(action)
            score += temp_score
            invaderNN_model.history_buffer.append(temp_observation)
            done = done | temp_done
            t+=1

        new_state = invaderNN_model.preprocess()
        invaderNN_model.save_history((initial_state, action, score, done, new_state))

        if len(invaderNN_model.history) > 100:
            batch = random.sample(invaderNN_model.history, 32)
            s_batch, a_batch, r_batch, d_batch, s2_batch = list(map(np.array, list(zip(*batch))))
            if score >= np.mean(scores):
                invaderNN_model.epsilon *= invaderNN_model.epsilon_decay
                invaderNN_model.epsilon = max(invaderNN_model.epsilon, invaderNN_model.epsilon_min)
                print("better")
                invaderNN_model.fit()
                invaderNN_model.fit_target()


    scores.append(score)
    print("Episode {0} Score: {1} Mean: {2:0.2f} Epsilon: {3:0.2f} History: {4}".format(i_episode, score,np.mean(scores), invaderNN_model.epsilon, len(invaderNN_model.history)))
    outfile.write("{0},{1},{2}\n".format(i_episode, score, np.mean(scores)))
    invaderNN_model.save()

outfile.close()

xi = np.arange(n_episodes)
plt.figure()
plt.title("mean={}".format(np.mean(scores)))
plt.plot(xi,scores,'o', alpha=.3)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.show()
plt.savefig("outfile.png")
#print([(hist['episode'], hist['action'],hist['reward']) for hist in invaderNN_model.history if hist['reward'] > 0])
