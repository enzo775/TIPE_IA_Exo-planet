import numpy as np
import math
import random

def generate_sample(n_points=10):
    aligned = random.random() < 0.5
    data = []

    if aligned:
        a = random.uniform(0.5, 2)  # pente
        b = random.uniform(-10, 10)  # ordonnée à l'origine
        for _ in range(n_points):
            x = random.uniform(-50, 50)
            y = a * x + b + random.uniform(0, 0.5)
            data.extend([x, y])
    else:
        for _ in range(n_points):
            x = random.uniform(-50, 50)
            y = random.uniform(-50, 50)
            data.extend([x, y])

    return np.array(data), aligned


class AlignmentAgent:
    """
    Agent d'apprentissage par renforcement pour détecter
    si un ensemble de points est aligné sur une droite.
    """

    N_POINTS = 10
    INPUT_SIZE = 2 * N_POINTS

    N_HIDDEN_1 = 32
    N_HIDDEN_2 = 16

    N_ACTIONS = 2  # 0 = non aligné, 1 = aligné

    LEARNING_RATE = 5e-4
    DROPOUT_RATE = 0.1

    EPSILON_INIT = 0.9
    EPSILON_MIN = 0.05
    EPSILON_DECAY = 10_000

    DELTA = 1.0

    def __init__(self):
        """
        Initialise les poids et biais du réseau.
        """
        self.W1 = np.random.randn(self.N_HIDDEN_1, self.INPUT_SIZE) * 0.05
        self.b1 = np.zeros(self.N_HIDDEN_1)

        self.W2 = np.random.randn(self.N_HIDDEN_2, self.N_HIDDEN_1) * 0.05
        self.b2 = np.zeros(self.N_HIDDEN_2)

        self.Wq = np.random.randn(self.N_ACTIONS, self.N_HIDDEN_2) * 0.05
        self.bq = np.zeros(self.N_ACTIONS)

        self.Wp = np.random.randn(self.N_HIDDEN_2) * 0.05
        self.bp = 0.0

    def relu(self, x):
        """
        Fonction d'activation ReLU.

        entrée : float x
        sortie : float max(0, x)
        """
        return max(0.0, x)

    def relu_derivative(self, x):
        """
        Dérivée de ReLU.

        entrée : float x
        sortie : float (1 si x > 0, sinon 0)
        """
        return 1.0 if x > 0 else 0.0

    def sigmoid(self, x):
        """
        Fonction sigmoïde.

        entrée : float x
        sortie : float dans ]0,1[
        """
        return 1.0 / (1.0 + math.exp(-x))

    def forward(self, state, training=True):
        """
        Propagation avant du réseau.

        entrée :
            state : list[float] de taille INPUT_SIZE
            training : bool, applique le dropout si True

        sortie :
            q_values : list[float] de taille N_ACTIONS
        """

        # ----- Couche cachée 1 -----
        self.z1 = [0.0] * self.N_HIDDEN_1
        self.a1 = [0.0] * self.N_HIDDEN_1

        for i in range(self.N_HIDDEN_1):
            s = 0
            for j in range(self.INPUT_SIZE):
                s += self.W1[i, j] * state[j]
            s += self.b1[i]
            self.z1[i] = s
            self.a1[i] = self.relu(s)

        # Dropout
        self.dropout_mask = [1.0] * self.N_HIDDEN_1
        if training:
            for i in range(self.N_HIDDEN_1):
                if random.random() < self.DROPOUT_RATE:
                    self.dropout_mask[i] = 0.0
                    self.a1[i] = 0.0

        # ----- Couche cachée 2 -----
        self.z2 = [0.0] * self.N_HIDDEN_2
        self.a2 = [0.0] * self.N_HIDDEN_2

        for i in range(self.N_HIDDEN_2):
            s = 0
            for j in range(self.N_HIDDEN_1):
                s += self.W2[i, j] * self.a1[j]
            s += self.b2[i]
            self.z2[i] = s
            self.a2[i] = self.relu(s)

        # ----- Q-values -----
        q_values = [0.0] * self.N_ACTIONS
        for i in range(self.N_ACTIONS):
            s = 0
            for j in range(self.N_HIDDEN_2):
                s += self.Wq[i, j] * self.a2[j]
            s += self.bq[i]
            q_values[i] = s

        return q_values

    def predict_probability(self):
        """
        Prédit la probabilité que les points soient alignés.

        entrée : aucune (utilise la dernière propagation)
        sortie : float proba dans ]0,1[
        """
        s = self.bp
        for j in range(self.N_HIDDEN_2):
            s += self.Wp[j] * self.a2[j]
        return self.sigmoid(s)

    def epsilon(self, episode):
        """
        Calcule epsilon décroissant exponentiellement.

        entrée : int episode
        sortie : float epsilon
        """
        return self.EPSILON_MIN + (self.EPSILON_INIT - self.EPSILON_MIN) * \
               math.exp(-episode / self.EPSILON_DECAY)

    def huber_gradient(self, error):
        """
        Gradient de la Huber loss.

        entrée : float error = Q - reward
        sortie : float gradient
        """
        if abs(error) <= self.DELTA:
            return error
        return self.DELTA * (1 if error > 0 else -1)

    def train(self, state, label, episode):
        """
        Effectue une étape d'apprentissage par renforcement.

        entrée :
            state : list[float]
            label : int (0 ou 1)
            episode : int

        sortie : aucune
        """

        q_values = self.forward(state, training=True)
        eps = self.epsilon(episode)

        if np.random.rand() < eps:
            action = random.choice([k for k in range(self.N_ACTIONS)])
        else:
            action = np.argmax(q_values) # 0 if q_values[0] > q_values[1] else 1

        reward = 1 if action == label else -1

        # ----- Erreur TD -----
        error = q_values[action] - reward
        grad = self.huber_gradient(error)

        # ----- Gradient Q -----
        dq = [0.0, 0.0]
        dq[action] = grad

        # ----- Rétropropagation -----
        for j in range(self.N_HIDDEN_2):
            self.Wq[action, j] -= self.LEARNING_RATE * dq[action] * self.a2[j]
        self.bq[action] -= self.LEARNING_RATE * dq[action]

        da2 = [self.Wq[action, j] * dq[action] for j in range(self.N_HIDDEN_2)]
        dz2 = [da2[i] * self.relu_derivative(self.z2[i]) for i in range(self.N_HIDDEN_2)]

        for i in range(self.N_HIDDEN_2):
            for j in range(self.N_HIDDEN_1):
                self.W2[i, j] -= self.LEARNING_RATE * dz2[i] * self.a1[j]
            self.b2[i] -= self.LEARNING_RATE * dz2[i]

        da1 = [0.0] * self.N_HIDDEN_1
        for j in range(self.N_HIDDEN_1):
            for i in range(self.N_HIDDEN_2):
                da1[j] += self.W2[i, j] * dz2[i]
            da1[j] *= self.dropout_mask[j]

        dz1 = [da1[i] * self.relu_derivative(self.z1[i]) for i in range(self.N_HIDDEN_1)]

        for i in range(self.N_HIDDEN_1):
            for j in range(self.INPUT_SIZE):
                self.W1[i, j] -= self.LEARNING_RATE * dz1[i] * state[j]
            self.b1[i] -= self.LEARNING_RATE * dz1[i]

        # ----- Apprentissage probabiliste -----
        p = self.predict_probability()
        dp = p - label

        for j in range(self.N_HIDDEN_2):
            self.Wp[j] -= self.LEARNING_RATE * dp * self.a2[j]
        self.bp -= self.LEARNING_RATE * dp


agent = AlignmentAgent()

for ep in range(40000):
    state, label = generate_sample()
    agent.train(state, label, ep)

# Test
correct = 0
for _ in range(2000):
    s, y = generate_sample()
    q = agent.forward(s, training=False)
    correct += (np.argmax(q) == y)

print("Précision :", correct / 2000)

# Exemple clair
while 1:
    state, _ = generate_sample()
    agent.forward(state, training=False)
    print(_==1)
    print("Probabilité aligné :", agent.predict_probability())
    input()