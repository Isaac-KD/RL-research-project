from Simulateurs.AbstractSimulator import AbstractSimulator
from typing import List, Any, Tuple
import math
import random
import numpy as np
import copy
from math import sqrt, atan2
import matplotlib.pyplot as plt
# ---------------------------
# Global Constants and Utility Functions
# ---------------------------
FRICTION = 0.85
BOOST_SPEED = 650   # Boost thrust value
MAX_THRUST = 100
MAX_ANGLE_CHANGE = 18  # Maximum rotation per turn (in degrees)
CHECKPOINT_RADIUS = 600
laps = 3  # Number of laps (example value)
map_center = [8000, 4500]   # Center of the map

def map_value(v, frommin, frommax, tomin, tomax):
    return tomin + (tomax - tomin) * (v - frommin) / (frommax - frommin)

def calc_absolute_angle(x1, y1, x2, y2):
    vector_1 = np.array([x1, y1])
    vector_2 = np.array([x2, y2])
    norm1 = np.clip(np.linalg.norm(vector_1), 0.00001, 1e8)
    norm2 = np.clip(np.linalg.norm(vector_2), 0.00001, 1e8)
    unit_vector_1 = vector_1 / norm1
    unit_vector_2 = vector_2 / norm2
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    dot_product = np.clip(dot_product, -1, 1)
    return np.degrees(np.arccos(dot_product))

def calc_angle(x1, y1, x2, y2):
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))

# ---------------------------
# Pod Simulation Class (with Physics Update)
# ---------------------------
class PodSim:
    def __init__(self, x, y, vx, vy, angle, loop=0, boost=1):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.angle = angle  # in degrees
        self.loop = loop    # number of laps completed
        self.boost = boost  # boost availability

    def rotate_towards(self, target_x, target_y):
        # Rotate current angle toward the target point, limited by MAX_ANGLE_CHANGE.
        desired_angle = math.degrees(math.atan2(target_y - self.y, target_x - self.x)) % 360
        current_angle = self.angle % 360
        diff = (desired_angle - current_angle + 180) % 360 - 180
        if diff > MAX_ANGLE_CHANGE:
            diff = MAX_ANGLE_CHANGE
        elif diff < -MAX_ANGLE_CHANGE:
            diff = -MAX_ANGLE_CHANGE
        self.angle = (current_angle + diff) % 360

    def update_physics(self, thrust_value):
        # Apply thrust in the direction of the current angle.
        if thrust_value == "BOOST":
            thrust = BOOST_SPEED
        else:
            thrust = float(thrust_value)
        rad = math.radians(self.angle)
        ax = math.cos(rad) * thrust
        ay = math.sin(rad) * thrust
        self.vx = (self.vx + ax) * FRICTION
        self.vy = (self.vy + ay) * FRICTION
        self.x += self.vx
        self.y += self.vy

    def reaches_checkpoint(self, checkpoint):
        dx = checkpoint[0] - self.x
        dy = checkpoint[1] - self.y
        d = math.sqrt(dx * dx + dy * dy)
        return d <= CHECKPOINT_RADIUS

    def __repr__(self):
        return (f"PodSim(x={self.x:.2f}, y={self.y:.2f}, vx={self.vx:.2f}, vy={self.vy:.2f}, "
                f"angle={self.angle:.2f}°, loop={self.loop}, boost={self.boost})")

# ---------------------------
# Simulator Class Implementation
# ---------------------------

class PodSimulator(AbstractSimulator):
    def __init__(self, pod: PodSim):
        self.pod = pod
        self.reset()
        self.dist_total = 0
        self.tour=0
    #@staticmethod
    def init_map(self):
        """
        Creates a list of checkpoints with a variable length between 3 and 5.
        Each checkpoint is represented by a pair of random coordinates (x, y).
        """
        N: int = random.randint(3, 5)   # Number of checkpoints
        l = [(random.randint(0, 16000), random.randint(0, 9000)) for _ in range(N)]
        self.dist_total = 0
        for i in range(len(l)):
            self.dist_total += np.sqrt((l[i][0]-l[(i+1)%N][0])**2+(l[i][1]-l[(i+1)%N][1])**2)*laps  # modulo N car il doit retroun a poitn de depart a la fin du circuit
        return l

    def set_checkpoint(self, checkpoint: int) -> None:
        """
        Set the current checkpoint coordinate.
        """
        self.current_checkpoint = checkpoint

    def get_legal_actions(self, state: Any) -> List[Any]:
        """
        Returns 6 possible actions:
        (50, False), (80, False), (100, False), (50, True), (80, True), (100, True)
        where each tuple is (thrust, boost_flag).
        """
        actions = []
        for thrust in [1,25,50,75, 100]:
            for boost in [True]:#[False,True]:
                actions.append((thrust, boost))
        return actions

    def take_action(self, action: Any, player: Any) -> Any:
        """
        Executes the action:
          - Rotates the pod toward the current checkpoint.
          - Updates physics with the given thrust (or boost).
          - If the pod reaches the current checkpoint, increments the lap count.
        """
        thrust, boost_flag = action
        cp_x, cp_y = self.checkpoints[self.current_checkpoint]
        self.pod.rotate_towards(cp_x, cp_y)

        if boost_flag and self.pod.boost :
            thrust_value = "BOOST"
            self.pod.boost = 0
        else : 
            thrust_value = str(thrust)

        self.pod.update_physics(thrust_value)

        return self.get_discreet_state()

    def get_state(self) -> Any:
        """
        Returns the current state of the simulator as a dictionary.
        """
        return {
            "x": self.pod.x,
            "y": self.pod.y,
            "vx": self.pod.vx,
            "vy": self.pod.vy,
            "angle": self.pod.angle,
            "lap": self.pod.loop,
            "current_checkpoint": self.current_checkpoint
        }

    def get_discreet_state(self) -> int:
        """
        Returns a discrete state based on the distance from the current checkpoint.
        The state is determined by the following distance ranges:
        - 0: if the pod is within 600 units of the checkpoint
        - 1: if the pod is between 600 and 900 units from the checkpoint
        - 2: if the pod is between 900 and 1200 units
        - 3: if the pod is between 1200 and 1500 units
        - 4: if the pod is farther than 1500 units
        """
        dx = self.checkpoints[self.current_checkpoint][0] - self.pod.x
        dy = self.checkpoints[self.current_checkpoint][1] - self.pod.y  # Corrected index for y-coordinate
        d = math.sqrt(dx ** 2 + dy ** 2)
        
        cp_x, cp_y = self.checkpoints[self.current_checkpoint]
        angle_to_cp = calc_angle(self.pod.x, self.pod.y, cp_x, cp_y)
        angle_diff = abs((angle_to_cp - self.pod.angle + 180) % 360 - 180) # Différence dans [-180, 180]
        v= abs(self.pod.vx)+abs(self.pod.vy)
        
        if v<= 300:
            state_v =0
        elif v <=400:
            state_v =1
        else:
            state_v = 2
            
  
             
        if angle_diff <=10:
            angle=10
        elif angle_diff <= 45:
            angle = 45
        elif angle_diff <=90:
            angle = 90
        else: 
            angle=180
            
        if d <= 700:
            return (0,angle,state_v)
        if d<=900:
            return (1,angle,state_v)
        elif d <=1200:
            return (2,angle,state_v)
        elif d <= 1500:
            return (3,angle,state_v)
        
        else:
            return (4,angle,state_v)

    def victory(self, player: Any) -> bool:
        """
        Returns True if the pod has completed the required number of laps.
        """
        #print(" victory",self.pod.loop, "laps", laps*self.N_checkpoints)
        return self.pod.loop >= laps*self.N_checkpoints

    def step(self, action: Any) -> Tuple[Any, int, bool]:
        """
        """
        self.tour+=1
        state = self.take_action(action, None)
        
        #cp_x, cp_y = self.checkpoints[self.current_checkpoint]
        #d = np.sqrt((self.pod.x - cp_x)**2 + (self.pod.y - cp_y)**2)
        #max_dist = 2000  # Distance max considérée
        #rewards = 2 * (1 - min(d / max_dist, 1))  # Entre 0 et 2
        #rewards = [-0.1]*6
        #rewards = [-0.1]*4 + [-0.5]*2
        #rewards = [10, 0, -0.3, -0.5, -0.75,-5]    #sarsa
        #rewards = [10, 1, 1, 1, -1,-10]             # sarsa
        
        rewards = [10,0.75,0.5,0.2,-0.1,-0.5]
        #rewards = [10,5,0.5,0.2,-1]
        #rewards = [10,2,1,0,-1]
        
        done = self.victory(None)
        reward = 10 if self.pod.reaches_checkpoint(self.checkpoints[self.current_checkpoint]) else rewards[state[0]]
        if self.pod.reaches_checkpoint(self.checkpoints[self.current_checkpoint]) :
            self.current_checkpoint = (self.current_checkpoint + 1) % self.N_checkpoints
            self.pod.loop += 1
            
        #if self.tour >1000: return  state, -100, True
        return state, reward, done
    
    def reset(self):
        self.checkpoints = self.init_map()
        self.N_checkpoints = len(self.checkpoints)
        self.current_checkpoint = 1
        self.pod.x = self.checkpoints[0][0]
        self.pod.y = self.checkpoints[0][1]
        self.pod.loop=0
        self.tour=0
        return self.get_discreet_state()

# ---------------------------
# Test Simulation using the Simulator Class
# ---------------------------
def test_simulation():
    # Set initial state (example values)
    initial_x = 10481
    initial_y = 6476
    initial_vx = 0
    initial_vy = 0
    # Compute initial angle toward the given checkpoint (provided later)
    # For the purpose of this test, we set an initial checkpoint.
    initial_checkpoint = (3585, 5205)
    init_angle = math.degrees(math.atan2(initial_checkpoint[1] - initial_y, initial_checkpoint[0] - initial_x)) % 360
    pod = PodSim(initial_x, initial_y, initial_vx, initial_vy, init_angle)
    
    simulator = PodSimulator(pod)
    # Instead of using a global list, we set the checkpoint each step (here it's constant for test)
    simulator.set_checkpoint(0)
    
    steps = 10  # Number of simulation steps
    # For demonstration, we'll choose a constant action (thrust=80, no boost)
    action = (80, False)
    
    for i in range(steps):
        state, reward, done = simulator.step(action)
        print(f"Step {i+1}: state = {state}, reward = {reward}, done = {done}")
        if done:
            print("Victory reached!")
            break

"""
    # Simulation avec boost activé
    print("\n=== Simulation avec BOOST activé ===")
    initial_position = Vec2(0, 0)
    initial_velocity = Vec2(0, 0)
    angle = 0          # 0° signifie que le pod se dirige vers la droite
    thrust = 100       # La valeur n'est pas utilisée puisque boost est True
    boost = True
    steps = 10
    simulate_pod(initial_position, initial_velocity, angle, thrust, boost, steps)
"""
def plays(simulator: PodSimulator, agent: Any,max_steps=3000, delay=0.05,display=True,data=False):
    nb_tour=0
    state = simulator.reset()
    step_count = 0

    # Initialisation du graphique
    if display:
        fig, ax = plt.subplots()
        ax.set_xlim(-16000*0.3, 16000*1.3)
        ax.set_ylim(-9000*0.3, 9000*1.3)
        # Affichage des checkpoints
        checkpoints_x, checkpoints_y = zip(*simulator.checkpoints)
        ax.scatter(checkpoints_x, checkpoints_y, c='red', s=200, label='Checkpoints')

        # Affichage du pod
        pod_marker, = ax.plot([], [], 'bo', markersize=10, label='Pod')
        action_text = ax.text(0, 0, "", fontsize=12, color="blue", bbox=dict(facecolor='white', alpha=0.6))
        angle_text = ax.text(0, 0, "", fontsize=12, color="green", bbox=dict(facecolor='white', alpha=0.6))
        # Activation de la légende
        ax.legend()

    # Démarre l'animation
    while step_count < max_steps:
        legal_actions = simulator.get_legal_actions(state)
        action = agent.choose_action(state,legal_actions)   # random.choice(legal_actions)  # Choix aléatoire de l'action
        state, reward, done = simulator.step(action)

        if display:
            # Mise à jour de la position du pod
            pod_x, pod_y = simulator.pod.x, simulator.pod.y
            pod_marker.set_data([simulator.pod.x], [simulator.pod.y])  # Update the pod position
            # Mise à jour du texte de l'action
            action_text.set_position((pod_x + 500, pod_y + 500))  # Décalé légèrement pour la visibilité
            action_text.set_text(f"Action: {action}")
            
            # Calcul de l'angle entre le pod et le checkpoint actuel
            cp_x, cp_y = simulator.checkpoints[simulator.current_checkpoint]
            angle_to_cp = calc_angle(simulator.pod.x, simulator.pod.y, cp_x, cp_y)
            angle_diff = (angle_to_cp - simulator.pod.angle + 180) % 360 - 180  # Différence dans [-180, 180]

            # Mise à jour de l'affichage de l'angle
            angle_text.set_position((pod_x + 500, pod_y - 500))  # Position ajustée pour la visibilité
            angle_text.set_text(f"Angle: {angle_diff:.2f}° Zone : {state} vitesse = {round(abs(simulator.pod.vx)+abs(simulator.pod.vy),2)}")
            # Rafraîchissement du graphique
            plt.draw()  # Redessine le graphique avec les nouvelles coordonnées
            plt.pause(delay)  # Attente pour simuler l'animation

        if data: print(f"Step {step_count}: Action={action}, State={state}, Reward={reward}")

        if done:
            if data: print("Victoire ! Simulation terminée.")
            break

        step_count += 1
        if not  simulator.pod.loop%simulator.N_checkpoints: nb_tour+=1
        if data: print("Nb tour : ",nb_tour)
        
    if step_count >= max_steps:
        print("Nombre maximal d'étapes atteint. Fin de la simulation.")
        print(action,state)

    if display: plt.show()
    return simulator.dist_total/step_count,step_count

class BotMpr():
    def __init__(self,value):
        self.value = value
    def choose_action(self,state,legal_actions):
        return (self.value,True)
    
if __name__ == "__main__":
    # test_simulation()
    random.seed(42)
    simu = PodSimulator(PodSim(1000, 100, 0, 0, 0, 0, 0))
    print(simu.checkpoints)
    print(simu.get_state())
    print(simu.get_discreet_state())
    #test_angle()