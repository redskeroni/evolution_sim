import numpy as np
import pygame
import random
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

# Initialize Pygame
pygame.init()
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()

class Herbivore:
    def __init__(self, x, y, color=(155, 40, 155), neural_network=None):
        self.x = x
        self.y = y
        self.radius = 5
        self.color = color
        self.r = self.color[0]
        self.g = self.color[1]
        self.b = self.color[2]
        self.age = 0
        self.speed = 30
        self.health = 400
        self.lerp_t = 0
        self.lerp_duration = 1
        self.target_x = x
        self.target_y = y
        self.neural_network = neural_network or self.create_neural_network()
        self.t = 0
        self.hold_pos = (x, y)

    def lerp(self, start, end, t):
        return start + t * (end - start)

    def softmax(self, it):
        e_it = np.exp(it - np.max(it))
        return e_it / e_it.sum()

    def create_neural_network(self):
        # Create a simple neural network with input, hidden, and output layers
        input_layer = np.random.randn(8, 10)
        hidden_layer = np.random.randn(10, 10)
        output_layer = np.random.randn(10, 4)
        return [input_layer, hidden_layer, output_layer]

    def mutate(self):
        # Mutate the neural network by adding small random values
        for layer in self.neural_network:
            layer += np.random.normal(0, 0.2, layer.shape)

    def update_position(self, nearest_plant, second_plant, nearest_pred):
        # Calculate the input features for the neural network
        input_features = np.array([
            (nearest_plant.x - self.x),
            (nearest_plant.y - self.y),
            (second_plant.x - self.x),
            (second_plant.y - self.y),
            (nearest_pred.x - self.x),
            (nearest_pred.y - self.y),
            nearest_plant.evil,
            second_plant.evil,
        ])

        # Calculate the output of the neural network
        input_layer, hidden_layer, output_layer = self.neural_network
        hidden_output = np.tanh(input_features @ input_layer)
        output = np.tanh(hidden_output @ hidden_layer) @ output_layer

        values = [output[0], output[1], output[2], output[3]]
        choice = ['random', 'food', 'second', 'run']
        max_value = max(values)
        max_index = values.index(max_value)
        choice = choice[max_index]

        # Calculate where organism moves
        distance_x = nearest_plant.x - self.x
        distance_y = nearest_plant.y - self.y
        distance = max(1, (distance_x**2 + distance_y**2)**0.5)

        distance_x_2 = second_plant.x - self.x
        distance_y_2 = second_plant.y - self.y
        distance_2 = max(1, (distance_x_2**2 + distance_y_2**2)**0.5)

        run_distance_x = nearest_pred.x - self.x
        run_distance_y = nearest_pred.y - self.y
        run_distance = (run_distance_x**2 + run_distance_y**2)**0.5

        self.t -= 1

        if choice == 'random':
            self.target_x = self.x + random.randint(-abs(self.speed), abs(self.speed))
            self.target_y = self.y + random.randint(-abs(self.speed), abs(self.speed))
            self.lerp_t = 0
            self.health -= 1

        elif choice == 'food':
            self.target_x = self.x + (distance_x / distance) * self.speed
            self.target_y = self.y + (distance_y / distance) * self.speed
            self.lerp_t = 0
            self.age += 1
            self.health -= 1
        
        elif choice == 'second':
            self.target_x = self.x + (distance_x_2 / distance_2) * self.speed
            self.target_y = self.y + (distance_y_2 / distance_2) * self.speed
            self.lerp_t = 0
            self.age += 1
            self.health -= 1

        elif choice == 'run':
            self.target_x = self.x - (run_distance_x / run_distance) * self.speed
            self.target_y = self.y - (run_distance_y / run_distance) * self.speed
            self.lerp_t = 0
            self.age += 1
            self.health -= 1

        if self.lerp_t < 2:
            self.lerp_t += 1 / (self.lerp_duration * 60)
            self.x = self.lerp(self.x, self.target_x, self.lerp_t)
            self.y = self.lerp(self.y, self.target_y, self.lerp_t)

        self.x = max(min(self.x, screen_width - self.radius), self.radius)
        self.y = max(min(self.y, screen_height - self.radius), self.radius)

        if self.t <= 0:
            self.t = 60
            dist_x = self.x - self.hold_pos[0]
            dist_y = self.y - self.hold_pos[1]
            dist = (dist_x**2 + dist_y**2)**0.5
            if dist >= 10:
                self.age += 20
            else:
                self.age -= 20
            self.hold_pos = (self.x, self.y)

def reproduce_and_mutate(herbs, best_herb):
    if len(herbs) <= 10:
        print("reproduced :)")
        new_herbs = [Herbivore(random.randint(0, screen_width), random.randint(0, screen_height), neural_network=best_herb.neural_network) for _ in range(10)]
        for new in new_herbs:
            new.color = (
                max(0, min(255, random.randint(best_herb.r - 15, best_herb.r + 15))),
                max(0, min(255, random.randint(best_herb.g - 15, best_herb.g + 15))),
                max(0, min(255, random.randint(best_herb.b - 15, best_herb.b + 15)))
            )
            new.mutate
            herbs.append(new)
    else:
        selected_herb = random.choice(herbs)
        if random.randint(0, 20) == 0:
            child_neural_network = []
            for layer1, layer2 in zip(selected_herb.neural_network, best_herb.neural_network):
                offspring_layer = (layer1 + layer2) / 2
                child_neural_network.append(offspring_layer)
            child_color = (
                max(0, min(255, random.randint(selected_herb.r - 15, selected_herb.r + 15))),
                max(0, min(255, random.randint(selected_herb.g - 15, selected_herb.g + 15))),
                max(0, min(255, random.randint(selected_herb.b - 15, selected_herb.b + 15)))
            )
            child = Herbivore(selected_herb.x, selected_herb.y, neural_network=child_neural_network, color=child_color)
            child.mutate()

            # Add the child to the herbivores list
            herbs.append(child)

class Carnivore:
    def __init__(self, x, y, color=(200, 40, 40), neural_network=None):
        self.x = x
        self.y = y
        self.radius = 9
        self.color = color
        self.r = self.color[0]
        self.g = self.color[1]
        self.b = self.color[2]
        self.age = 0
        self.speed = 20
        self.health = 500
        self.lerp_t = 0
        self.lerp_duration = 1
        self.target_x = x
        self.target_y = y
        self.neural_network = neural_network or self.create_neural_network()
        self.t = 0
        self.hold_pos = (self.x, self.y)

    def lerp(self, start, end, t):
        return start + t * (end - start)

    def softmax(self, it):
        e_it = np.exp(it - np.max(it))
        return e_it / e_it.sum()

    def create_neural_network(self):
        # Create a simple neural network with input, hidden, and output layers
        input_layer = np.random.randn(3, 5)
        hidden_layer = np.random.randn(5, 5)
        output_layer = np.random.randn(5, 2)
        return [input_layer, hidden_layer, output_layer]

    def mutate(self):
        # Mutate the neural network by adding small random values
        for layer in self.neural_network:
            layer += np.random.normal(0, 0.2, layer.shape)

    def update_position(self, nearest_herb):
        # Calculate the input features for the neural network
        input_features = np.array([
            (nearest_herb.x - self.x),
            (nearest_herb.y - self.y),
            self.health,
        ])

        # Calculate the output of the neural network
        input_layer, hidden_layer, output_layer = self.neural_network
        hidden_output = np.tanh(input_features @ input_layer)
        output = np.tanh(hidden_output @ hidden_layer) @ output_layer

        values = [output[0], output[1]]
        choice = ['random', 'food']
        max_value = max(values)
        max_index = values.index(max_value)
        choice = choice[max_index]

        # Calculate where organism moves
        distance_x = nearest_herb.x - self.x
        distance_y = nearest_herb.y - self.y
        distance = max(1, (distance_x**2 + distance_y**2)**0.5)

        self.t -= 1

        if choice == 'random':
            self.target_x = self.x + random.randint(-abs(self.speed), abs(self.speed))
            self.target_y = self.y + random.randint(-abs(self.speed), abs(self.speed))
            self.lerp_t = 0
            self.health -= 1

        elif choice == 'food':
            self.target_x = self.x + (distance_x / distance) * self.speed
            self.target_y = self.y + (distance_y / distance) * self.speed
            self.lerp_t = 0
            self.age += 1
            self.health -= 1
        
        

        if self.lerp_t < 2:
            self.lerp_t += 1 / (self.lerp_duration * 60)
            self.x = self.lerp(self.x, self.target_x, self.lerp_t)
            self.y = self.lerp(self.y, self.target_y, self.lerp_t)

        self.x = max(min(self.x, screen_width - self.radius), self.radius)
        self.y = max(min(self.y, screen_height - self.radius), self.radius)

        if self.t <= 0:
            self.t = 60
            dist_x = self.x - self.hold_pos[0]
            dist_y = self.y - self.hold_pos[1]
            dist = (dist_x**2 + dist_y**2)**0.5
            if dist >= 10:
                self.age += 5

            self.hold_pos = (self.x, self.y)


def reproduce_and_mutate_pred(preds, best_pred):
    if len(preds) <= 5:
        print("reproduced :)")
        new_preds = [Carnivore(random.randint(0, screen_width), random.randint(0, screen_height), neural_network=best_pred.neural_network) for _ in range(10)]
        for new in new_preds:
            new.color = (
                max(0, min(255, random.randint(best_pred.r - 15, best_pred.r + 15))),
                max(0, min(255, random.randint(best_pred.g - 15, best_pred.g + 15))),
                max(0, min(255, random.randint(best_pred.b - 15, best_pred.b + 15)))
            )
            new.mutate
            preds.append(new)
    else:
        selected_pred = random.choice(preds)
        if random.randint(0, 30) == 0:
            child_neural_network = []
            for layer1, layer2 in zip(selected_pred.neural_network, best_pred.neural_network):
                offspring_layer = (layer1 + layer2) / 2
                child_neural_network.append(offspring_layer)
            child_color = (
                max(0, min(255, random.randint(selected_pred.r - 15, selected_pred.r + 15))),
                max(0, min(255, random.randint(selected_pred.g - 15, selected_pred.g + 15))),
                max(0, min(255, random.randint(selected_pred.b - 15, selected_pred.b + 15)))
            )
            child = Carnivore(selected_pred.x, selected_pred.y, neural_network=child_neural_network, color=child_color)
            child.mutate()

            # Add the child to the herbivores list
            preds.append(child)

class Plant:
    def __init__(self, x, y, EVIL=False):
        self.x = x
        self.y = y
        self.radius = 3
        self.evil = EVIL
        if EVIL:
            self.color = (255, 0, 0)
        else:
            self.color = (0, 255, 0)

def find_closest_plant(herb, plants):
    closest_plant = None
    scnd_closest = None
    scnd_distance = float('inf')
    closest_distance = float('inf')
    for plant in plants:
        distance = math.sqrt((herb.x - plant.x)**2 + (herb.y - plant.y)**2)
        if distance < scnd_distance:
            if distance < closest_distance:
                closest_distance = distance
                closest_plant = plant
            else:
                scnd_distance = distance
                scnd_closest = plant
    return closest_plant, scnd_closest

def draw_plants(plants, screen):
    for plant in plants:
        pygame.draw.circle(screen, plant.color, (plant.x, plant.y), plant.radius)

max_plants = 300
def spawn_plants(plants):
    while len(plants) < max_plants:
        evilchance = random.randint(0, 3)
        if evilchance == 0:
            x = random.randint(0, screen_width)
            y = random.randint(0, screen_height)    
            plants.append(Plant(x, y, True))
        else:
            x = random.randint(0, screen_width)
            y = random.randint(0, screen_height)
            plants.append(Plant(x, y))

def check_collision_and_health(herbs, plants):
    for herb in herbs:
        if herb.health <= 0:
            herbs.remove(herb)
        for plant in plants:
            distance = math.sqrt((herb.x - plant.x)**2 + (herb.y - plant.y)**2)
            if distance <= herb.radius + plant.radius:
                if not plant.evil:
                    herb.health += 50 #min(100, 50 + herb.health)
                    herb.age += 50
                    plants.remove(plant)
                else:
                    if herb in herbs:
                        herb.age -= 50 # punish herbivores for eating
                        herbs.remove(herb) # kills them if they eat an evil plant >:(
                    plants.remove(plant)

def check_predator_collision(herbs, carns):
    for carn in carns:
        if carn.health <= 0:
            carns.remove(carn)
        for herb in herbs:
            distance = math.sqrt((carn.x - herb.x)**2 + (carn.y - herb.y)**2)
            if distance <= herb.radius + carn.radius:
                carn.health += 50
                herbs.remove(herb) # eats the herby

def find_best_herbs(herbs, prev_best):
    if prev_best == None:
        best_herb = None
        best_herb_age = 0
        for herb in herbs:
            if herb.age >= best_herb_age:
                best_herb = herb
                best_herb_age = herb.age
    else:
        best_herb = prev_best
        best_herb_age = prev_best.age
        for herb in herbs:
            if herb.age >= best_herb_age:
                best_herb = herb
                best_herb_age = herb.age
    return best_herb

def visualize_neural_network(neural_network):
    # Define colors for different layer types
    input_color = 'blue'
    hidden_color = 'green'
    output_color = 'red'

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Calculate the positions of the nodes and connections
    layers = len(neural_network)
    max_nodes = max([layer.shape[0] for layer in neural_network])

    # Create a colormap for the connection weights
    cmap = plt.get_cmap('coolwarm')
    norm = mcolors.Normalize(vmin=-1, vmax=1)

    for l, layer in enumerate(neural_network):
        for i in range(layer.shape[0]):
            for j in range(layer.shape[1]):
                x1, y1 = l * 1.5, max_nodes - i
                x2, y2 = (l + 1) * 1.5, max_nodes - j

                weight = layer[i, j]
                color = cmap(norm(weight))

                # Draw connection lines
                ax.plot([x1, x2], [y1, y2], color=color, lw=0.5 + 1.5 * np.abs(weight), alpha=0.5, zorder=1)

                # Draw nodes
                if l == 0:
                    ax.add_patch(mpatches.Circle((x1, y1), 0.2, color=input_color, zorder=2))
                else:
                    ax.add_patch(mpatches.Circle((x1, y1), 0.2, color=hidden_color, zorder=2))

                if l == layers - 1:
                    ax.add_patch(mpatches.Circle((x2, y2), 0.2, color=output_color, zorder=2))
                else:
                    ax.add_patch(mpatches.Circle((x2, y2), 0.2, color=hidden_color, zorder=2))

    # Add input and output labels
    input_labels = ['food_x', 'food_y', 'food2_x', 'food2_y', 'pred_x', 'pred_y', 'color', 'color2', 'health']
    for i, label in enumerate(input_labels):
        ax.text(-0.5, max_nodes - i, label, fontsize=10, ha='right', va='center')
    
    output_labels = ['random', 'food', 'food2','run']
    for i, label in enumerate(output_labels):
        ax.text(layers - 0.5, max_nodes - i, label, fontsize=10, ha='left', va='center')

    # Set axis limits and aspect ratio
    ax.set_xlim(-0.5, (layers - 1) * 3 + 0.5)
    ax.set_ylim(-0.5, max_nodes - 0.5)
    ax.set_aspect('equal')

    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])

    # Add a colorbar to represent the weights
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.1, aspect=100)

    # Show the plot
    plt.show()


def main():
    herbs = [Herbivore(random.randint(0, screen_width), random.randint(0, screen_height)) for _ in range(11)]
    carns = [Carnivore(random.randint(0, screen_width), random.randint(0, screen_height)) for _ in range(11)]
    plants = []
    best = None
    best_carn = None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    for carn in carns:
                        carn.mutate
                if event.key == pygame.K_LEFT:
                    visualize_neural_network(best_carn.neural_network)
                if event.key == pygame.K_DOWN:
                    for herb in herbs:
                        herb.mutate
                if event.key == pygame.K_UP:
                    visualize_neural_network(best.neural_network)

        
        spawn_plants(plants)
        draw_plants(plants, screen)

        for herb in herbs:
            pygame.draw.circle(screen, herb.color, (herb.x, herb.y), herb.radius)
            herb.age += 1
            near_plant, scnd_plant = find_closest_plant(herb, plants)
            near_pred, null = find_closest_plant(herb, carns)
            herb.update_position(near_plant, scnd_plant, near_pred)
        
        for carn in carns:
            pygame.draw.circle(screen, carn.color, (carn.x, carn.y), carn.radius)
            carn.age += 1
            near_herb, null = find_closest_plant(carn, herbs)
            carn.update_position(near_herb)

        check_predator_collision(herbs, carns)
        check_collision_and_health(herbs, plants)
        best = find_best_herbs(herbs, best)
        best_carn = find_best_herbs(carns, best_carn)
        reproduce_and_mutate(herbs, best)
        reproduce_and_mutate_pred(carns, best_carn)

        pygame.display.flip()
        clock.tick(60)
        screen.fill((0,0,0))
    # Assuming you have the best organism's neural network stored in a variable called best_neural_network
    print(best.neural_network)
    visualize_neural_network(best.neural_network)
    

    
    
if __name__ == "__main__":
    main()
