import sys
import math
import random
import pygame
import time

# Initialize pygame 
pygame.init()
width, height = 1000, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("evolution_simulation")

class Herbivore:
    def __init__(self, color, alpha, beta, fitness, health, move_chance, x, y) -> None:
        self._color = color
        self._red = self.color[0]
        self._blue = self.color[1]
        self._green = self.color[2]
        self._alpha = alpha
        self._beta = beta
        self._fitness = fitness
        self._initial_health = health
        self._health = health
        self._move_chance = move_chance
        self._lerp_t = 0
        self._lerp_duration = 1
        self._target_x = x
        self._target_y = y
        self._x = x
        self._y = y
        self._age = 0

    # Properties
    @property
    def color(self):
        return self._color

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @property
    def fitness(self):
        return self._fitness

    @property
    def health(self):
        return self._health

    # Methods
    def lerp(self, start, end, t):
        return start + t * (end - start)

    def health_change(self, value):
        self._health += value

    def create_offspring(self):
        return Herbivore(
            (
                max(0, min(255, random.randint(self._red - 30, self._red + 30))),
                max(0, min(255, random.randint(self._blue - 30, self._blue + 30))),
                max(0, min(255, random.randint(self._green - 30, self._green + 30))),
            ),
            max(20, min(255, random.randint(self._alpha - 10, self._alpha + 10))),
            max(40, min(255, random.randint(self._beta - 10, self._beta + 10))),
            max(1, min(70, random.randint(self._fitness - 10, self._fitness + 10))),
            max(1, min(20, random.randint(self._initial_health - 10, self._initial_health + 10))),
            max(1, min(255, random.randint(self._move_chance - 10, self._move_chance + 10))),
            self._x,
            self._y,
        )

class Carnivore:
    def __init__(self, color, alpha, smarts, fitness, health, movechance, x, y) -> None:
        self._color = color
        self._red = self.color[0]
        self._blue = self.color[1]
        self._green = self.color[2]
        self._alpha = alpha
        self._smarts = smarts
        self._fitness = fitness
        self._initial_health = health
        self._health = health
        self._movechance = movechance
        self._lerp_t = 0
        self._lerp_duration = 1
        self._target_x = x
        self._target_y = y
        self._x = x
        self._y = y
        self._age = 0
        self._starvation = False

    @property
    def color(self):
        return self._color

    @property
    def alpha(self):
        return self._alpha

    @property
    def fitness(self):
        return self._fitness

    @property
    def health(self):
        return self._health

    def lerp(self, start, end, t):
        return start + t * (end - start)

    def health_change(self, int):
        self._health += int

    def change_x(self, int):
        self._x += int

    def change_y(self, int):
        self._y += int

    def create_offspring(self):
        return Carnivore(
            (max(0, min(255, random.randint(self._red - 10, self._red + 10))),
             max(0, min(255, random.randint(self._blue - 10, self._blue + 10))),
             max(0, min(255, random.randint(self._green - 10, self._green + 10)))),
            max(20, min(255, random.randint(self._alpha - 10, self._alpha + 10))),
            max(1, min(20, random.randint(self._smarts - 10, self._smarts + 10))),
            max(1, min(70, random.randint(self._fitness - 10, self._fitness + 10))),
            max(1, min(20, random.randint(self._initial_health - 10, self._initial_health + 10))),
            max(1, min(255, random.randint(self._movechance - 10, self._movechance + 10))),
            self._x, self._y)

class Plant:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    # Properties
    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y


def spawn_plants(plants, max_plants, width, height):
    while len(plants) < max_plants:
        x = random.randint(0, width)
        y = random.randint(0, height)
        plants.append(Plant(x, y))

# Constants
max_plants = 100
plants = []
spawn_plants(plants, max_plants, width, height)
plant_color = (0, 255, 0)
plant_radius = 2


def draw_plants(plants, screen):
    for plant in plants:
        pygame.draw.circle(screen, plant_color, (plant.x, plant.y), plant_radius)


def move_herbivores(herbivores):
    for herb in herbivores:
        if random.randint(0, herb._move_chance) == herb._move_chance:
            herb._target_x = herb._x + random.randint(-abs(herb.fitness), abs(herb.fitness))
            herb._target_y = herb._y + random.randint(-abs(herb.fitness), abs(herb.fitness))
            herb._lerp_t = 0
            herb.health_change(- 1)
            #herb.health_change(- herb.fitness/herb._initial_health + herb._initial_health*0.02)

        if herb._lerp_t < 1:
            herb._lerp_t += 1 / (herb._lerp_duration * 60)  # Assuming 60 FPS
            herb._x = herb.lerp(herb._x, herb._target_x, herb._lerp_t)
            herb._y = herb.lerp(herb._y, herb._target_y, herb._lerp_t)

def move_carnivores(carnivores):
    for carn in carnivores:
        if random.randint(0, carn._movechance) == carn._movechance:
            carn._target_x = carn._x + random.randint(-abs(carn.fitness), abs(carn.fitness))
            carn._target_y = carn._y + random.randint(-abs(carn.fitness), abs(carn.fitness))
            carn._lerp_t = 0
            # carn.health_change(- (carn.fitness/carn._initial_health + carn._initial_health*0.02))
            carn.health_change(- 2)
            # if carn._starvation:
            #     carn.health_change(- (carn.fitness * 0.1 + carn._initial_health * 0.1) * 4)

        if carn._lerp_t < 1:
            carn._lerp_t += 1 / (carn._lerp_duration * 60)  # Assuming 60 FPS
            carn._x = carn.lerp(carn._x, carn._target_x, carn._lerp_t)
            carn._y = carn.lerp(carn._y, carn._target_y, carn._lerp_t)

def handle_herbivore_plant_interaction(herbivores, plants):
    for herb in herbivores:
        closest_plant = None
        closest_distance = float('inf')

        # Find the closest plant
        for plant in plants:
            distance = math.sqrt((herb._x - plant.x) ** 2 + (herb._y - plant.y) ** 2)
            if distance < closest_distance:
                closest_distance = distance
                closest_plant = plant

        if closest_plant is not None:
            # Eat the closest plant if close enough
            if closest_distance <= 5:
                herb.health_change(herb._initial_health / 2)
                plants.remove(closest_plant)
                spawn_plants(plants, max_plants, width, height)

            # Move towards the closest plant
            elif random.randint(0, herb.alpha) == herb.alpha:
                dx = closest_plant.x - herb._x
                dy = closest_plant.y - herb._y
                if closest_distance != 0:
                    move_x = (dx / closest_distance) * herb.fitness
                    move_y = (dy / closest_distance) * herb.fitness
                else:
                    move_x, move_y = 0, 0

                herb._target_x = herb._x + move_x
                herb._target_y = herb._y + move_y
                herb._lerp_t = 0

        if herb.health <= 0:
            herbivores.remove(herb)

def blackness_of_herb(color):
    r, g, b = color
    distance_from_black = math.sqrt(r**2 + g**2 + b**2)  # calculate Euclidean distance from black (0,0,0)
    return 1 - (distance_from_black // 15)


def handle_carnivore_on_herbivore(herbivores, carnivores):
    # predatores attacking herbivores
    for carn in carnivores:
        closest_prey = None
        closest_distance = float('inf')

        # Find the closest prey
        for herb in herbivores:
            distance = math.sqrt((herb._x - carn._x) ** 2 + (herb._y - carn._y) ** 2)
            if distance < closest_distance:
                closest_distance = distance
                closest_prey = herb
        
        if len(herbivores) / 4 < len(carnivores):
            carn._starvation = True
            if random.randint(0, carn._smarts) == 0:
                closest_prey = None
        else:
            carn._starvation = False

        if closest_prey is not None:
            # Eat the closest prey if close enough
            if closest_distance <= 5:
                carn.health_change(herb._initial_health)
                herbivores.remove(closest_prey)

            # Move towards the closest prey
            elif random.randint(0, carn.alpha) == carn.alpha or blackness_of_herb(closest_prey.color) == 1:
                dx = closest_prey._x - carn._x
                dy = closest_prey._y - carn._y
                if closest_distance != 0:
                    move_x = (dx / closest_distance) * carn.fitness
                    move_y = (dy / closest_distance) * carn.fitness
                else:
                    move_x, move_y = 0, 0

                carn._target_x = carn._x + move_x
                carn._target_y = carn._y + move_y
                carn._lerp_t = 0

        if carn.health <= 0:
            carnivores.remove(carn)

def handle_herbivore_on_carnivore(herbivores, carnivores):
    # predatores attacking herbivores
    for herb in herbivores:
        closest_pred = None
        closest_distance = float('inf')

        # Find the closest prey
        for carn in carnivores:
            distance = math.sqrt((herb._x - carn._x) ** 2 + (herb._y - carn._y) ** 2)
            if distance < closest_distance:
                closest_distance = distance
                closest_pred = carn

        if closest_distance >= 200:
            closest_pred = None

        if closest_pred is not None:
            # Move towards the closest prey
            if random.randint(0, herb.beta) == herb.beta:
                dx = closest_pred._x - herb._x
                dy = closest_pred._y - herb._y
                if closest_distance != 0:
                    move_x = (dx / closest_distance) * herb.fitness
                    move_y = (dy / closest_distance) * herb.fitness
                else:
                    move_x, move_y = 0, 0

                herb._target_x = herb._x - move_x
                herb._target_y = herb._y - move_y
                herb._lerp_t = 0

def reproduce_carnivores(carnivores):
    # Choose a random herbivore to check for reproduction
    if carnivores == [] or len(carnivores) >= 30:
        pass
    else:
        selected_carn = random.choice(carnivores)
        if random.randint(0, 20) == 5:
            offspring = selected_carn.create_offspring()
            carnivores.append(offspring)

def reproduce_herbivores(herbivores):
    # Choose a random herbivore to check for reproduction
    if herbivores == []:
        pass
    else:
        selected_herb = random.choice(herbivores)
        if random.randint(0, 10) == 5:
            offspring = selected_herb.create_offspring()
            herbivores.append(offspring)
            herbivores.append(offspring)

def create_herbivores(herbs):
    # Get cursor position and mouse button state
    cursor_position = pygame.mouse.get_pos()
    mouse_button_down = pygame.mouse.get_pressed()[0]  # Left button
    pygame.key.name
    if mouse_button_down:
        herbs.append(Herbivore(
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 
            20, 30, 10, 30, 5, 
            cursor_position[0], cursor_position[1]
            ))

def handle_key_actions(herbs, carns, event):
    def c():
        oldest_carn = 0
        oldest_carn_age = None
        for carn in carns:
            if carn._age > oldest_carn_age:
                oldest_carn = carn
                oldest_carn_age = carn._age
        print(
            'alpha:', oldest_carn.alpha, '\n',
            'fitness:', oldest_carn.fitness, '\n',
            'initial_health:', oldest_carn._initial_health, '\n',
            'move_chance:', oldest_carn._movechance, '\n'
        )
    def h():
        oldest_herb = 0
        oldest_herb_age = None
        for herb in herbs:
            if herb._age > oldest_herb_age:
                oldest_herb = herb
                oldest_herb_age = herb._age
    keys = {
        pygame.K_c: c,
        pygame.K_h: h
    }
    if event.type == pygame.KEYDOWN:
        if keys.get(event.key, -1) == -1:
            pass
        else:
            keys[event.key]()

def main():
    # Initialize herbivores & carnivores
    carns = [
        Carnivore(
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 
            10, 1, 10, 50, 5, 100, height // 2)
    ]

    herbs = [
         Herbivore(
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 
            20, 30, 15, 50, 5, width - 100, height * 0.75),
         Herbivore(
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 
            20, 30, 15, 50, 5, width - 100, height * 0.25),
    ]

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Draw plants
        draw_plants(plants, screen)

        # Draw and move herbivores
        for herb in herbs:
            pygame.draw.circle(screen, herb.color, (herb._x, herb._y), 5)
        herb._age += 1
        move_herbivores(herbs)

        # Draw and move carnivores
        for carn in carns:
            pygame.draw.circle(screen, carn.color, (carn._x, carn._y), 9)
        carn._age += 1
        move_carnivores(carns)

        # Interaction between herbivores and plants
        handle_herbivore_plant_interaction(herbs, plants)

        # Interaction between carnivores and herbivores
        handle_carnivore_on_herbivore(herbs, carns)

        # Herbivores running from carnivores
        handle_herbivore_on_carnivore(herbs, carns)

        # Reproduction of herbivores and carnivores
        reproduce_carnivores(carns)
        reproduce_herbivores(herbs)

        # Handle mouse button down
        create_herbivores(herbs)
        handle_key_actions(herbs, carns, event)

        pygame.display.flip()
        time.sleep(1 / 60)
        screen.fill((0, 0, 0))

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
