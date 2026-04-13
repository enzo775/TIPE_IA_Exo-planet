import pyxel
import math

# Taille de la grille
GRID_SIZE = 64
CELL_SIZE = 8  # Taille de chaque "pixel" à l'écran

# Initialisation de Pyxel avec un FPS plus élevé
pyxel.init(GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE, title="Dessin avec pinceau", fps=120)

# Grille de dessin (valeurs entre 0 et 1)
grid = [[0.0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

# Taille et forme du pinceau (rayon du cercle)
BRUSH_RADIUS = 2  # Rayon en nombre de cellules

# Fonction pour dessiner la grille
def draw_grid():
    pyxel.cls(0)
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            gray = int(grid[y][x] * 15)  # 16 niveaux de gris
            pyxel.rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE, gray)

# Fonction pour effacer la grille
def clear_grid():
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            grid[y][x] = 0.0

# Fonction pour exporter la grille
def export_grid():
    print("Matrice exportée (valeurs entre 0 et 1) :")
    for row in grid:
        print(row)

# Fonction pour mettre à jour la grille avec un pinceau en cercle/croix
def update_grid(x, y):
    center_x = x // CELL_SIZE
    center_y = y // CELL_SIZE

    for i in range(-BRUSH_RADIUS, BRUSH_RADIUS + 1):
        for j in range(-BRUSH_RADIUS, BRUSH_RADIUS + 1):
            dx = center_x + i
            dy = center_y + j
            if 0 <= dx < GRID_SIZE and 0 <= dy < GRID_SIZE:
                distance = math.sqrt(i**2 + j**2)
                if distance <= BRUSH_RADIUS:
                    intensity = 1.0 - (distance / BRUSH_RADIUS) * 0.7
                    grid[dy][dx] = max(grid[dy][dx], intensity)

# Boucle principale
def update():
    if pyxel.btnp(pyxel.KEY_Q):
        pyxel.quit()
    if pyxel.btnp(pyxel.KEY_C):
        clear_grid()
    if pyxel.btnp(pyxel.KEY_E):
        export_grid()

    # Gestion de la souris
    x, y = pyxel.mouse_x, pyxel.mouse_y
    if pyxel.btn(pyxel.MOUSE_BUTTON_LEFT):
        update_grid(x, y)

def draw():
    draw_grid()
    # Affichage du curseur (cercle rouge plus gros)
    pyxel.circ(pyxel.mouse_x, pyxel.mouse_y, 2, 8)  # 8 = rouge

pyxel.run(update, draw)
