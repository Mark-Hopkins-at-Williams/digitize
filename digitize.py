import pygame as pg
import torch
from training import MNistClassifier

# graphical layout
WINDOW_WIDTH = 790
MARGIN = 60
GRID_WIDTH = int(0.7 * (WINDOW_WIDTH - 3*MARGIN))
WINDOW_HEIGHT = 2 * MARGIN + GRID_WIDTH
BAR_GRAPH_WIDTH = int(0.3 * (WINDOW_WIDTH - 3*MARGIN))
NUM_ROWS = 28
PIXEL_WIDTH = int(GRID_WIDTH / NUM_ROWS)
GRID_TOPLEFT = (MARGIN, MARGIN)
BAR_GRAPH_TOPLEFT = (2*MARGIN + GRID_WIDTH, MARGIN)

# fonts and colors
BG_COLOR = (0, 0, 40)
ON_COLOR = "indigo"
OFF_COLOR = "darkgray"
FONT_COLOR = "navy"
BAR_GRAPH_BAR_COLOR = "indigo"
BAR_GRAPH_FONT = "monospace"


class ProgramState:
    def __init__(self):
        self.currently_drawing = False
        self.selected_pixels = set()
        self.digit_probs = [.1 for _ in range(10)]
        self.grid_colors = ["#00aaff", "#00bbff", "#00ccff", "#00ddff", "#00eeff", "#00ffff"]
        self.current_grid_color = 0
        
    def reset(self):
        self.selected_pixels = set()

    def select_pixel(self, row, col):
        self.selected_pixels.add((row, col))

    def is_selected(self, row, col):
        return (row, col) in self.selected_pixels

    def update_grid_color(self):
        self.current_grid_color += 1
        self.current_grid_color %= len(self.grid_colors)
        
    def grid_color(self):
        return self.grid_colors[self.current_grid_color]

    def get_grid(self):
        matrix = []
        for row in range(28):
            new_row = []
            for col in range(28):
                new_row.append(1. if (row, col) in self.selected_pixels else -1.)
            matrix.append(new_row)
        return torch.tensor(matrix)


def which_pixel_is_at(x, y):
    start_x, start_y = GRID_TOPLEFT
    if start_x < x < start_x + PIXEL_WIDTH * NUM_ROWS:
        if start_y < y < start_y + PIXEL_WIDTH * NUM_ROWS:
            x -= start_x
            y -= start_y
            pixel_row = y // PIXEL_WIDTH
            pixel_col = x // PIXEL_WIDTH
            return (pixel_row, pixel_col)
    return None


def draw_pixel_grid(state, screen):
    for row in range(28):
        for col in range(28):
            fill_color = ON_COLOR if (row, col) in state.selected_pixels else OFF_COLOR
            square = pg.Surface((PIXEL_WIDTH, PIXEL_WIDTH))
            square.fill(state.grid_color())
            pg.draw.rect(square, fill_color, pg.Rect(1, 1, PIXEL_WIDTH-2, PIXEL_WIDTH-2))          
            x = 1 + GRID_TOPLEFT[1] + col*PIXEL_WIDTH
            y = 1 + GRID_TOPLEFT[0] + row*PIXEL_WIDTH
            screen.blit(square, (x, y))


def draw_bar_graph(state, screen):
    width, height = BAR_GRAPH_WIDTH, PIXEL_WIDTH*NUM_ROWS
    bar_width = height / len(state.digit_probs)
    graphic = pg.Surface((width, height))
    graphic.fill("lavender")
    font = pg.font.SysFont(BAR_GRAPH_FONT, 36)
    for i, value in enumerate(state.digit_probs):
        text = font.render(str(i), 1, FONT_COLOR)            
        pg.draw.rect(graphic, BAR_GRAPH_BAR_COLOR, 
                     pg.Rect(20, (i * bar_width) + 1, 
                             value*width, bar_width-2))      
        graphic.blit(text, (0, i * bar_width))
    screen.blit(graphic, (BAR_GRAPH_TOPLEFT[0], BAR_GRAPH_TOPLEFT[1]))



def run(model_file):      
    classifier = MNistClassifier(model_file)
    pg.init()
    pg.display.set_caption("Digitize!")
    screen = pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    screen.fill(BG_COLOR)
    state = ProgramState()
    quit_now = False
    pg.time.set_timer(pg.USEREVENT, 100) # fires every tenth of a second
    while not quit_now:  
        draw_pixel_grid(state, screen)
        draw_bar_graph(state, screen)    
        pg.display.flip()  # moves to the next frame 
        events = list(pg.event.get())
        for event in events:
            if event.type == pg.QUIT:
                quit_now = True
            elif event.type == pg.USEREVENT:
                state.update_grid_color()
                results = classifier.classify(state.get_grid())
                state.digit_probs = results.tolist()
            elif event.type == pg.KEYDOWN and event.key == 27: # ESC key
                state.reset()
            elif event.type == pg.MOUSEBUTTONDOWN:
                state.currently_drawing = True
                pixel = which_pixel_is_at(event.pos[0], event.pos[1])
                if pixel is not None:
                    state.select_pixel(pixel[0], pixel[1])
            elif event.type == pg.MOUSEMOTION and state.currently_drawing:
                pixel = which_pixel_is_at(event.pos[0], event.pos[1])
                if pixel is not None:
                    state.select_pixel(pixel[0], pixel[1])
            elif event.type == pg.MOUSEBUTTONUP:
                state.currently_drawing = False
                


if __name__ == "__main__": 
    import sys
    if len(sys.argv) < 2:
        print("\nUsage: digitize.py <model file>")
        print("Please provide the model file!")
    else:
        run(sys.argv[1])