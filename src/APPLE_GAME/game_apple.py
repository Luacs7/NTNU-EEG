import pygame
import sys
import random
import time
import logging
import threading
from pythonosc import dispatcher, osc_server
from pylsl import StreamInfo, StreamOutlet
import os
# Logger setup for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Absolute path of the current script's directory
script_directory = os.path.dirname(os.path.abspath(__file__))
# Construct the absolute path to 'src/APPLE_GAME'
apple_game_directory = os.path.join(script_directory, '..', 'APPLE_GAME')
# Change the current working directory to 'src/APPLE_GAME'
os.chdir(apple_game_directory)
# Create a new stream for markers
load_time_seconds_before = .2  # Time between each apple drop
load_time_seconds_marker = .1
load_time_seconds = .6
info = StreamInfo('markers', 'Markers', 1, 1/load_time_seconds, 'string', 'MyMarkerStream')
outlet = StreamOutlet(info)

# Constants
TRAINING_MODE = 2
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
PLAYER_WIDTH = 150
PLAYER_HEIGHT = 150
APPLE_SIZE = 80
BACKGROUND_COLOR = (255, 255, 255)
PLAYER_COLOR = (0, 0, 0)
APPLE_COLOR = (255, 0, 0)
LEFT_HAND_OPEN_PATH = "left_hand_open.png"  # Path to the left hand open image
LEFT_HAND_CLOSED_PATH = "left_hand_closed.png"  # Path to the left hand closed image
RIGHT_HAND_OPEN_PATH = "right_hand_open.png"  # Path to the right hand open image
RIGHT_HAND_CLOSED_PATH = "right_hand_closed.png"  # Path to the right hand closed image
APPLE_IMAGE_PATH = "apple.png"  # Path to the apple image
TREE_IMAGE_PATH = "treee.png"
LOAD_BAR_HEIGHT = 20
LOAD_BAR_COLOR = (0, 255, 0)  # Green color for the load bar
MARKER_BAR_COLOR = (255, 128, 0)  # Orange color for the marker line
FPS = 30

# Initialize global variable
global p
p = 0.8

# OSC server initialization
def handle_data_global(unused_addr, left, right):
    global p
    logger.info(f"Received package: Left: {left}, Right: {right}")
    p = right

def start_osc_server():
    osc_thread = threading.Thread(target=init_osc_server)
    osc_thread.daemon = True
    osc_thread.start()

def init_osc_server():
    disp = dispatcher.Dispatcher()
    disp.map("/data", handle_data_global)
    server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", 9001), disp)
    logger.info("Serving on {}".format(server.server_address))
    server.serve_forever()

class Game:
    def __init__(self,PLAYER_HEIGHT,PLAYER_WIDTH,SCREEN_WIDTH,LOAD_BAR_HEIGHT,SCREEN_HEIGHT,APPLE_SIZE,TRAINING_MODE,load_time_seconds_before,load_time_seconds_marker,load_time_seconds):
        pygame.init()
        self.ldtb = load_time_seconds_before
        self.ldtm = load_time_seconds_marker
        self.ldt = load_time_seconds
        self.TRAINING_MODE = TRAINING_MODE
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT + LOAD_BAR_HEIGHT))
        pygame.display.set_caption("Apple Catcher Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)
        self.marker_sent = False
        self.player_pos = [SCREEN_WIDTH // 2, SCREEN_HEIGHT - PLAYER_HEIGHT]
        self.apple_pos = [self.get_random_apple_position(), 0]
        self.apple_speed = (SCREEN_HEIGHT -APPLE_SIZE/2  )/ (load_time_seconds * FPS)
        self.score = 0
        self.failures = 0
        self.start_time = time.time()
        self.input_processed= True
        self.tree_image = pygame.image.load(TREE_IMAGE_PATH)
        self.marker_not_finished = True
        self.prob = 0.2
        # Load and scale images
        self.left_hand_open = pygame.image.load(LEFT_HAND_OPEN_PATH)
        self.left_hand_open = pygame.transform.scale(self.left_hand_open, (PLAYER_WIDTH, PLAYER_HEIGHT))
        self.left_hand_closed = pygame.image.load(LEFT_HAND_CLOSED_PATH)
        self.left_hand_closed = pygame.transform.scale(self.left_hand_closed, (PLAYER_WIDTH, PLAYER_HEIGHT))
        self.right_hand_open = pygame.image.load(RIGHT_HAND_OPEN_PATH)
        self.right_hand_open = pygame.transform.scale(self.right_hand_open, (PLAYER_WIDTH, PLAYER_HEIGHT))
        self.right_hand_closed = pygame.image.load(RIGHT_HAND_CLOSED_PATH)
        self.right_hand_closed = pygame.transform.scale(self.right_hand_closed, (PLAYER_WIDTH, PLAYER_HEIGHT))
        self.apple_image = pygame.image.load(APPLE_IMAGE_PATH)
        self.apple_image = pygame.transform.scale(self.apple_image, (APPLE_SIZE, APPLE_SIZE))

        self.hand_status = {"left": "closed", "right": "closed"}

    def get_random_apple_position(self):
        return random.choice([SCREEN_WIDTH // 4 - APPLE_SIZE // 2, 3 * SCREEN_WIDTH // 4 - APPLE_SIZE // 2])

    def draw_player(self):
        if self.hand_status["left"] == "open":
            self.screen.blit(self.left_hand_open, (self.player_pos[0] - PLAYER_WIDTH, self.player_pos[1]))
        else:
            self.screen.blit(self.left_hand_closed, (self.player_pos[0] - PLAYER_WIDTH, self.player_pos[1]))

        if self.hand_status["right"] == "open":
            self.screen.blit(self.right_hand_open, (self.player_pos[0], self.player_pos[1]))
        else:
            self.screen.blit(self.right_hand_closed, (self.player_pos[0], self.player_pos[1]))

    def draw_apple(self):
        self.screen.blit(self.apple_image, (self.apple_pos[0], self.apple_pos[1]))

    def draw_background(self):
        self.screen.blit(self.tree_image, (0, 0))

    def update_apple(self):
        self.apple_pos[1] += self.apple_speed
        if self.apple_pos[1] >= SCREEN_HEIGHT:
            self.apple_pos = [self.get_random_apple_position(), 0]
            self.failures += 1
            self.input_processed= True
            self.start_time = time.time()
            self.marker_sent = False
            self.hand_status["left"] = "closed"
            self.hand_status["right"] = "closed"

    def handle_input(self):
        global p
        if self.input_processed == False:
            if self.TRAINING_MODE:
                self.prob = int(self.apple_pos[0]/(SCREEN_WIDTH // 2) + 0.5 )
                self.input_processed = True
            if self.TRAINING_MODE==2:
                self.prob = random.choice([int(self.apple_pos[0]/(SCREEN_WIDTH // 2) + 0.5 ),random.random(),int(self.apple_pos[0]/(SCREEN_WIDTH // 2) + 0.5 )])
                self.input_processed = True
            if self.TRAINING_MODE ==  False :
                self.prob = p
        if self.prob < 0.5 and self.marker_finished:  # Open left hand
            self.hand_status["left"] = "open"
            self.hand_status["right"] = "closed"
            self.check_catch("left")
        if self.prob >= 0.5 and self.marker_finished:  # Open right hand
            self.hand_status["right"] = "open"
            self.hand_status["left"] = "closed"
            self.check_catch("right")

    def check_catch(self, hand):
        if hand == "left" and self.apple_pos[0] < SCREEN_WIDTH // 2:
            print(self.apple_pos[1] + APPLE_SIZE- self.player_pos[1])
            if self.apple_pos[1]  >= self.player_pos[1]:
                self.score += 1
                self.hand_status["left"] = "closed"
                self.apple_pos = [self.get_random_apple_position(), 0]
                self.start_time = time.time()
                self.marker_sent = False

        elif hand == "right" and self.apple_pos[0] >= SCREEN_WIDTH // 2:
            if self.apple_pos[1]  >= self.player_pos[1]:
                self.score += 1
                self.hand_status["right"] = "closed"
                self.apple_pos = [self.get_random_apple_position(), 0]
                self.start_time = time.time()
                self.marker_sent = False
    def draw_scoreboard(self):
        score_text = self.font.render(f'Score: {self.score}', True, (0, 0, 0))
        failures_text = self.font.render(f'Failures: {self.failures}', True, (255, 0, 0))
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(failures_text, (10, 50))

    def draw_load_bar(self):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.ldt:
            elapsed_time = self.ldt  # Cap the elapsed time at load_time_seconds

        load_ratio = elapsed_time / self.ldt
        load_bar_width = SCREEN_WIDTH * load_ratio
        load_bar_rect = pygame.Rect(0, SCREEN_HEIGHT, load_bar_width, LOAD_BAR_HEIGHT)
        pygame.draw.rect(self.screen, LOAD_BAR_COLOR, load_bar_rect)

        marker_position = SCREEN_WIDTH * self.ldtb /self.ldt
        marker_position_end = SCREEN_WIDTH * (self.ldtb + self.ldtm)/self.ldt
        marker_rect = pygame.Rect(marker_position, SCREEN_HEIGHT, 2, LOAD_BAR_HEIGHT)
        pygame.draw.rect(self.screen, MARKER_BAR_COLOR, marker_rect)
        marker_rect_end = pygame.Rect(marker_position_end, SCREEN_HEIGHT, 2, LOAD_BAR_HEIGHT)
        pygame.draw.rect(self.screen, MARKER_BAR_COLOR, marker_rect_end)
        if elapsed_time < self.ldtb  and not(self.marker_sent):
            self.marker_not_finished = True
            self.marker_finished = False
        if elapsed_time >= self.ldtb  and not(self.marker_sent):
            if self.apple_pos[0] < SCREEN_WIDTH // 2:
                outlet.push_sample(["left"])
            else:
                outlet.push_sample(["right"])
            self.marker_sent = True
        if elapsed_time >= self.ldtb + self.ldtm + 0.1 and self.marker_not_finished:
            self.input_processed= False
            self.marker_not_finished = False
        if elapsed_time >= self.ldtb + self.ldtm + 0.1:
            self.marker_finished = True

    def show_menu(self):
        menu_font = pygame.font.SysFont(None, 36)
        title_font = pygame.font.SysFont(None, 54)
        title = title_font.render('Apple Catcher Game', True, (0, 0, 0))
        start_text = menu_font.render('Start Game', True, (0, 0, 0))
        quit_text = menu_font.render('Quit', True, (0, 0, 0))
        
        title_rect = title.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
        start_rect = start_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 100))
        quit_rect = quit_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 150))

        mode_font = pygame.font.SysFont(None, 24)
        mode_0 = mode_font.render('Test', True, (0,0,0))
        mode_1 = mode_font.render('Training', True, (0,0,0))
        mode_2 = mode_font.render('Define', True, (0,0,0))
        mode_0_rect = mode_0.get_rect(center=(SCREEN_WIDTH/4,SCREEN_HEIGHT-100))
        mode_1_rect = mode_1.get_rect(center=(SCREEN_WIDTH/2,SCREEN_HEIGHT-100))
        mode_2_rect = mode_2.get_rect(center=((SCREEN_WIDTH*3)/4,SCREEN_HEIGHT-100))

        while True:
            self.screen.fill(BACKGROUND_COLOR)
            self.draw_background()
            self.screen.blit(title, title_rect)
            self.screen.blit(start_text, start_rect)
            self.screen.blit(quit_text, quit_rect)

            self.screen.blit(mode_0, mode_0_rect)
            self.screen.blit(mode_1, mode_1_rect)
            self.screen.blit(mode_2, mode_2_rect)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if start_rect.collidepoint(mouse_pos):
                        return 'start'
                    elif quit_rect.collidepoint(mouse_pos):
                        pygame.quit()
                        sys.exit()
                    if mode_0_rect.collidepoint(mouse_pos):
                        self.TRAINING_MODE=False
                        mode_0 = mode_font.render('Test', True, (255,0,0))
                        mode_1 = mode_font.render('Training', True, (0,0,0))
                        mode_2 = mode_font.render('Define', True, (0,0,0))
                    if mode_1_rect.collidepoint(mouse_pos):
                        self.TRAINING_MODE = True
                        mode_0 = mode_font.render('Test', True, (0,0,0))
                        mode_1 = mode_font.render('Training', True, (255,0,0))
                        mode_2 = mode_font.render('Define', True, (0,0,0))                       
                    if mode_2_rect.collidepoint(mouse_pos):
                        self.TRAINING_MODE=2
                        mode_0 = mode_font.render('Test', True, (0,0,0))
                        mode_1 = mode_font.render('Training', True, (0,0,0))
                        mode_2 = mode_font.render('Define', True, (255,0,0))


            pygame.display.flip()
            pygame.time.Clock().tick(FPS)

    def run(self):
        running = True
        while running:

            if self.score > 40:
                outlet.push_sample(['STOP'])
                running = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.screen.fill(BACKGROUND_COLOR)
            self.draw_background()
            self.draw_scoreboard()
            self.draw_load_bar()
            self.handle_input()
            self.update_apple()
            self.draw_player()
            self.draw_apple()

            pygame.display.flip()
            self.clock.tick(FPS)
            print(p)


        pygame.quit()
        sys.exit()



if __name__ == "__main__":
    start_osc_server()
    game = Game(PLAYER_HEIGHT,PLAYER_WIDTH,SCREEN_WIDTH,LOAD_BAR_HEIGHT,SCREEN_HEIGHT,APPLE_SIZE,TRAINING_MODE,load_time_seconds_before,load_time_seconds_marker,load_time_seconds)

    if game.show_menu()=='start':
        game.start_time=time.time()
        game.run()