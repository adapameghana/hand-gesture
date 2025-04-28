import cv2
import mediapipe as mp
import pygame
import numpy as np
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, 
                       min_tracking_confidence=0.8)

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 600, 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand Gesture Tic-Tac-Toe")
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED   = (255, 0, 0)
GREEN = (0, 255, 0)

# Game variables
grid = np.zeros((3, 3), dtype=int)  # 3x3 board: 1 for X, -1 for O
player = 1  # Player 1 (X) starts; then switches to -1 (O)
last_click_time = 0
SELECTION_DELAY = 2.0  # Seconds to keep finger extended before a move is registered

# Set up font for end-game messages
font = pygame.font.SysFont("Arial", 50)

def draw_grid():
    """Draws the Tic-Tac-Toe grid lines."""
    for i in range(1, 3):
        pygame.draw.line(SCREEN, BLACK, (200 * i, 0), (200 * i, HEIGHT), 15)
    for i in range(1, 3):
        pygame.draw.line(SCREEN, BLACK, (0, 200 * i), (WIDTH, 200 * i), 15)

def draw_marks():
    """Draws X (red) and O (green) marks on the grid based on the grid array."""
    for row in range(3):
        for col in range(3):
            center_x = col * 200 + 100
            center_y = row * 200 + 100
            if grid[row, col] == 1:
                pygame.draw.line(SCREEN, RED, (center_x - 50, center_y - 50),
                                 (center_x + 50, center_y + 50), 15)
                pygame.draw.line(SCREEN, RED, (center_x + 50, center_y - 50),
                                 (center_x - 50, center_y + 50), 15)
            elif grid[row, col] == -1:
                pygame.draw.circle(SCREEN, GREEN, (center_x, center_y), 50, 15)

def detect_cell(x, y):
    """Converts pointer coordinates to a grid cell (row, col)."""
    row = y // 200
    col = x // 200
    if 0 <= row < 3 and 0 <= col < 3:
        return row, col
    return -1, -1

def check_winner():
    """Checks if a winning condition is met and returns winner mark (1 or -1) or 0."""
    # Check rows and columns
    for row in grid:
        if np.sum(row) == 3:
            return 1
        elif np.sum(row) == -3:
            return -1
    for col in grid.T:
        if np.sum(col) == 3:
            return 1
        elif np.sum(col) == -3:
            return -1
    # Check diagonals
    diag1 = np.trace(grid)
    diag2 = np.trace(np.fliplr(grid))
    if diag1 == 3 or diag2 == 3:
        return 1
    if diag1 == -3 or diag2 == -3:
        return -1
    return 0

# Initialize OpenCV capture with resolution 640x480.
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

running = True
while running:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror view.
    frame = cv2.flip(frame, 1)
    # Convert BGR to RGB for MediaPipe processing.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    cursor_pos = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Use landmark 8 (index finger tip) and landmark 6 (index finger PIP) for extension check.
            index_tip = hand_landmarks.landmark[8]
            index_pip = hand_landmarks.landmark[6]
            if index_tip.y < index_pip.y:  # Finger is extended.
                # Convert normalized coordinates to pixel coordinates (640x480).
                x = int(index_tip.x * 640)
                y = int(index_tip.y * 480)
                # Direct mapping without inverting the x-coordinate.
                pygame_x = int(x / 640 * WIDTH)
                pygame_y = int(y / 480 * HEIGHT)
                cursor_pos = (pygame_x, pygame_y)
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
            else:
                cursor_pos = None

    # Convert the frame to a Pygame surface using frombuffer.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_surface = pygame.image.frombuffer(frame_rgb.tobytes(), 
                                              (frame_rgb.shape[1], frame_rgb.shape[0]), 
                                              "RGB")
    video_surface = pygame.transform.scale(video_surface, (WIDTH, HEIGHT))
    
    # Clear the screen and draw the video background.
    SCREEN.fill(WHITE)
    SCREEN.blit(video_surface, (0, 0))
    
    # Draw the Tic-Tac-Toe grid and any existing marks.
    draw_grid()
    draw_marks()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # If pointer is active, simulate a cell selection.
    if cursor_pos:
        row, col = detect_cell(*cursor_pos)
        if row != -1 and col != -1 and grid[row, col] == 0:
            if time.time() - last_click_time > SELECTION_DELAY:
                grid[row, col] = player
                last_click_time = time.time()
                win_mark = check_winner()
                
                if win_mark != 0:
                    winning_player = "X" if win_mark == 1 else "O"
                    message = f"Game Over! {winning_player} wins!"
                    text_surface = font.render(message, True, RED)
                    text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
                    SCREEN.blit(text_surface, text_rect)
                    pygame.display.update()
                    pygame.time.delay(2000)
                    grid = np.zeros((3, 3), dtype=int)
                    player = 1
                
                elif not np.any(grid == 0):  # Check if the board is full and no one won
                    message = "Game Over! It's a tie!"
                    text_surface = font.render(message, True, BLACK)
                    text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
                    SCREEN.blit(text_surface, text_rect)
                    pygame.display.update()
                    pygame.time.delay(2000)
                    grid = np.zeros((3, 3), dtype=int)
                    player = 1
                
                else:
                    player *= -1
    
    pygame.display.update()

cap.release()
pygame.quit()



"""Hand Gesture-Controlled Tic-Tac-Toe is an interactive game that uses real-time hand tracking to play Tic-Tac-Toe. It combines MediaPipe for detecting hand gestures, OpenCV for capturing video, and Pygame for displaying the game. Players use their index finger to point at cells on the grid, and a move is registered when the finger stays extended for 2 seconds. """