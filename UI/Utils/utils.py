import pygame

IMAGE_PATH = "coleoi.jpg"

WIDTH = 1280
HEIGHT = 720
BG_COLOR = (255, 255, 255)

pygame.init()
screen = pygame.display.set_mode( (WIDTH, HEIGHT) )
image = pygame.image.load(IMAGE_PATH).convert_alpha()

def display():
    screen.fill(BG_COLOR)
    screen.blit(image, (0, 0))

pygame.display.set_caption( 'Image' )
sysfont = pygame.font.get_default_font() #Police d'écriture
font = pygame.font.SysFont(None, 48)
display()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouseX = event.pos[0] 
            mouseY = event.pos[1]
            
    pygame.display.update()

pygame.quit()