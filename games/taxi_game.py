import random

class TaxiEnv:
    def __init__(self):
        self.grid_size = 5
        self.locations = [(0,0), (0,4), (4,0), (4,3)] # R, G, Y, B
        self.reset()

    def reset(self):
        # Стан таксі (row, col)
        self.taxi_row = random.randint(0, self.grid_size - 1)
        self.taxi_col = random.randint(0, self.grid_size - 1)
        
        # Індекс локації пасажира і призначення
        self.pass_idx = random.randint(0, 3)
        self.dest_idx = random.randint(0, 3)
        
        # Переконуємось, що призначення не там, де пасажир
        while self.pass_idx == self.dest_idx:
            self.dest_idx = random.randint(0, 3)
            
        self.pass_in_taxi = False
        return self.get_state()

    def get_state(self):
        # Кодуємо стан в ОДНЕ ЧИСЛО для Q-Table
        # (taxi_row * 100) + (taxi_col * 20) + (pass_idx * 4) + dest_idx
        p_idx = 4 if self.pass_in_taxi else self.pass_idx
        return (self.taxi_row * 100) + (self.taxi_col * 20) + (p_idx * 4) + self.dest_idx

    def step(self, action):
        # Actions: 0=S, 1=N, 2=E, 3=W, 4=Pickup, 5=Dropoff
        reward = -1 # Штраф за кожен крок (щоб поспішав)
        done = False
        
        # Рух (Grid logic)
        if action == 0: self.taxi_row = min(self.taxi_row + 1, 4)
        elif action == 1: self.taxi_row = max(self.taxi_row - 1, 0)
        elif action == 2: self.taxi_col = min(self.taxi_col + 1, 4)
        elif action == 3: self.taxi_col = max(self.taxi_col - 1, 0)
        
        # Логіка пасажира
        pass_loc = self.locations[self.pass_idx]
        dest_loc = self.locations[self.dest_idx]

        # Pickup (Підбір)
        if action == 4:
            if not self.pass_in_taxi and (self.taxi_row, self.taxi_col) == pass_loc:
                self.pass_in_taxi = True
                reward = -1 
            else:
                reward = -10 # Штраф за спробу підібрати повітря

        # Dropoff (Висадка)
        elif action == 5:
            if self.pass_in_taxi and (self.taxi_row, self.taxi_col) == dest_loc:
                reward = 20 # ВЕЛИКА НАГОРОДА (Jackpot)
                done = True
            else:
                reward = -10 # Штраф за висадку не там
                
        return self.get_state(), reward, done