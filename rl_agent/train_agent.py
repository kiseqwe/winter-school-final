import numpy as np
import random
import time
import os

# --- ЗМІНИ ТУТ: ПРИБИРАЄМО TRY/EXCEPT ---
from taxi_game import TaxiEnv 
# ----------------------------------------

# --- 1. НАЛАШТУВАННЯ (Hyperparameters) ---
EPISODES = 5000        # Скільки ігор зіграє агент для навчання
MAX_STEPS = 99         # Максимум кроків за гру (щоб не зациклився)

LEARNING_RATE = 0.1    # Alpha: наскільки швидко агент забуває старе і вчить нове
DISCOUNT_RATE = 0.9    # Gamma: наскільки важлива майбутня нагорода (0.9 = дуже важлива)

# Параметри дослідження (Exploration vs Exploitation)
EPSILON = 1.0          # Шанс зробити випадковий хід (спочатку 100%)
EPSILON_DECAY = 0.999  # Зменшуємо випадковість з кожною грою
EPSILON_MIN = 0.01     # Мінімальний шанс випадковості (1%)

# --- 2. ІНІЦІАЛІЗАЦІЯ ---
env = TaxiEnv()

# Розмір Q-Table: Кількість станів x Кількість дій
# У нашій грі 500 станів і 6 дій
q_table = np.zeros((500, 6))

print("🚖 Start Training (Q-Learning)...")

# --- 3. ТРЕНУВАЛЬНИЙ ЦИКЛ ---
for episode in range(EPISODES):
    state = env.reset()
    done = False
    
    for step in range(MAX_STEPS):
        # A. Вибір дії (Epsilon-Greedy Strategy)
        if random.uniform(0, 1) < EPSILON:
            action = random.randint(0, 5)  # Exploration: Випадковий тиць
        else:
            action = np.argmax(q_table[state]) # Exploitation: Найкращий відомий хід

        # B. Виконуємо дію
        next_state, reward, done = env.step(action)

        # C. Оновлюємо Q-Table (Формула Беллмана)
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        # Q(s,a) = (1-lr)*Q(s,a) + lr*(reward + gamma*maxQ(s',a'))
        new_value = (1 - LEARNING_RATE) * old_value + LEARNING_RATE * (reward + DISCOUNT_RATE * next_max)
        q_table[state, action] = new_value

        state = next_state

        if done:
            break
            
    # Зменшуємо Epsilon (агент стає впевненішим)
    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY

    # Логування прогресу
    if (episode + 1) % 500 == 0:
        print(f"Episode: {episode + 1} | Epsilon: {EPSILON:.4f}")

print("✅ Training Finished!\n")

# --- 4. DEMO (SHOWTIME) ---
# Зараз ми покажемо, як грає вже навчений агент
input("Натисни Enter, щоб подивитися демо-гру навченого агента...")
os.system('cls' if os.name == 'nt' else 'clear') # Очистити консоль

state = env.reset()
done = False
total_reward = 0
actions_map = ["South 👇", "North 👆", "East 👉", "West 👈", "PICKUP 🎒", "DROPOFF 🏁"]

print("*** 🚖 SMART TAXI DEMO ***")

for step in range(25):
    # Тільки Exploitation (використовуємо знання)
    action = np.argmax(q_table[state])
    
    # Виконуємо дію
    next_state, reward, done = env.step(action)
    
    print(f"Step {step+1}: {actions_map[action]} (Reward: {reward})")
    
    total_reward += reward
    state = next_state
    
    # Маленька пауза для ефекту кіно
    time.sleep(0.5) 
    
    if done:
        print(f"\n🏆 SUCCESS! Total Score: {total_reward}")
        break

if not done:
    print("\n⚠️ Failed to complete in 25 steps.")