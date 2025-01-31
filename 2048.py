import cupy as cp
import random
import time
import threading
import queue
import pickle
from tkinter import Tk, Canvas
import gym
from gym import spaces
from collections import OrderedDict

# ============ 1. 常數 (constants) ============
GRID_LEN = 5
SIZE = 400
GRID_PADDING = 10

BACKGROUND_COLOR_GAME = "#92877d"
BACKGROUND_COLOR_CELL_EMPTY = "#9e948a"

BACKGROUND_COLOR_DICT = {
    2:      "#eee4da",
    4:      "#ede0c8",
    8:      "#f2b179",
    16:     "#f59563",
    32:     "#f67c5f",
    64:     "#f65e3b",
    128:    "#edcf72",
    256:    "#edcc61",
    512:    "#edc850",
    1024:   "#edc53f",
    2048:   "#edc22e",
    4096:   "#3c3a32",
    8192:   "#3c3a32",
}
CELL_COLOR_DICT = {
    2:      "#776e65",
    4:      "#776e65",
    8:      "#f9f6f2",
    16:     "#f9f6f2",
    32:     "#f9f6f2",
    64:     "#f9f6f2",
    128:    "#f9f6f2",
    256:    "#f9f6f2",
    512:    "#f9f6f2",
    1024:   "#f9f6f2",
    2048:   "#f9f6f2",
    4096:   "#f9f6f2",
}
FONT = ("Verdana", 14, "bold")

VERBOSE = False
def debug_print(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

# ============ 2. 核心遊戲邏輯 (以 CuPy 實作) ============

def new_game(n=GRID_LEN):
    matrix = cp.zeros((n, n), dtype=cp.int32)
    for _ in range(2):
        matrix = add_two(matrix)
    debug_print("初始化遊戲棋盤:")
    debug_print_matrix(matrix)
    return matrix

def add_two(matrix):
    empty_coords = cp.argwhere(matrix == 0)
    if empty_coords.size == 0:
        debug_print("無空格可新增數字。")
        return matrix

    idx = cp.random.randint(0, empty_coords.shape[0])
    i, j = empty_coords[idx]
    rand_val = cp.random.rand()
    val = 2 if rand_val < 0.9 else 4
    matrix[i, j] = val
    debug_print(f"新增值 {val} 在位置 ({int(i)}, {int(j)})")
    return matrix

def game_state(matrix):
    if cp.any(matrix == 2048):
        return 'win'
    if cp.any(matrix == 0):
        return 'not over'
    if cp.any(matrix[:, :-1] == matrix[:, 1:]):
        return 'not over'
    if cp.any(matrix[:-1, :] == matrix[1:, :]):
        return 'not over'
    return 'lose'

def debug_print_matrix(matrix):
    if not VERBOSE:
        return
    mat_cpu = matrix.get()
    print("-" * 20)
    for row in mat_cpu:
        print("\t".join(str(x) if x != 0 else "." for x in row))
    print("-" * 20)

def reverse_matrix(matrix):
    return matrix[:, ::-1]

def reverse_rows(matrix):
    return matrix[::-1, :]

def transpose_matrix(matrix):
    return matrix.T

def compress_gpu(matrix):
    mat_before = matrix.copy()
    new_matrix = cp.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        row = matrix[i]
        nonzeros = row[row != 0]
        new_matrix[i, :nonzeros.size] = nonzeros
    changed = not cp.all(new_matrix == mat_before)
    return new_matrix, changed

def merge_gpu(matrix):
    mat_before = matrix.copy()
    merge_occurred = False
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1] - 1):
            if matrix[i, j] != 0 and matrix[i, j] == matrix[i, j + 1]:
                matrix[i, j] *= 2
                matrix[i, j + 1] = 0
                merge_occurred = True
    changed = not cp.all(matrix == mat_before)
    if VERBOSE:
        print("棋盤在合併前:")
        debug_print_matrix(mat_before)
        print("棋盤在合併後:")
        debug_print_matrix(matrix)
    return matrix, merge_occurred

def move_up_detailed(matrix):
    transposed = transpose_matrix(matrix)
    compressed, changed1 = compress_gpu(transposed)
    merged, merged_flag = merge_gpu(compressed)
    compressed_again, changed3 = compress_gpu(merged)
    final = transpose_matrix(compressed_again)
    changed_any = (changed1 or merged_flag or changed3)
    return final, changed_any, merged_flag

def move_down_detailed(matrix):
    reversed_rows_ = reverse_rows(matrix)
    moved_up, changed_any, merged_flag = move_up_detailed(reversed_rows_)
    final = reverse_rows(moved_up)
    return final, changed_any, merged_flag

def move_left_detailed(matrix):
    compressed_m, changed1 = compress_gpu(matrix)
    merged_m, merged_flag = merge_gpu(compressed_m)
    compressed_again, changed3 = compress_gpu(merged_m)
    final = compressed_again
    changed_any = (changed1 or merged_flag or changed3)
    return final, changed_any, merged_flag

def move_right_detailed(matrix):
    reversed_m = reverse_matrix(matrix)
    moved_left, changed_any, merged_flag = move_left_detailed(reversed_m)
    final = reverse_matrix(moved_left)
    return final, changed_any, merged_flag

def move_up(matrix):
    final, changed_any, _ = move_up_detailed(matrix)
    return final, changed_any

def move_down(matrix):
    final, changed_any, _ = move_down_detailed(matrix)
    return final, changed_any

def move_left(matrix):
    final, changed_any, _ = move_left_detailed(matrix)
    return final, changed_any

def move_right(matrix):
    final, changed_any, _ = move_right_detailed(matrix)
    return final, changed_any


# ============ 3. Gym-like 環境 (簡化) ============
class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = None
        self.game = None
        self.last_action = None
        self.last_merged = False
        self.last_moved = False

    def reset(self):
        self.game = new_game(GRID_LEN)
        debug_print("遊戲重置")
        self.last_action = None
        self.last_merged = False
        self.last_moved = False
        return self._get_obs()

    def step(self, action):
        self.last_action = action
        old_game = self.game.copy()
        changed_any = False
        merged_flag = False

        next_matrix, changed_any, merged_flag = self._apply_action_detailed(old_game, action)
        
        # 若該動作沒有任何改變，就嘗試 fallback 動作（此邏輯只是範例，可自行修改）
        if not changed_any:
            fallback_actions = [0, 1]  # Up / Down
            for fb_act in fallback_actions:
                alt_matrix, alt_changed, alt_merged = self._apply_action_detailed(old_game, fb_act)
                if alt_changed:
                    next_matrix = alt_matrix
                    changed_any = True
                    merged_flag = alt_merged
                    self.last_action = fb_act
                    break

        self.last_merged = merged_flag
        self.last_moved = changed_any

        # 若真的有移動，才在空白處新增 2 or 4
        if changed_any:
            self.game = add_two(next_matrix)
        else:
            self.game = next_matrix

        done = False
        reward = 0.0
        state = game_state(self.game)
        if state == 'win':
            done = True
            reward = 1.0
        elif state == 'lose':
            done = True
            reward = -1.0
        return self._get_obs(), reward, done, {}

    def _apply_action_detailed(self, matrix, action):
        if action == 0:
            return move_up_detailed(matrix)
        elif action == 1:
            return move_down_detailed(matrix)
        elif action == 2:
            return move_left_detailed(matrix)
        elif action == 3:
            return move_right_detailed(matrix)
        else:
            return matrix, False, False

    def _get_obs(self):
        return self.game

    def render(self, mode='human'):
        debug_print_matrix(self.game)

# ============ 4. Expectimax Agent (GPU 版) ============
class ExpectimaxAgent:
    def __init__(self, max_depth=2, cache_size=10000, cache_file='cache.pkl', move_history_file='move_history_incremental.pkl'):
        self.max_depth = max_depth
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.cache_file = cache_file
        self.move_history_file = move_history_file
        self.load_cache()
        self.move_history = []
        self.move_history_fh = open(self.move_history_file, 'ab')

    def load_cache(self):
        try:
            with open(self.cache_file, 'rb') as f:
                self.cache = pickle.load(f)
            debug_print(f"成功載入緩存，包含 {len(self.cache)} 個條目。")
        except (FileNotFoundError, EOFError):
            debug_print("沒有找到緩存文件，開始使用空緩存。")
            self.cache = OrderedDict()

    def save_cache(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
        debug_print(f"緩存已保存，包含 {len(self.cache)} 個條目。")

    def get_action(self, matrix):
        best_move = None
        best_score = -float('inf')
        debug_print("搜尋最佳動作...")

        for action in [0, 1, 2, 3]:
            new_mat, changed = self._apply_action(matrix, action)
            if not changed:
                debug_print(f"動作 {action} 不會改變棋盤，跳過。")
                continue
            score = self.expectimax(new_mat, self.max_depth, is_chance=True)
            debug_print(f"動作 {action} 的評分: {score}")
            if score > best_score:
                best_move = action
                best_score = score

        if best_move is None:
            debug_print("沒有有效的動作，隨機選擇動作。")
            chosen_move = random.choice([0, 1, 2, 3])
        else:
            debug_print(f"選擇動作 {best_move}，評分: {best_score}")
            chosen_move = best_move

        if chosen_move is not None:
            self.record_move(chosen_move)

        return chosen_move

    def expectimax(self, matrix, depth, is_chance):
        state = game_state(matrix)
        if depth == 0 or state in ('win', 'lose'):
            eval_score = self._evaluate(matrix)
            debug_print(f"評估節點，深度: {depth}, 狀態: {state}, 評分: {eval_score}")
            return eval_score

        key = self._generate_key(matrix)
        if key in self.cache:
            self.cache.move_to_end(key)
            debug_print(f"使用緩存中的評分，棋盤狀態鍵: {key}")
            return self.cache[key]

        if not is_chance:
            # Max 節點
            best_score = -float('inf')
            for action in [0, 1, 2, 3]:
                new_mat, changed = self._apply_action(matrix, action)
                if not changed:
                    continue
                score = self.expectimax(new_mat, depth - 1, True)
                if score > best_score:
                    best_score = score
            if best_score == -float('inf'):
                best_score = self._evaluate(matrix)
            self.cache[key] = best_score
            if len(self.cache) > self.cache_size:
                removed_key, removed_val = self.cache.popitem(last=False)
                debug_print(f"移除緩存中的棋盤狀態鍵: {removed_key}")
            return best_score
        else:
            # Chance 節點
            empty_coords = cp.argwhere(matrix == 0)
            if empty_coords.size == 0:
                eval_score = self._evaluate(matrix)
                self.cache[key] = eval_score
                if len(self.cache) > self.cache_size:
                    removed_key, removed_val = self.cache.popitem(last=False)
                    debug_print(f"移除緩存中的棋盤狀態鍵: {removed_key}")
                return eval_score

            expected_score = 0.0
            num_empty = empty_coords.shape[0]
            prob_per_cell = 1.0 / num_empty

            for idx in range(empty_coords.shape[0]):
                i, j = empty_coords[idx]
                i, j = int(i), int(j)
                matrix_2 = matrix.copy()
                matrix_2[i, j] = 2
                s2 = self.expectimax(matrix_2, depth - 1, False)
                matrix_4 = matrix.copy()
                matrix_4[i, j] = 4
                s4 = self.expectimax(matrix_4, depth - 1, False)

                expected_score += prob_per_cell * (0.9 * s2 + 0.1 * s4)

            self.cache[key] = expected_score
            if len(self.cache) > self.cache_size:
                removed_key, removed_val = self.cache.popitem(last=False)
                debug_print(f"移除緩存中的棋盤狀態鍵: {removed_key}")
            return expected_score

    def record_move(self, action):
        timestamp = time.time()
        move = {'action': action, 'timestamp': timestamp}
        self.move_history.append(move)
        try:
            pickle.dump(move, self.move_history_fh)
        except Exception as e:
            print(f"無法保存移動到硬碟: {e}")

    def save_move_history(self):
        try:
            self.move_history_fh.close()
            debug_print(f"移動歷史已保存到 {self.move_history_file}")
        except Exception as e:
            print(f"無法關閉移動歷史檔案: {e}")

    def _apply_action(self, matrix, action):
        if action == 0:
            return move_up(matrix)
        elif action == 1:
            return move_down(matrix)
        elif action == 2:
            return move_left(matrix)
        elif action == 3:
            return move_right(matrix)
        else:
            return matrix, False

    def _evaluate(self, matrix):
        sum_score = cp.sum(matrix).astype(cp.float32)
        max_score = cp.max(matrix).astype(cp.float32) * 2
        empties = cp.sum(matrix == 0).astype(cp.float32) * 10
        diffs = cp.abs(matrix[:, :-1] - matrix[:, 1:])
        smoothness = cp.sum(diffs).astype(cp.float32) * -1
        total_score = sum_score + max_score + empties + smoothness
        return float(total_score)

    def _generate_key(self, matrix):
        mat_cpu = matrix.get()
        return tuple(map(tuple, mat_cpu.tolist()))

# ============ 5. 整個棋盤滑動的 Tkinter GUI ============


class GameGrid:
    def __init__(self, root, update_queue, env):
        self.root = root
        self.update_queue = update_queue
        self.env = env
        self.root.title('2048 - GPU Expectimax')
        self.canvas = Canvas(self.root, width=SIZE, height=SIZE, bg=BACKGROUND_COLOR_GAME)
        self.canvas.grid()
        self.old_matrix = cp.zeros((GRID_LEN, GRID_LEN), dtype=cp.int32)
        self.cell_size = (SIZE - (GRID_LEN + 1) * GRID_PADDING) / GRID_LEN
        self._draw_static_background()

    def _draw_static_background(self):
        self.canvas.delete("background")
        for i in range(GRID_LEN):
            for j in range(GRID_LEN):
                x0 = j * (self.cell_size + GRID_PADDING) + GRID_PADDING
                y0 = i * (self.cell_size + GRID_PADDING) + GRID_PADDING
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                self.canvas.create_rectangle(
                    x0, y0, x1, y1,
                    fill=BACKGROUND_COLOR_CELL_EMPTY,
                    tags="background",
                    outline=""
                )

    def render(self, matrix_gpu):
        """
        渲染：若有合併或有移動，就做逐格動畫；
             否則直接畫出最終畫面。
        """
        new_matrix = matrix_gpu.copy()
        old_cpu = self.old_matrix.get()
        new_cpu = new_matrix.get()

        direction = self.env.last_action
        merged = getattr(self.env, 'last_merged', False)
        moved = getattr(self.env, 'last_moved', False)

        # 只有在真的有「合併」或「有方塊移動」的情況才做動畫
        if direction is not None and (merged or moved):
            self.animate_board(old_cpu, new_cpu, direction)
        else:
            # 否則直接畫出新狀態
            self.canvas.delete("tiles")
            self._draw_board(new_cpu, 0, 0, tag_prefix="tiles")

        self.old_matrix = new_matrix

    def animate_board(self, old_cpu, new_cpu, direction):
        """
        以 "tile-based" 的方式做動畫，確保小數字不會「穿過或撞上」不同的大數字。
        """
        self.canvas.delete("tiles")
        # 先做 tile-based 的移動模擬，求出每個 tile 從 (r, c) 移動到 (new_r, new_c) 的路徑
        movement_map = self.simulate_move_for_animation(old_cpu, direction)

        # 逐步繪製動畫
        steps = 8  # 可自行調整幀數
        for step in range(1, steps + 1):
            progress = step / float(steps)
            self.canvas.delete("tiles")
            # 根據 progress，在舊位置與新位置之間插值
            for (r, c), info in movement_map.items():
                val = info['val']
                nr, nc = info['new_r'], info['new_c']
                # 如果這個 tile 被合併掉(等於最終沒有單獨存在)，那就不畫它
                if info['merged']:
                    continue

                # 計算 tile 的 x, y 漸進位置
                delta_r = nr - r
                delta_c = nc - c
                cur_r = r + delta_r * progress
                cur_c = c + delta_c * progress

                offset_x = cur_c * (self.cell_size + GRID_PADDING) + GRID_PADDING
                offset_y = cur_r * (self.cell_size + GRID_PADDING) + GRID_PADDING
                x0 = offset_x
                y0 = offset_y
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size

                bg_color = BACKGROUND_COLOR_DICT.get(val, "#f9f6f2")
                fg_color = CELL_COLOR_DICT.get(val, "#776e65")
                self.canvas.create_rectangle(
                    x0, y0, x1, y1,
                    fill=bg_color,
                    tags="tiles",
                    outline=""
                )
                self.canvas.create_text(
                    (x0 + x1) / 2,
                    (y0 + y1) / 2,
                    text=str(val),
                    font=FONT,
                    fill=fg_color,
                    tags="tiles"
                )

            self.root.update_idletasks()
            time.sleep(0.02)

        # 最後把真正的新局面完整畫出（包含合併後的值）
        self.canvas.delete("tiles")
        self._draw_board(new_cpu, 0, 0, tag_prefix="tiles")

    def simulate_move_for_animation(self, old_cpu, direction):
        """
        直接計算 tile 在原本矩陣內的移動，而不是轉置或反轉。
        回傳字典 movement_map:  { (r, c): { 'val': v, 'new_r': nr, 'new_c': nc, 'merged': bool } }
        """
        movement_map = {}  # (r, c) -> {...}

        if direction in [0, 1]:  # Up or Down
            movement_map = self.simulate_vertical_move(old_cpu, direction)
        elif direction in [2, 3]:  # Left or Right
            movement_map = self.simulate_horizontal_move(old_cpu, direction)

        return movement_map

    def simulate_vertical_move(self, old_cpu, direction):
        """
        計算「往上」或「往下」的 tile 移動。
        direction: 0 = Up, 1 = Down
        """
        movement_map = {}

        for col in range(GRID_LEN):
            tiles = []
            for row in range(GRID_LEN):
                val = old_cpu[row][col]
                if val != 0:
                    tiles.append({'row': row, 'col': col, 'val': val, 'merged': False})

            if direction == 1:  # Down
                tiles = tiles[::-1]

            new_tiles = []
            skip = False
            i = 0
            while i < len(tiles):
                if skip:
                    skip = False
                    i +=1
                    continue
                if i +1 < len(tiles) and tiles[i]['val'] == tiles[i+1]['val']:
                    new_val = tiles[i]['val'] * 2
                    new_tiles.append({'val': new_val, 'merged': False})
                    # Map the first tile to the new position
                    new_tiles_idx = len(new_tiles) -1
                    new_row = new_tiles_idx if direction ==0 else GRID_LEN -1 - new_tiles_idx
                    movement_map[(tiles[i]['row'], tiles[i]['col'])] = {
                        'val': new_val,
                        'new_r': new_row,
                        'new_c': col,
                        'merged': False
                    }
                    # Map the second tile as merged
                    movement_map[(tiles[i+1]['row'], tiles[i+1]['col'])] = {
                        'val': new_val,
                        'new_r': new_row,
                        'new_c': col,
                        'merged': True
                    }
                    skip = True
                else:
                    new_tiles.append({'val': tiles[i]['val'], 'merged': False})
                    new_tiles_idx = len(new_tiles) -1
                    new_row = new_tiles_idx if direction ==0 else GRID_LEN -1 - new_tiles_idx
                    movement_map[(tiles[i]['row'], tiles[i]['col'])] = {
                        'val': tiles[i]['val'],
                        'new_r': new_row,
                        'new_c': col,
                        'merged': False
                    }
                i +=1

            # Fill the rest with zeros
            while len(new_tiles) < GRID_LEN:
                new_tiles.append({'val':0, 'merged': False})

        return movement_map

    def simulate_horizontal_move(self, old_cpu, direction):
        """
        計算「往左」或「往右」的 tile 移動。
        direction: 2 = Left, 3 = Right
        """
        movement_map = {}

        for row in range(GRID_LEN):
            tiles = []
            for col in range(GRID_LEN):
                val = old_cpu[row][col]
                if val != 0:
                    tiles.append({'row': row, 'col': col, 'val': val, 'merged': False})

            if direction == 3:  # Right
                tiles = tiles[::-1]

            new_tiles = []
            skip = False
            i = 0
            while i < len(tiles):
                if skip:
                    skip = False
                    i +=1
                    continue
                if i +1 < len(tiles) and tiles[i]['val'] == tiles[i+1]['val']:
                    new_val = tiles[i]['val'] *2
                    new_tiles.append({'val': new_val, 'merged': False})
                    # Map the first tile to the new position
                    new_tiles_idx = len(new_tiles) -1
                    new_col = new_tiles_idx if direction ==2 else GRID_LEN -1 - new_tiles_idx
                    movement_map[(tiles[i]['row'], tiles[i]['col'])] = {
                        'val': new_val,
                        'new_r': row,
                        'new_c': new_col,
                        'merged': False
                    }
                    # Map the second tile as merged
                    movement_map[(tiles[i+1]['row'], tiles[i+1]['col'])] = {
                        'val': new_val,
                        'new_r': row,
                        'new_c': new_col,
                        'merged': True
                    }
                    skip = True
                else:
                    new_tiles.append({'val': tiles[i]['val'], 'merged': False})
                    new_tiles_idx = len(new_tiles) -1
                    new_col = new_tiles_idx if direction ==2 else GRID_LEN -1 - new_tiles_idx
                    movement_map[(tiles[i]['row'], tiles[i]['col'])] = {
                        'val': tiles[i]['val'],
                        'new_r': row,
                        'new_c': new_col,
                        'merged': False
                    }
                i +=1

            # Fill the rest with zeros
            while len(new_tiles) < GRID_LEN:
                new_tiles.append({'val':0, 'merged': False})

        return movement_map

    def _draw_board(self, cpu_matrix, offset_x, offset_y, tag_prefix="tiles"):
        for i in range(GRID_LEN):
            for j in range(GRID_LEN):
                val = cpu_matrix[i][j]
                if val != 0:
                    self._draw_single_tile(i, j, val, offset_x, offset_y, tag_prefix)

    def _draw_single_tile(self, row, col, val, offset_x, offset_y, tag_prefix="tiles"):
        base_x0 = col * (self.cell_size + GRID_PADDING) + GRID_PADDING
        base_y0 = row * (self.cell_size + GRID_PADDING) + GRID_PADDING
        x0 = base_x0 + offset_x
        y0 = base_y0 + offset_y
        x1 = x0 + self.cell_size
        y1 = y0 + self.cell_size

        bg_color = BACKGROUND_COLOR_DICT.get(val, "#f9f6f2")
        fg_color = CELL_COLOR_DICT.get(val, "#776e65")
        self.canvas.create_rectangle(
            x0, y0, x1, y1,
            fill=bg_color,
            tags=tag_prefix,
            outline=""
        )
        self.canvas.create_text(
            (x0 + x1)/2,
            (y0 + y1)/2,
            text=str(val),
            font=FONT,
            fill=fg_color,
            tags=tag_prefix
        )
# ============ 6. 主程式 - 執行多局 ============
def run_gpu_expectimax(num_episodes=3, max_depth=2, max_steps_per_episode=1000):
    root = Tk()
    update_queue = queue.Queue()
    env = Game2048Env()
    gui = GameGrid(root, update_queue, env)
    agent = ExpectimaxAgent(max_depth=max_depth)

    def play_episodes():
        try:
            for episode in range(1, num_episodes + 1):
                env.reset()
                update_queue.put(env.game.get())
                debug_print(f"開始回合 {episode}")
                done = False
                steps = 0

                while not done and steps < max_steps_per_episode:
                    steps += 1
                    action = agent.get_action(env.game)
                    _, _, done, _ = env.step(action)
                    update_queue.put(env.game.get())
                    debug_print(f"回合 {episode} 步驟 {steps} 執行動作 {action}")
                    time.sleep(0.05)

                if steps >= max_steps_per_episode:
                    debug_print(f"回合 {episode} 超過最大步數 {max_steps_per_episode}，強制結束。")
                agent.save_cache()
                agent.save_move_history()
        except Exception as e:
            print(f"發生錯誤: {e}")
        finally:
            print("所有回合已完成。")

    def process_queue():
        try:
            while not update_queue.empty():
                matrix_cpu = update_queue.get_nowait()
                matrix_gpu = cp.array(matrix_cpu)
                gui.render(matrix_gpu)
        except queue.Empty:
            pass
        root.after(100, process_queue)

    thread = threading.Thread(target=play_episodes, daemon=True)
    thread.start()
    root.after(100, process_queue)
    root.mainloop()
    agent.save_move_history()

if __name__ == "__main__":
    print("Cupy CUDA Device Count:", cp.cuda.runtime.getDeviceCount())
    cp.show_config()
    VERBOSE = True
    run_gpu_expectimax(num_episodes=3, max_depth=2, max_steps_per_episode=20000)
