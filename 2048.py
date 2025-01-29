import cupy as cp
import random
import time
import threading
import queue
import pickle  # 引入pickle模組來序列化緩存
from tkinter import Tk, Frame, Label, CENTER
import gym
from gym import spaces
from collections import OrderedDict  # 引入有序字典，用於LRU緩存

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

# 是否要印出除錯訊息
VERBOSE = False
def debug_print(*args, **kwargs):
    """可用來控制印出細節，用 VERBOSE 控制是否顯示"""
    if VERBOSE:
        print(*args, **kwargs)

# ============ 2. 核心遊戲邏輯 (以 CuPy 實作) ============
def new_game(n=GRID_LEN):
    """
    回傳一個 Cupy 的 n x n 陣列（初始填 0）並在其中兩個位置生成 2 或 4。
    """
    matrix = cp.zeros((n, n), dtype=cp.int32)
    for _ in range(2):
        matrix = add_two(matrix)
    debug_print("初始化遊戲棋盤:")
    debug_print_matrix(matrix)
    return matrix

def add_two(matrix):
    """
    在隨機空白位置加上 2 或 4。隨機機率：2 ~ 90%，4 ~ 10%。
    使用 GPU 端的方式找到空白位置並挑選。
    """
    # GPU 端找出空白格
    empty_coords = cp.argwhere(matrix == 0)  # shape=(k, 2)
    if empty_coords.size == 0:
        debug_print("無空格可新增數字。")
        return matrix

    # 在 GPU 上隨機挑選一個索引
    idx = cp.random.randint(0, empty_coords.shape[0])
    i, j = empty_coords[idx]
    # 產生 2 或 4，使用 CuPy 的隨機數
    rand_val = cp.random.rand()
    val = 2 if rand_val < 0.9 else 4
    matrix[i, j] = val
    debug_print(f"新增值 {val} 在位置 ({int(i)}, {int(j)})")
    return matrix

def game_state(matrix):
    """
    回傳 'win', 'lose', 或 'not over'。
    使用 GPU 上的向量化比較來檢查。
    """
    # 1. 是否達到 2048
    if cp.any(matrix == 999999):
        return 'win'
    # 2. 是否還有空格
    if cp.any(matrix == 0):
        return 'not over'
    # 3. 檢查水平方向
    if cp.any(matrix[:, :-1] == matrix[:, 1:]):
        return 'not over'
    # 4. 檢查垂直方向
    if cp.any(matrix[:-1, :] == matrix[1:, :]):
        return 'not over'
    return 'lose'

def debug_print_matrix(matrix):
    """在終端列印 (可用於快速偵錯)"""
    if not VERBOSE:
        return
    mat_cpu = matrix.get()
    print("-" * 20)
    for row in mat_cpu:
        print("\t".join(str(x) if x != 0 else "." for x in row))
    print("-" * 20)

def reverse_matrix(matrix):
    """左右翻轉"""
    return matrix[:, ::-1]

def transpose_matrix(matrix):
    """轉置"""
    return matrix.T

def compress_gpu(matrix):
    """
    GPU 版壓縮：將 row 的非零項目往左對齊。
    回傳 (new_matrix, changed)
    """
    mat_before = matrix.copy()
    new_matrix = cp.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        row = matrix[i]
        nonzeros = row[row != 0]
        new_matrix[i, :nonzeros.size] = nonzeros
    changed = not cp.all(new_matrix == mat_before)
    return new_matrix, changed

def merge_gpu(matrix):
    """
    GPU 版合併：相鄰且相同的元素合併到左格，右格歸 0。
    只合併一次，如 [2, 2, 4] => [4, 0, 4]
    回傳 (new_matrix, changed)
    """
    mat_before = matrix.copy()
    # 找出可以合併的位置
    can_merge = (matrix[:, :-1] == matrix[:, 1:]) & (matrix[:, :-1] != 0)
    
    # 使用 cp.where 來正確修改矩陣
    matrix[:, :-1] = cp.where(can_merge, matrix[:, :-1] * 2, matrix[:, :-1])
    matrix[:, 1:] = cp.where(can_merge, 0, matrix[:, 1:])
    
    changed = not cp.all(matrix == mat_before)
    
    if VERBOSE:
        print("棋盤在合併前:")
        debug_print_matrix(mat_before)
        print("棋盤在合併後:")
        debug_print_matrix(matrix)
    
    return matrix, changed

def move_up(matrix):
    transposed = transpose_matrix(matrix)
    compressed, changed1 = compress_gpu(transposed)
    merged, changed2 = merge_gpu(compressed)
    compressed_again, changed3 = compress_gpu(merged)
    final = transpose_matrix(compressed_again)
    return final, (changed1 or changed2 or changed3)

def move_down(matrix):
    reversed_m = reverse_matrix(matrix)
    moved_up, changed_up = move_up(reversed_m)
    final = reverse_matrix(moved_up)
    return final, changed_up

def move_left(matrix):
    compressed_m, changed1 = compress_gpu(matrix)
    merged_m, changed2 = merge_gpu(compressed_m)
    compressed_again, changed3 = compress_gpu(merged_m)
    return compressed_again, (changed1 or changed2 or changed3)

def move_right(matrix):
    reversed_m = reverse_matrix(matrix)
    moved_left, changed_left = move_left(reversed_m)
    final = reverse_matrix(moved_left)
    return final, changed_left

# ============ 3. Gym-like 環境 (簡化) ============
class Game2048Env(gym.Env):
    """
    簡化版 2048 環境，使用 Cupy 陣列做運算。
    """
    def __init__(self):
        super(Game2048Env, self).__init__()
        self.action_space = spaces.Discrete(4)  # 0=上, 1=下, 2=左, 3=右
        self.observation_space = None  # 簡化處理
        self.game = None

    def reset(self):
        self.game = new_game(GRID_LEN)
        debug_print("遊戲重置")
        return self._get_obs()

    def step(self, action):
        old_game = self.game.copy()
        # 如果使用者想要做「左(2)」或「右(3)」時，我們先試試看
        # 如果失敗就再自動嘗試「上(0)」或「下(1)」
        if action in [2, 3]:
            # 第一次嘗試左 or 右
            next_matrix, changed = self._apply_action(old_game, action)
            if not changed:
                # 無法左右 => 嘗試上下
                fallback_actions = [0, 1]  # 0=上, 1=下
                changed_success = False
                for fb_act in fallback_actions:
                    alt_matrix, alt_changed = self._apply_action(old_game, fb_act)
                    if alt_changed:
                        # 找到能動的替代方向
                        next_matrix = alt_matrix
                        changed = True
                        changed_success = True
                        debug_print(f"動作 {action} 無效，改為動作 {fb_act}")
                        break
                if not changed_success:
                    # 完全動不了 => 就保持不變
                    next_matrix = old_game
                    changed = False
                    debug_print(f"動作 {action} 和後備動作 {fallback_actions} 都無效")
        else:
            # 原本上下(0,1)不做特殊fallback
            next_matrix, changed = self._apply_action(old_game, action)
        
        # 如果有改變就加新數字，否則保持
        if changed:
            self.game = add_two(next_matrix)
        else:
            self.game = next_matrix
        
        # 判斷是否結束
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

    def _apply_action(self, matrix, action):
        """把重複的 move_* 呼叫獨立出來方便使用。"""
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

    def _get_obs(self):
        # 觀測直接返回 GPU 端陣列
        return self.game

    def render(self, mode='human'):
        debug_print_matrix(self.game)

# ============ 4. Expectimax Agent (GPU 版) ============
class ExpectimaxAgent:
    """
    使用 Expectimax 搜尋，將棋盤用 Cupy 運算。
    加入持久化記憶化緩存來提升運行速度。
    """
    def __init__(self, max_depth=2, cache_size=10000, cache_file='cache.pkl'):
        self.max_depth = max_depth
        self.cache = OrderedDict()  # 初始化記憶化緩存，使用有序字典實現 LRU 緩存
        self.cache_size = cache_size
        self.cache_file = cache_file
        self.load_cache()
        self.move_history = {}  # 用於記錄移動軌跡

    def load_cache(self):
        """從文件載入緩存（如果存在）。"""
        try:
            with open(self.cache_file, 'rb') as f:
                self.cache = pickle.load(f)
            debug_print(f"成功載入緩存，包含 {len(self.cache)} 個條目。")
        except (FileNotFoundError, EOFError):
            debug_print("沒有找到緩存文件，開始使用空緩存。")
            self.cache = OrderedDict()

    def save_cache(self):
        """將緩存保存到文件。"""
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
        debug_print(f"緩存已保存，包含 {len(self.cache)} 個條目。")

    def get_action(self, matrix):
        """
        嘗試四種動作，選擇期望值最高者。
        matrix: cupy array, shape=(4,4)
        """
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

        # 記錄移動軌跡
        if chosen_move is not None:
            self.record_move(matrix, chosen_move)

        return chosen_move

    def expectimax(self, matrix, depth, is_chance):
        """
        Expectimax 遞迴搜尋。
        is_chance=True 表示隨機節點 (環境產生 2 or 4)。
        使用持久化記憶化緩存來儲存已計算過的棋盤狀態評分。
        """
        state = game_state(matrix)
        if depth == 0 or state in ('win', 'lose'):
            eval_score = self._evaluate(matrix)
            debug_print(f"評估節點，深度: {depth}, 狀態: {state}, 評分: {eval_score}")
            return eval_score

        # 生成棋盤狀態的哈希鍵
        key = self._generate_key(matrix)
        if key in self.cache:
            # 將鍵移至末尾，表示最近使用
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
                # 沒有任何改變 => 直接回傳評估值
                best_score = self._evaluate(matrix)
            # 儲存到緩存
            self.cache[key] = best_score
            # 如果超出緩存大小，移除最早的項目
            if len(self.cache) > self.cache_size:
                removed_key, removed_val = self.cache.popitem(last=False)
                debug_print(f"移除緩存中的棋盤狀態鍵: {removed_key}")
            return best_score
        else:
            # Chance 節點 (在空格處生成 2 or 4)
            empty_coords = cp.argwhere(matrix == 0)
            if empty_coords.size == 0:
                eval_score = self._evaluate(matrix)
                self.cache[key] = eval_score
                if len(self.cache) > self.cache_size:
                    removed_key, removed_val = self.cache.popitem(last=False)
                    debug_print(f"移除緩存中的棋盤狀態鍵: {removed_key}")
                return eval_score

            # 期望值
            expected_score = 0.0
            num_empty = empty_coords.shape[0]
            prob_per_cell = 1.0 / num_empty

            for idx in range(empty_coords.shape[0]):
                i, j = empty_coords[idx]
                i, j = int(i), int(j)

                # 2 出現機率 0.9
                matrix_2 = matrix.copy()
                matrix_2[i, j] = 2
                s2 = self.expectimax(matrix_2, depth - 1, False)

                # 4 出現機率 0.1
                matrix_4 = matrix.copy()
                matrix_4[i, j] = 4
                s4 = self.expectimax(matrix_4, depth - 1, False)

                expected_score += prob_per_cell * (0.9 * s2 + 0.1 * s4)

            # 儲存到緩存
            self.cache[key] = expected_score
            # 如果超出緩存大小，移除最早的項目
            if len(self.cache) > self.cache_size:
                removed_key, removed_val = self.cache.popitem(last=False)
                debug_print(f"移除緩存中的棋盤狀態鍵: {removed_key}")
            return expected_score

    def record_move(self, matrix, action):
        """
        記錄每一步的移動軌跡到字典檔中。
        """
        key = self._generate_key(matrix)
        if key not in self.move_history:
            self.move_history[key] = []
        self.move_history[key].append(action)

    def _apply_action(self, matrix, action):
        """直接呼叫 move_xxx，回傳 (新盤面, 是否改變)"""
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
        """
        簡易評估函式：鼓勵總和/最大值/空格多、懲罰行的差異等。
        完全在 GPU 上運算。
        """
        # 1. 整體總和
        sum_score = cp.sum(matrix).astype(cp.float32)
        # 2. 最大值
        max_score = cp.max(matrix).astype(cp.float32) * 2
        # 3. 空格數
        empties = cp.sum(matrix == 0).astype(cp.float32) * 10  # 增加空格數的權重
        # 4. 行平滑度
        diffs = cp.abs(matrix[:, :-1] - matrix[:, 1:])
        smoothness = cp.sum(diffs).astype(cp.float32) * -1
        # 總評分
        total_score = sum_score + max_score + empties + smoothness
        return float(total_score)

    def _generate_key(self, matrix):
        """
        將棋盤狀態轉換為可哈希的鍵（元組）。
        """
        mat_cpu = matrix.get()
        return tuple(map(tuple, mat_cpu.tolist()))

    def save_move_history(self, file_name='move_history.pkl'):
        """將移動軌跡保存到文件。"""
        with open(file_name, 'wb') as f:
            pickle.dump(self.move_history, f)
        debug_print(f"移動軌跡已保存，包含 {len(self.move_history)} 個條目。")

# ============ 5. Tkinter GUI ============
class GameGrid(Frame):
    def __init__(self, root, update_queue):
        super().__init__(root)
        self.root = root
        self.update_queue = update_queue
        self.grid()
        self.master.title('2048 - GPU Expectimax')
        self.grid_cells = []
        self.init_grid()
        # 初始化棋盤為空
        self.matrix = cp.zeros((GRID_LEN, GRID_LEN), dtype=cp.int32)
        self.update_grid_cells()

    def init_grid(self):
        background = Frame(self, bg=BACKGROUND_COLOR_GAME,
                           width=SIZE, height=SIZE)
        background.grid()
        for i in range(GRID_LEN):
            row_cells = []
            for j in range(GRID_LEN):
                cell = Frame(background,
                             bg=BACKGROUND_COLOR_CELL_EMPTY,
                             width=SIZE/GRID_LEN,
                             height=SIZE/GRID_LEN)
                cell.grid(row=i, column=j, padx=GRID_PADDING, pady=GRID_PADDING)
                label = Label(cell, text="",
                              bg=BACKGROUND_COLOR_CELL_EMPTY,
                              justify=CENTER, font=FONT,
                              width=5, height=2)
                label.grid()
                row_cells.append(label)
            self.grid_cells.append(row_cells)

    def update_grid_cells(self):
        mat_cpu = self.matrix.get()  # 取回 CPU 顯示
        for i in range(GRID_LEN):
            for j in range(GRID_LEN):
                val = mat_cpu[i][j]
                if val == 0:
                    self.grid_cells[i][j].configure(
                        text="",
                        bg=BACKGROUND_COLOR_CELL_EMPTY
                    )
                else:
                    self.grid_cells[i][j].configure(
                        text=str(val),
                        bg=BACKGROUND_COLOR_DICT.get(val, "#f9f6f2"),
                        fg=CELL_COLOR_DICT.get(val, "#776e65")
                    )
        self.update_idletasks()

    def render(self, matrix):
        if not cp.all(self.matrix == matrix):
            self.matrix = matrix
            self.update_grid_cells()

# ============ 6. 主程式 - 執行多局 ============
def run_gpu_expectimax(num_episodes=3, max_depth=2, max_steps_per_episode=1000):
    """
    執行多局 2048，以 Cupy + Expectimax。
    """
    root = Tk()
    update_queue = queue.Queue()
    gui = GameGrid(root, update_queue)

    env = Game2048Env()
    agent = ExpectimaxAgent(max_depth=max_depth)

    def play_episodes():
        try:
            for episode in range(1, num_episodes + 1):
                env.reset()
                update_queue.put(env.game.get())  # 初始棋盤放入隊列
                debug_print(f"開始回合 {episode}")

                done = False
                steps = 0

                while not done and steps < max_steps_per_episode:
                    steps += 1
                    action = agent.get_action(env.game)
                    _, _, done, _ = env.step(action)
                    update_queue.put(env.game.get())
                    debug_print(f"回合 {episode} 步驟 {steps} 執行動作 {action}")
                    time.sleep(0.05)  # 小延遲，方便觀察

                if steps >= max_steps_per_episode:
                    debug_print(f"回合 {episode} 超過最大步數 {max_steps_per_episode}，強制結束。")
                # 保存緩存和移動軌跡
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
        # 每 100 毫秒檢查一次隊列
        root.after(100, process_queue)

    thread = threading.Thread(target=play_episodes, daemon=True)
    thread.start()
    root.after(100, process_queue)
    root.mainloop()

if __name__ == "__main__":
    # 先檢查 GPU
    print("Cupy CUDA Device Count:", cp.cuda.runtime.getDeviceCount())
    cp.show_config()
    # 啟用除錯訊息
    VERBOSE = True
    # 執行 3 局，搜尋深度 2，最大步數 1000
    run_gpu_expectimax(num_episodes=3, max_depth=2, max_steps_per_episode=20000)
