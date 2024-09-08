import itertools
from math import ceil

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
from scipy.interpolate import griddata
import random


def test_data_getter():
    time_delay = []
    window_size = []
    # 数据
    for window in range(10, 110, 10):
        for time in range(-window, 10, (10 + window) // 20):
            window_size.append(window)
            time_delay.append(time)

    validation_data = [0.8120805369127517, 0.7359223300970874, 0.639589905362776, 0.7237318840579711,
                       0.7282809611829945, 0.7824310520939735, 0.8151351351351351,
                       0.7890547263681592, 0.8254963427377221, 0.7172958735733099, 0.7374031007751938,
                       0.5888285492629945, 0.46536523929471035, 0.2986590816741162,
                       0.34221432800473656, 0.18163471241170534, 0.14019562179785747, 0.10193933366484337,
                       0.07937285644292014, 0.07356219349086046, 0.6657894736842105,
                       0.7550607287449392, 0.6153262518968133, 0.7768331562167906, 0.7903055848261328,
                       0.8108720271800679, 0.6977973568281939, 0.8051668460710442, 0.7433102081268583,
                       0.6546938775510204, 0.7306238185255198, 0.7074074074074074, 0.8413793103448276,
                       0.7538910505836576, 0.8133187772925764, 0.7596899224806202, 0.7887887887887888,
                       0.7903066271018794, 0.8010204081632653, 0.6661211129296236,
                       0.6804942630185349, 0.6682242990654206, 0.5294545454545454, 0.3468671679197995, 0.29474216380182,
                       0.20240852246410376, 0.11842105263157894, 0.09292502639915523, 0.08110516934046345,
                       0.08190476190476191, 0.5467464472700074, 0.3815153143471252, 0.5559368565545642,
                       0.6929274843330349, 0.7369970559371933, 0.5479082321187584, 0.7878464818763327,
                       0.6766467065868264,
                       0.7395626242544732, 0.748062015503876, 0.7545717035611165, 0.6843910806174958, 0.867699642431466,
                       0.7671905697445972, 0.8044579533941236, 0.7846790890269151,
                       0.5379202501954652, 0.35437589670014347, 0.13682242990654206, 0.08695652173913043,
                       0.34476843910806176, 0.5438233264320221, 0.6398963730569949, 0.6, 0.5064935064935064,
                       0.6372819100091828, 0.5460251046025104, 0.5748148148148148, 0.5367498314227916,
                       0.658012533572068,
                       0.6619718309859155, 0.7796976241900648, 0.76, 0.6686746987951807,
                       0.8143322475570033, 0.7772126144455748, 0.7624365482233503, 0.6549180327868852,
                       0.8705202312138728,
                       0.8864426419466975, 0.763265306122449, 0.611062335381914, 0.3932835820895522,
                       0.16370269037847698,
                       0.08832644628099173, 0.2180667433831991, 0.47543966040024255, 0.4269081500646831,
                       0.542756183745583,
                       0.44256348246674726, 0.3700750469043152, 0.4062165058949625, 0.6078753076292043,
                       0.6132686084142395, 0.5551626591230552, 0.6779816513761467, 0.6224976167778837,
                       0.6412405699916177,
                       0.7490234375, 0.8108974358974359, 0.7798804780876494, 0.7737296260786194, 0.674908424908425,
                       0.33419689119170987, 0.11418685121107267, 0.016, 0.0, 0.0, 0.48774193548387096,
                       0.3297701763762694,
                       0.5311501597444089, 0.283493369913123, 0.5007012622720898, 0.39692307692307693,
                       0.6717325227963525,
                       0.46426426426426426, 0.5507669831994156, 0.3998993963782696, 0.4420410427066001,
                       0.6848591549295775,
                       0.6876712328767123, 0.8269896193771626, 0.7323420074349443, 0.8155872667398463,
                       0.8058455114822547,
                       0.7623762376237624, 0.3683409436834094, 0.15311653116531165, 0.07524807056229327,
                       0.016346153846153847, 0.02198391420911528, 0.0367170626349892, 0.0223463687150838,
                       0.01907356948228883, 0.5279661016949152, 0.3232222774738936, 0.3801566579634465,
                       0.46424242424242423,
                       0.38578404774823655, 0.5963060686015831, 0.6274509803921569, 0.8094674556213017,
                       0.7566735112936345,
                       0.636290967226219, 0.6576131687242799, 0.744927536231884, 0.7905138339920948, 0.502724795640327,
                       0.14166666666666666, 0.0284789644012945, 0.03147953830010493, 0.029069767441860465,
                       0.029156010230179028, 0.048366013071895426, 0, 0.11434108527131782, 0.03339281162375733,
                       0.1938181818181818, 0.42138009049773756, 0.47319655857048315, 0.32275368797496645,
                       0.7026455026455026,
                       0.6039933444259568, 0.7510917030567685, 0.5763216679076694, 0.7260663507109004,
                       0.8543689320388349,
                       0.82, 0.8464961067853171, 0.823721436343852, 0.3636919315403423, 0.08309037900874636,
                       0.040697674418604654, 0.010610079575596816, 0.0345855694692904, 0.0273768043802887,
                       0.02724935732647815, 0.0, 0, 0.01824817518248175, 0.0907492085824833, 0.1969286359530262,
                       0.4106050305914344, 0.3642703862660944, 0.4915364583333333, 0.6349344978165938,
                       0.8573333333333333,
                       0.5996955859969558, 0.7239776951672863, 0.8132337246531484, 0.6703672075149445,
                       0.1741409435061153,
                       0.029585798816568046, 0.02666666666666667, 0.035621198957428324, 0.01818181818181818,
                       0.034929356357927786, 0.02040816326530612, 0.03403565640194489, 0, 0.075177304964539,
                       0.020066889632107024, 0.09507042253521127, 0.25115781973637336, 0.24106066693451186,
                       0.5098176718092566, 0.5023761031907671, 0.48282694848084545, 0.5687272727272727,
                       0.7676438653637351,
                       0.7828746177370031, 0.7736585365853659, 0.7575462512171373, 0.24434824434824434]

    time_delay = np.array(time_delay)
    window_size = np.array(window_size)
    validation_data = np.array(validation_data)
    # 创建插值网格
    grid_x, grid_y = np.mgrid[time_delay.min():time_delay.max():110j, window_size.min():window_size.max():90j]

    # 插值
    grid_z = griddata((time_delay, window_size), validation_data, (grid_x, grid_y), method='cubic')
    grid_z = np.nan_to_num(grid_z, nan=0.0)
    grid_z[grid_z < 0] = 0

    return grid_z


def get_score(window_size, time_delay):
    scores = test_data_getter()
    x_coor = time_delay + 100
    y_coor = window_size - 10

    return scores[x_coor, y_coor]


def individual_counting(total_population):
    flattened_population = list(itertools.chain(*total_population))
    unique_population = set(flattened_population)

    # 计算不同元组的数量
    unique_count = len(unique_population)
    flattened_population = list(flattened_population)
    return unique_count, flattened_population


class HeuristicGenetic:

    def __init__(self, NUM_EPOCH=20, population_size=50, mutation_rate=0.2, generations=20):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.NUM_EPOCH = NUM_EPOCH

    def run(self):
        window_sizes = []
        delays = []
        for _ in range(self.NUM_EPOCH):
            best_parameters, num_sol = self.genetic_algorithm(self.population_size, self.mutation_rate,
                                                              self.generations)
            window_sizes.append(best_parameters[0])
            delays.append(best_parameters[1])
            print(
                f"最佳参数: height={best_parameters[0]}, width={best_parameters[1]}, score={get_score(*best_parameters)}"
                f"num of solution={num_sol}")

        print(window_sizes, delays)

    def initialize_population(self, size):
        population = []
        for _ in range(size):
            init_win = np.random.randint(10, 100)
            init_delay = np.random.randint(-100, 10)
            population.append((init_win, init_delay))
        return population

    # 计算适应度
    def calculate_fitness(self, population):
        fitness_scores = []
        for individual in population:
            window_size, time_delay = individual
            fitness_scores.append(get_score(window_size, time_delay))
        return fitness_scores

    # 选择
    def selection(self, population, fitness_scores):
        fitness_scores = np.array(fitness_scores)
        probabilities = fitness_scores / np.sum(fitness_scores)
        selected_indices = np.random.choice(len(population), size=len(population), p=probabilities)
        selected_population = [population[i] for i in selected_indices]
        return selected_population

    # 交叉
    def crossover(self, parent1, parent2):
        child1 = (parent1[0], parent2[1])
        child2 = (parent2[0], parent1[1])
        return child1, child2

    # 变异
    def mutate(self, individual, mutation_rate):
        if np.random.rand() < mutation_rate:
            indi_win = np.random.randint(10, 100)
            indi_delay = np.random.randint(-100, 10)
            individual = (indi_win, indi_delay)
        return individual

    def genetic_algorithm(self, population_size, mutation_rate, generations):
        population = self.initialize_population(population_size)
        total_population = []
        for generation in range(generations):
            total_population.append(population)
            fitness_scores = self.calculate_fitness(population)
            population = self.selection(population, fitness_scores)

            new_population = []
            for i in range(0, len(population), 2):
                parent1, parent2 = population[i], population[i + 1]
                child1, child2 = self.crossover(parent1, parent2)
                new_population.extend([self.mutate(child1, mutation_rate),
                                       self.mutate(child2, mutation_rate)])

            population = new_population

        num_sol, sols = individual_counting(total_population)
        best_individual = max(population, key=lambda individual: get_score(*individual))
        print(sols)
        return best_individual, num_sol


class Heuristic_Annel:

    def __init__(self):
        return

    def init_solution(self):
        init_win = np.random.randint(10, 100)
        init_delay = np.random.randint(-100, 10)
        return (init_win, init_delay)

    def generate_new_solution(self, prev_solution):
        prev_win, prev_delay = prev_solution
        new_win = prev_win + np.random.randint(-5, 6)
        new_delay = prev_delay + np.random.randint(-5, 6)

        # 确保新解在给定的范围内
        new_window = np.clip(new_win, 10, 99)
        new_time_delay = np.clip(new_delay, -100, 9)

        new_solution = (new_window, new_time_delay)
        return new_solution

    def simulated_annealing(self, initial_temp, final_temp, alpha, max_iter):
        current_solution = self.init_solution()
        current_score = get_score(*current_solution)
        best_solution = current_solution
        best_score = current_score
        temperature = initial_temp

        sol_counts = []
        while temperature > final_temp:
            for _ in range(max_iter):
                next_solution = self.generate_new_solution(current_solution)
                sol_counts.append([next_solution])
                next_score = get_score(*next_solution)

                delta_score = next_score - current_score

                if delta_score > 0 or np.exp(delta_score / temperature) > np.random.rand():
                    current_solution = next_solution
                    current_score = next_score

                    if current_score > best_score:
                        best_solution = current_solution
                        best_score = current_score

            temperature *= alpha

        num_sol, sols = individual_counting(sol_counts)
        return best_solution, best_score, num_sol, sols

    # 参数设置
    def run(self):
        initial_temp = 1000
        final_temp = 5
        alpha = 0.9
        max_iter = 20
        window_sizes, delays = [], []
        # 运行模拟退火算法
        for _ in range(20):
            best_parameters, best_score, num_sol, sols = self.simulated_annealing(initial_temp, final_temp, alpha, max_iter)
            window_sizes.append(best_parameters[0])
            delays.append(best_parameters[1])
            print(
                f"最佳参数: height={best_parameters[1]}, width={best_parameters[0]}, score={best_score}, number of solution = {num_sol}")
            print(sols)

        print(window_sizes, delays)

class HeuristicAnt:
    def __init__(self):
        self.n_ants = 40  # 蚂蚁的数量
        self.n_iterations = 20  # 迭代次数
        self.rho = 0.5  # 信息素挥发系数
        self.p0 = 0.3
        # Q = 1  # 常数，用于更新信息素
        # 定义搜索空间
        self.window_size_range = np.arange(10, 100)
        self.time_delay_range = np.arange(-100, 10)

        # 初始化
        self.X = np.zeros((self.n_ants, 2), dtype=int)  # 蚂蚁位置矩阵
        self.tau = np.zeros(self.n_ants)  # 每只蚂蚁的适应度值
        self.trace = np.zeros(self.n_iterations)  # 每代蚂蚁的最优适应度
        return

    def run(self):

        total_model = []
        for i in range(self.n_ants):
            self.X[i, 0] = 10 + int((100 - 10) * np.random.rand())
            self.X[i, 1] = -100 + int((10 - (-100)) * np.random.rand())
            self.tau[i] = get_score(self.X[i, 0], self.X[i, 1])

        step = 2  # 局部搜索步长
        P = np.zeros((self.n_iterations, self.n_ants))  # 每代中各只蚂蚁的状态转移概率

        # 初始化信息素矩阵，均匀分布初始信息素

        for NC in range(self.n_iterations):
            lambda_ = 1 / (NC + 1)  # 防止除零
            Tau_best = np.max(self.tau)  # 当前最优适应度
            BestIndex = np.argmax(self.tau)  # 当前最优蚂蚁的索引

            # 计算状态转移概率
            for i in range(self.n_ants):
                P[NC, i] = (Tau_best - self.tau[i]) / Tau_best

            # 更新蚂蚁位置
            for i in range(self.n_ants):
                if P[NC, i] < self.p0:
                    # 局部搜索
                    temp1 = self.X[i, 0] + ceil((2 * np.random.rand() - 1) * step * lambda_)
                    temp2 = self.X[i, 1] + ceil((2 * np.random.rand() - 1) * step * lambda_)
                else:
                    # 全局搜索
                    temp1 = self.X[i, 0] + ceil((100 - 10) * (np.random.rand() - 0.5))
                    temp2 = self.X[i, 1] + ceil((100 - 10) * (np.random.rand() - 0.5))

                # 边界处理
                temp1 = np.clip(temp1, 10, 99)
                temp2 = np.clip(temp2, -100, 9)
                new_temp = (temp1, temp2)
                total_model.append([new_temp])
                # 判断蚂蚁当前位置的状态是否更优
                if get_score(temp1, temp2) < self.tau[i]:
                    self.X[i, 0] = temp1
                    self.X[i, 1] = temp2

            # 更新信息素
            for i in range(self.n_ants):
                self.tau[i] = (1 - self.rho) * self.tau[i] + get_score(self.X[i, 0], self.X[i, 1])

            self.trace[NC] = Tau_best  # 记录当前最优适应度

        # 获取最优解
        maxIndex = np.argmax(self.tau)
        maxX = self.X[maxIndex, 0]
        maxY = self.X[maxIndex, 1]
        maxScore = get_score(maxX, maxY)
        num_sol, sols = individual_counting(total_model)
        print(f"总模型数: {num_sol}, \n{sols}")
        # 打印最优解和最优值
        print(f"最优解 x = {maxX}, y = {maxY}")
        print(f"最优值 = {maxScore}")
        return (maxX, maxY)


class HeuristicTabu:
    def __init__(self):
        self.window_size_range = (10, 99)
        self.time_delay_range = (-100, -9)
        self.max_iterations = 30
        self.tabu_list_size = 20
        self.neighbor_step = 30  # 每次调整的步长
        self.neighbor_len = 20
        self.tabu_list = self.init_tabu()
        return

    def init_solution(self):
        init_win = np.random.randint(*self.window_size_range)
        init_delay = np.random.randint(*self.time_delay_range)
        return (init_win, init_delay)
        # 初始化

    def generate_neighbour(self, prev_solution):
        neighbors = []

        for _ in range(self.neighbor_len):
            prev_win, prev_delay = prev_solution
            new_win = prev_win + np.random.randint(-self.neighbor_step, self.neighbor_step)
            new_delay = prev_delay + np.random.randint(-self.neighbor_step, self.neighbor_step)

            # 确保新解在给定的范围内
            new_window = np.clip(new_win, 10, 99)
            new_time_delay = np.clip(new_delay, -100, 9)

            new_solution = (new_window, new_time_delay)
            if new_solution not in self.tabu_list:
                neighbors.append(new_solution)

        return neighbors

    def init_tabu(self):
        tabu = deque(maxlen=self.tabu_list_size)
        return tabu

    def run(self):
        current_solution = self.init_solution()
        current_score = get_score(*current_solution)
        best_solution = current_solution
        best_score = current_score
        model_count = []

        # 主循环
        for iteration in range(self.max_iterations):
            neighbors = self.generate_neighbour(current_solution)
            # 如果没有可用的邻域解，则跳过
            model_count.append(neighbors)
            if not neighbors:
                continue

            # 评估邻域解，选择得分最高的解
            best_neighbor = max(neighbors, key=lambda x: get_score(*x))
            best_neighbor_score = get_score(*best_neighbor)

            # 更新当前解
            if best_neighbor_score > best_score:
                best_solution = best_neighbor
                best_score = best_neighbor_score

            current_solution = best_neighbor
            self.tabu_list.append(current_solution)

            # print(f"迭代 {iteration + 1}: 当前最优解 {best_solution}，得分 {best_score}")

        num_model, sols = individual_counting(model_count)
        # 最终结果
        print(f"最优解为: {best_solution}，得分为: {best_score}, 模型数：{num_model}")
        print(sols)
        return best_solution


def main():
#     genetic = HeuristicGenetic()
#     genetic.run()

    # simulate_annel = Heuristic_Annel()
    # simulate_annel.run()

    win_list, delay_list = [], []
    ant = HeuristicAnt()
    for _ in range(20):
        win, delay = ant.run()
        win_list.append(win)
        delay_list.append(delay)
    print(win_list, delay_list)

    # win_list, delay_list = [], []
    # for _ in range(20):
    #     tabu = HeuristicTabu()
    #     solu = tabu.run()
    #     win, delay = solu
    #     win_list.append(win)
    #     delay_list.append(delay)

    # print(win_list, delay_list)


if __name__ == "__main__":
    print(get_score(40,-3))