#from .training import Trainer
from csci545_cs_1 import get_dataloader, save_and_plot_histories
from imports import device, math, np

class FireflyOptimizer:
    def __init__(self, objective_fcn, encoder, data, rbm_param='gauss', pop_test=20, iterations=100, gamma=1, alpha=.02, output_path_base='./'): 
        self.objective_fcn = objective_fcn
        self.rbm = encoder
        self.data = data.to(device)
        self.rbm_param = rbm_param

        self.pop_test = pop_test 
        self.hdfa_iterations = hdfa_iterations
        self.alpha = alpha
        self.gamma = gamma
        self.output_path_base = output_path_base
          
        if rbm_param=='stud_t':
            ### {lr, momentum, sharp_sig, dropout, weight_decay, nu, sigma, scale, batch_size, layer_length}
            self.bounds = [(1e-4, 1e-8), (1e-6, .999), (.1, .999), (1e-6, .9), (1e-6, .9), (1, 9), (.1, 1), (.1, 1), (8, 64), (16, 16)]
        elif rbm_param=='gauss':
            ### {lr, momentum, sharp_sig, dropout, weight_decay, sigma, batch_size, layer_length}
            self.bounds = [(1e-4, 1e-8), (1e-6, .999), (.1, .999), (1e-6, .9), (1e-6, .9), (1, 9), (8, 64), (16, 16)]
        elif rbm_param=='bern':
            ### {lr, momentum, sharp_sig, dropout, weight_decay, batch_size, layer_length}
            self.bounds = [(1e-4, 1e-8), (1e-6, .999), (.1, .999), (1e-6, .9), (1e-6, .9), (8, 64), (16, 16)]

        self.pop_positions = self.initialize_positions('initial')
        self.pop_attractiveness = np.ones(self.pop_test)
        self.pop_fitness = np.zeros(self.pop_test)
        self.pop_alpha = np.ones(self.pop_test)
        self.pop_objectives = [_ for _ in range(self.pop_test)]

        self.firefly_positions_history = {}
        self.fitness_history = {} 
        self.alpha_history = {}
        self.objectives_history = {}
        self.compute_fitness()

    def compute_objective(self, position):
        train_out = 0
        trainer = Trainer(self.data, epochs=1500)
        vis_dim = int(len(next(iter(self.data))))

        if self.rbm_param == 'stud_t':
            self.rbm.reinitialize(vis_dim, int(position[9]), position[2], position[3], position[4], position[5], position[6], position[7])
            dataloader = get_dataloader(self.data, int(position[8]))
            self.rbm, train_out = trainer.contrastive_divergence_training(dataloader, self.rbm, self.rbm_param, epochs= 1500, lr=position[0], lr_momentum=position[1], batch_size=position[8])
        elif self.rbm_param == 'gauss':
            self.rbm.reinitialize(vis_dim, int(position[7]), position[2], position[3], position[4], position[5])
            dataloader = get_dataloader(self.data, int(position[6]))
            self.rbm, train_out = trainer.contrastive_divergence_training(dataloader, self.rbm, self.rbm_param, epochs= 1500, lr=position[0], lr_momentum=position[1], batch_size=position[6])
        elif self.rbm_param == 'bern':
            self.rbm.reinitialize(vis_dim, int(position[6]), position[2], position[3], position[4])
            dataloader = get_dataloader(self.data, int(position[5]))
            self.rbm, train_out = trainer.contrastive_divergence_training(dataloader, self.rbm, self.rbm_param, epochs= 1500, lr=position[0], lr_momentum=position[1], batch_size=position[5])
        
        if isinstance(train_out, dict):
            if self.rbm_param == 'stud_t':
                return self.objective_fcn(self.rbm, self.data, compute_lam=True)
            else:
                return self.objective_fcn(self.rbm, self.data)

        if train_out >= 1e9 or np.isnan(train_out):
            return 1e9, 1e9
        else:
            if self.rbm_param == 'stud_t':
                return self.objective_fcn(self.rbm, self.data, compute_lam=True)
            else:
                return self.objective_fcn(self.rbm, self.data)

    def compute_fitness(self):
        for idx in range(self.pop_test):
            fitness, objectives = self.compute_objective(self.pop_positions[idx])
            self.pop_fitness[idx] = fitness
            self.pop_objectives[idx] = objectives

    def generate_new_position(self):
        new_pos = np.zeros(len(self.bounds))
        if self.rbm_param=='stud_t':
            for idx in [0,1,2,3,4,5,6,7]:
                low, high = self.bounds[idx]
                new_pos[idx] = 10 ** np.random.uniform(np.log10(low), np.log10(high))
            low, high = self.bounds[2]
            new_pos[2] = np.random.uniform(low, high)
            low, high = self.bounds[3]
            new_pos[3] = np.random.uniform(low, high)
            batch_low, batch_high = self.bounds[8]
            layer_low, layer_high = self.bounds[9]
            possible_batches = np.array([2**i for i in range(3, 7)])
            possible_batches = possible_batches[(possible_batches >= batch_low) & (possible_batches <= batch_high)]
            new_pos[8] = int(np.random.choice(possible_batches))
            new_pos[9] = int(np.random.randint(layer_low, layer_high + 1))

        elif self.rbm_param=='gauss':
            for idx in [0,1,2,3,4,5]:
                low, high = self.bounds[idx]
                new_pos[idx] = 10 ** np.random.uniform(np.log10(low), np.log10(high))
            low, high = self.bounds[2]
            new_pos[2] = np.random.uniform(low, high)
            low, high = self.bounds[3]
            new_pos[3] = np.random.uniform(low, high)
            batch_low, batch_high = self.bounds[6]
            layer_low, layer_high = self.bounds[7]
            possible_batches = np.array([2**i for i in range(3, 7)])  # 8,16,32,64
            possible_batches = possible_batches[(possible_batches >= batch_low) & (possible_batches <= batch_high)]
            new_pos[6] = int(np.random.choice(possible_batches))
            new_pos[7] = int(np.random.randint(layer_low, layer_high + 1))

        elif self.rbm_param=='bern':
            for idx in [0,1,2,3,4]:
                low, high = self.bounds[idx]
                new_pos[idx] = 10 ** np.random.uniform(np.log10(low), np.log10(high))
            low, high = self.bounds[2]
            new_pos[2] = np.random.uniform(low, high)
            low, high = self.bounds[3]
            new_pos[3] = np.random.uniform(low, high)
            batch_low, batch_high = self.bounds[5]
            layer_low, layer_high = self.bounds[6]
            possible_batches = np.array([2**i for i in range(3, 7)])
            possible_batches = possible_batches[(possible_batches >= batch_low) & (possible_batches <= batch_high)]
            new_pos[5] = int(np.random.choice(possible_batches))
            new_pos[6] = int(np.random.randint(layer_low, layer_high + 1))
        return new_pos

    def initialize_positions(self, stage='initial'):
        if stage == 'initial':
            return [self.generate_new_position() for _ in range(self.pop_test)]


    def l2_norm(self, ff_idx_1, ff_idx_2):
        diff = self.pop_fitness[ff_idx_1] - self.pop_fitness[ff_idx_2]
        diff = np.clip(diff, -700, 700)
        return 1/(1 + np.exp(diff)+1e-6)

    def compute_attractiveness(self, idx1, idx2):
        norm1 = self.l2_norm(idx1, idx2)
        self.pop_attractiveness[idx1] = (norm1 * self.pop_attractiveness[idx1]) + 1e-6  
        return self.pop_attractiveness[idx1]        

    def compute_positions(self):
        new_positions = self.pop_positions.copy()
        for i in range(self.pop_test):
            for j in range(self.pop_test):
                if i == j:
                    continue
                new_positions[i] = self.update_position(
                    pos1=new_positions[i],
                    pos2=self.pop_positions[j],
                    fitness1=self.pop_fitness[i],
                    fitness2=self.pop_fitness[j],
                    bounds=self.bounds
                )
        self.pop_positions = new_positions

    def update_position(self, pos1, pos2, fitness1, fitness2, bounds, gamma=1.0, alpha_base=0.02, mutation_prob=0.1):
        pos1 = np.array(pos1, dtype=float)
        pos2 = np.array(pos2, dtype=float)
        distance = np.linalg.norm(pos1 - pos2)
        beta = 1 / (1 + np.exp(fitness1 - fitness2) + 1e-6)
        attraction = beta * np.exp(-gamma * distance**2) * (pos2 - pos1)
        alpha = alpha_base * (1 + np.random.randn(*pos1.shape))
        random_term = alpha * (np.random.rand(*pos1.shape) - 0.5)
        new_pos = pos1 + attraction + random_term
        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])
        new_pos = np.clip(new_pos, lower_bounds, upper_bounds)
        if self.rbm_param == 'stud_t':
            discrete_idx = [8, 9]
        elif self.rbm_param == 'gauss':
            discrete_idx = [6, 7]
        elif self.rbm_param == 'bern':
            discrete_idx = [5, 6]
        for idx in discrete_idx:
            new_pos[idx] = int(np.round(new_pos[idx]))
        for i in range(len(new_pos)):
            if np.random.rand() < mutation_prob:
                perturb = (upper_bounds[i] - lower_bounds[i]) * 0.05
                new_pos[i] += np.random.uniform(-perturb, perturb)
                new_pos[i] = np.clip(new_pos[i], lower_bounds[i], upper_bounds[i])
                if i in discrete_idx:
                    new_pos[i] = int(np.round(new_pos[i]))
        return new_pos

    def calculate_distance(self, pos1, pos2):
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def optimize(self):
        print("Beggining Hd-Firefly-SA Optimization")
        last_alpha = float('inf')
        nonincreasing_alpha_counter = 0
        hdfa_ctr = 0
        min_reg_fitness = float('inf')
        new_fitness = float('inf')
        best_position = None
        best_fitness = float('inf')
        avg_fitness = 0
        while hdfa_ctr < self.hdfa_iterations:
            print("Current HdFa Iteration: ", hdfa_ctr)

            self.compute_positions()
            self.compute_fitness()

            avg_fitness = self.pop_fitness[self.pop_fitness != 1e9]
            if avg_fitness.size > 0:
                avg_fitness = np.min(avg_fitness)
            else:
                self.pop_positions = self.initialize_positions('initial')
                continue
                #avg_fitness = 1e9

            for idx in range(self.pop_test):
                new_fitness = self.pop_fitness[idx]
                self.pop_alpha[idx] = np.abs(new_fitness - avg_fitness)

                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_position = self.pop_positions[idx]

            alpha_avg = np.max(self.pop_alpha) - np.min(self.pop_alpha)

            self.firefly_positions_history[hdfa_ctr] = self.pop_positions.copy()
            self.fitness_history[hdfa_ctr] = self.pop_fitness.copy()
            self.alpha_history[hdfa_ctr] = self.pop_alpha.copy()
            self.objectives_history[hdfa_ctr] = self.pop_objectives.copy()
            hdfa_ctr += 1

            print("Current Alpha Average: ", alpha_avg)
            print('HDFA Iteration: ', hdfa_ctr)
            print('Steps Without Increasing Alpha: ', nonincreasing_alpha_counter)

            if last_alpha >= alpha_avg:
                nonincreasing_alpha_counter += 1
            else:
                nonincreasing_alpha_counter = 0
            last_alpha = alpha_avg
            if nonincreasing_alpha_counter == 10 or alpha_avg == 0 or hdfa_ctr == self.hdfa_iterations or math.isnan(nonincreasing_alpha_counter):
                print("Early Convergence")
                break

        print("FF Optimization Complete")
        print("Final Hd-FF Position Estimate: ", best_position, ' with fitness: ', best_fitness)
        save_and_plot_histories(self.firefly_positions_history, self.fitness_history, self.alpha_history, self.objectives_history, self.output_path_base)
        return best_position
