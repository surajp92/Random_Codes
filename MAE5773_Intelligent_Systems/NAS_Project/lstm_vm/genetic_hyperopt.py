import numpy as np
from sklearn.model_selection import KFold


class GeneticHyperopt:
    def __init__(self, learner, X, y, fitness_func, maximize, num_folds=3,
                 pop_size=10, num_gen=2, elite_percent=0.2, competitiveness=0.4, mutation_prob=0.2):
        self.learner = learner
        self.X = X
        self.y = y
        self.num_folds = num_folds
        self.indices = [(train_ind, val_ind) for (train_ind, val_ind) in KFold(n_splits=self.num_folds).split(X, y)]
        self.fitness_func = fitness_func
        self.maximize = maximize
        self.pop_size = pop_size
        self.num_gen = num_gen
        self.num_elites = int(pop_size * elite_percent)  # check if even
        self.tournament_size = int(pop_size * competitiveness)
        self.mutation_prob = mutation_prob
        self.params = []

    def add_param(self, param):
        self.params.append(param)
        return self

    def _initialize_population(self):
        return [[param.sample() for param in self.params] for _ in range(self.pop_size)]

    def _to_param_dict(self, ind_params):
        param_dict = {}
        for i in range(len(self.params)):
            param_dict[self.params[i].name] = ind_params[i]
        return param_dict
    
    def _coeff_determination(self,y_true, y_pred):
        SS_res =  np.sum(np.square(y_true - y_pred))
        SS_tot = np.sum(np.square(y_true - np.mean(y_true)))
        return ( 1 - SS_res/(SS_tot + 1.0e-7) )

    def _evaluate_individual(self, ind_params):
#        print("Evaluating the individual...")
        param_dict = self._to_param_dict(ind_params)
        learner_obj = self.learner(**param_dict)
        
#        print(type(learner_obj))
        
        score = 0
        for train_ind, val_ind in self.indices:
            learner_obj.fit(self.X[train_ind, :], self.y[train_ind], epochs=5, batch_size=128, verbose=0)
            score += self.fitness_func(learner_obj.predict(self.X[val_ind, :]), self.y[val_ind])
#            score += self._coeff_determination(learner_obj.predict(self.X[val_ind, :]), self.y[val_ind])

        return score / self.num_folds

    def _evaluate_population(self):
        return [self._evaluate_individual(ind) for ind in self.population]

    def _select_parents(self):
        parents = [None] * (self.pop_size - self.num_elites)
        for i in range(self.pop_size - self.num_elites):
            candidates = np.random.choice(np.arange(self.pop_size), self.tournament_size, replace=False)
            parents[i] = self.population[min(candidates)][:]
        return parents

    def _generate_children(self, parents):
        children = [None] * len(parents)
        for i in range(int(len(parents) / 2)):
            child1 = parents[2 * i]
            child2 = parents[2 * i + 1]
            for j in range(len(self.params)):
                if np.random.rand() > 0.5:
                    temp = child1[j]
                    child1[j] = child2[j]
                    child2[j] = temp
            children[2 * i] = child1
            children[2 * i + 1] = child2
        return children

    def _mutate(self, children):
        for i in range(len(children)):
            child = children[i][:]
            for j in range(len(self.params)):
                if np.random.rand() < self.mutation_prob:
                    child[j] = self.params[j].mutate(child[j])
            children[i] = child
        return children

    def evolve(self):
        self.population = self._initialize_population()
        
        plotting_stats = np.zeros((self.num_gen,self.pop_size+1))
        best_param_dict = {}
        
        for i in range(self.num_gen):
            # rank the population
            print("Generation", i)
            print("Calculating fitness...")
            fitness = self._evaluate_population()
            if self.maximize:
                fitness *= -1
            rank = np.argsort(fitness)
            self.population = [self.population[r] for r in rank]
            
            plotting_stats[i,0] = i
            plotting_stats[i,1:] = np.array(fitness)

            print("Best individual:", self._to_param_dict(self.population[0]))
            print("Best score:", np.min(fitness))
            print("Worst score:", np.max(fitness))
            print("Population mean:", np.mean(fitness))
            

            best_param_dict[i] = self._to_param_dict(self.population[0])
            
            # generate new generation
            print("Generating children...")
            parents = self._select_parents()
            children = self._generate_children(parents)
            children = self._mutate(children)
            self.population[self.num_elites:] = children

            print("---")

        return self._to_param_dict(self.population[0]), min(fitness), plotting_stats, best_param_dict
