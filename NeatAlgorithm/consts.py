# No fitness threshold
NEAT_TRAIN_CONFIG_FORMAT = """
#--- parameters for the XOR-2 experiment ---#

[NEAT]
fitness_criterion     = max
fitness_threshold     = inf
pop_size              = {pop_size}
reset_on_extinction   = False

[DefaultGenome]
# node activation options
activation_default      = tanh
activation_mutate_rate  = 0.01
activation_options      = sigmoid tanh sin gauss relu softplus identity clamped inv log exp abs hat square cube 

# node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

# node bias options
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

# connection add/remove rates
conn_add_prob           = {conn_rate}
conn_delete_prob        = {conn_rate}

# connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.05

# node add/remove rates
node_add_prob           = {node_rate}
node_delete_prob        = {node_rate}

feed_forward            = True
initial_connection      = full

# network parameters
num_inputs              = {num_inputs}
num_hidden              = 0
num_outputs             = {num_outputs}

# node response options
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

# connection weight options
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

# genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.4

[DefaultSpeciesSet]
compatibility_threshold = 1.5

[DefaultStagnation]
species_fitness_func = mean
species_elitism      = 1

[DefaultReproduction]
elitism            = 2
survival_threshold = {selection_rate}
"""

