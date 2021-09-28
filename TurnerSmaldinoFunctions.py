import defSim as ds
import networkx as nx
import warnings
import random
import numpy as np

from defSim import NewTiesModifier


# custom initializer in which initial range can be varied
class RandomContinuousScaledInitializer(ds.RandomContinuousInitializer):
    
    def __init__(self, distribution: str = 'uniform', scaling_factor = None, uncertainty_moderates = None, num_features = None, **kwargs):
    
        if num_features is not None:
            self.num_features = num_features
        else:
            warnings.warn("Number of features not specified, using 1 as default")
            self.num_features = 1   

        if scaling_factor is not None:
            self.factor = scaling_factor
        else:
            warnings.warn("Scaling factor not specified, using 1 as default")
            self.factor = 1

        if uncertainty_moderates is not None:
            self.uncertainty_moderates = uncertainty_moderates
        else:
            warnings.warn("Uncertainty not specified, using 1 as default")
            self.uncertainty_moderates = 0.6


        self.distribution = distribution

    def initialize_attributes(self, network: nx.Graph, **kwargs):
           
        offset = 0.5 - (self.factor / 2)

        for agent in network.nodes:
            network.nodes[agent]['uncertainty'] = self.uncertainty_moderates

        for i in range(self.num_features):
            name = 'f' + str("%02d" % (i + 1))
            ds.set_continuous_attribute(network, name, distribution = self.distribution)
            for i in network.nodes():  # iterate over all nodes
                network.nodes[i][name] = network.nodes[i][name] * self.factor + offset  # scale by factor


# custom initializer in which initial range can be varied
class ExtremeScaledInitializer(ds.RandomContinuousInitializer):
    
    def __init__(self, distribution: str = 'uniform', scaling_factor = None, uncertainty_moderates = None, num_features = None, **kwargs):
    
        if num_features is not None:
            self.num_features = num_features
        else:
            warnings.warn("Number of features not specified, using 1 as default")
            self.num_features = 1   

        if scaling_factor is not None:
            self.factor = scaling_factor
        else:
            warnings.warn("Scaling factor not specified, using 1 as default")
            self.factor = 1

        if uncertainty_moderates is not None:
            self.uncertainty_moderates = uncertainty_moderates
        else:
            warnings.warn("Uncertainty not specified, using 1 as default")
            self.uncertainty_moderates = 0.6


        self.distribution = distribution

    def initialize_attributes(self, network: nx.Graph, **kwargs):
           
        offset = 0.5 - (self.factor / 2)

        for agent in network.nodes:
            network.nodes[agent]['uncertainty'] = self.uncertainty_moderates

        for i in range(self.num_features):
            name = 'f' + str("%02d" % (i + 1))
            for i in network.nodes():  # iterate over all nodes
                network.nodes[i][name] = random.choice([0, 1]) * self.factor + offset  # scale by factor



from defSim import InfluenceOperator
from defSim.tools.NetworkDistanceUpdater import update_dissimilarity
from defSim.dissimilarity_component.dissimilarity_calculator import DissimilarityCalculator
from typing import List


class FlacheMacyInfluence(InfluenceOperator):
    
    def __init__(self, regime = 'many-to-one', modifiers = ["smooth"], noise_strength = 0):
        self.regime = regime
        self.modifiers = modifiers
        self.noise_strength = noise_strength

    def spread_influence(self,
                         network: nx.Graph,
                         agent_i: int,
                         agents_j: List[int] or int,
                         regime: str,
                         dissimilarity_measure: DissimilarityCalculator,
                         attributes: List[str] = None,
                         **kwargs) -> bool:
        """

        """

        negative_weights = kwargs.get("allow_negative_weights", False)
        
        if dissimilarity_measure is None:
            dissimilarity_measure = ds.ManhattanDistance(exclude = ['uncertainty'])

        if self.regime not in ['many-to-one', 'one-to-one']:
            raise NotImplementedError('Only many-to-one influence is implemented based on Flache & Macy 2011')

        if attributes is None:
            # if no specific attributes were given, take all of them
            attributes = [feature for feature in list(network.nodes[agent_i].keys()) if feature != "uncertainty"]
            
        # check modifiers
        if type(self.modifiers) != list:
            self.modifiers = [self.modifiers]
        if not all([modifier in ["influence_off", "smooth", "stubborn", "states_and_weights", "all_states", "noise_all_features"] for modifier in self.modifiers]):
            warnings.warn("Unrecognized modifier in __class__.")            

        # variable to return at the end of function
        success = False
        
        # In every step, the selected agent either updates the weights (distances)
        # or gets influenced on a randomly selected state
        if 'states_and_weights' in self.modifiers:
            update_state = True
            update_weights = True
        else:
            update_weights_and_not_state = random.choice([True, False])
            update_weights = update_weights_and_not_state
            update_state = not update_weights_and_not_state
        if 'all_states' in self.modifiers:
            update_all_states = True
        else:
            update_all_states = False     
            
        # turn influence off, leaving only noise?
        influence_off = 'influence_off' in self.modifiers       

        # apply noise to all features together
        noise_all_features = 'noise_all_features' in self.modifiers
        if noise_all_features and not update_all_states:
            warnings.warn("Applying noise to all features, but only one feature updated each step")
        
        if update_weights:
            # NOTE: because defSim assumes undirected networks, and distance is an edge
            # attribute, distances are updated for i --> j and  j --> i simultaneously
            update_dissimilarity(network, [agent_i], dissimilarity_measure)
            #success = True
        
        if update_state:
            if update_all_states:
                influenced_features = attributes
            else:
                influenced_features = [random.choice(attributes)]
            warnings.warn("Influenced features == {}".format(influenced_features))

            if noise_all_features:
                if self.noise_strength > 0: 
                        noise_value = np.random.normal(scale = self.noise_strength) # TS18 use normal, with sd to scale noise
                else:
                    noise_value = 0

            for influenced_feature in influenced_features:
                #print("Influenced feature: {}".format(influenced_feature))
                #print("Base feature value agent i: {}".format(network.nodes[agent_i][influenced_feature]))

                #print("::::::::::::::::START OF INFLUENCE::::::::::::::::")

                # in case of one-to-one, j is only one agent, but we still want to iterate over it
                if type(agents_j) != list:
                    agents_j = [agents_j]
                set_of_influencers = agents_j

                #print("Influence from neighbors: {}".format(set_of_influencers))

                if len(set_of_influencers) != 0:
                    # Weights based on Flache & Macy 2011 Equation 1 / 1a
                    # Based on attribute distance between agents, under the assumption that these
                    # distances are calculated as Manhattan distance scaled by number of features
                    # Both equations are adjusted for the [0,1] opinion scale in defSim, as opposed
                    # to the [-1, 1] opinion scale in Flache & Macy 2011
                    # Maximum distance between agents is 1. Maximally distant agents have weight 0
                    # if negative weights are not allowed, else -1
                    if negative_weights:
                        weights = [1 - (2 * network.edges[agent_i, agent_j]['dist']) for agent_j in set_of_influencers]
                    else:
                        weights = [1 - (network.edges[agent_i, agent_j]['dist']) for agent_j in set_of_influencers]

                    # Raw state change of agent i's opinion on the selected influenced feature
                    # based on weights and differences for each neighbor (Equation 2)
                    feature_differences = [network.nodes[agent_j][influenced_feature] - network.nodes[agent_i][influenced_feature] for agent_j in set_of_influencers]
                    weighted_feature_differences = [weights[i] * feature_differences[i] for i in range(len(set_of_influencers))]
                    overall_state_change = sum(weighted_feature_differences) / (2 * len(set_of_influencers))
                    
                    # Turn influence off completely?
                    if influence_off:
                        overall_state_change = 0  # set influence to 0, so any movement is noise
                    
                    if not noise_all_features:
                        if self.noise_strength > 0: 
                            noise_value = np.random.normal(scale = self.noise_strength) # TS18 use normal, with sd to scale noise
                            #print("noise", noise_value)
                        else:
                            noise_value = 0
                    overall_state_change = overall_state_change + noise_value
                    #print("resulting feature value", network.nodes[agent_i][influenced_feature])

                    ###### TEST SCALING #####
                    overall_state_change = 2 * overall_state_change

                    # Apply smoothing as desired
                    if 'stubborn' in self.modifiers:
                        overall_state_change = self._apply_stubbornness(overall_state_change, target = agent_i, influenced_feature = influenced_feature, network = network)
                    if 'smooth' in self.modifiers or not any([adjust_function in self.modifiers for adjust_function in ['smooth', 'stubborn']]):
                        overall_state_change = self._apply_smoothing(overall_state_change, target = agent_i, influenced_feature = influenced_feature, network = network)            

                    # Adjusted state of agent i's opinion on the selected influence feature
                    network.nodes[agent_i][influenced_feature] = network.nodes[agent_i][influenced_feature] + overall_state_change
                    
                    # Bind to opinion range [0,1]
                    network.nodes[agent_i][influenced_feature] = min(1, max(0, network.nodes[agent_i][influenced_feature]))

                success = True
                
        #print("::::::::::::::::END OF INFLUENCE::::::::::::::::")

        return success
    
    def _apply_smoothing(self, overall_state_change, target, influenced_feature, network):
        # based on Equation 2a with Corrigendum 2017
        if overall_state_change > 0:
            overall_state_change = (overall_state_change * (1 - network.nodes[target][influenced_feature]))
        else:
            overall_state_change = (overall_state_change * (0 + network.nodes[target][influenced_feature]))
        return overall_state_change
    
    def _apply_stubbornness(self, overall_state_change, target, influenced_feature, network):
        # based on Equation 2a WITHOUT Corrigendum 2017
        if network.nodes[target][influenced_feature] > 0.5:
            overall_state_change = (overall_state_change * (1 - network.nodes[target][influenced_feature]))
        else:
            overall_state_change = (overall_state_change * (0 + network.nodes[target][influenced_feature]))
        return overall_state_change                


# Implementing an output reporter which calculates polarization at the end of each run
## Output reporters are implemented based on the OutputTableCreator base class https://github.com/marijnkeijzer/defSim/blob/master/defSim/tools/CreateOutputTable.py
## Every output reporter should implement a static method called 'create_output', which calculates and returns the desired output
## Give the reporter a label if you want to customize the column name in the output
## Examples: https://github.com/marijnkeijzer/defSim/blob/master/defSim/tools/OutputMeasures.py

class PolarizationReporter(ds.tools.CreateOutputTable.OutputTableCreator):
    
    label = "Polarization"
    
    @staticmethod
    def create_output(network: nx.Graph, distance = "Manhattan", **kwargs):
        """
        Will calculate polarization for a given network.
        Polarization is based on the observed variance in opinion distances. 
        Opinion distances are based on a euclidean distance across all features of the agents.
        We recalculate these distances here because during the simulation these distances are only calculated 
        between directly connected agents. 
        
        :param network: A networkx graph on which to calculate polarization
        :param distance: Whether to use distance calculation from original study ("FlacheMacy") or 
        euclidean distance ("euclidean")
        :return: A single value for polarization, between 0 and 1.
        """
        
        distances = []

        agents = list(network.nodes())

        exclude = kwargs.get('exclude_polarization', ['uncertainty'])
        
        # for each agent calculate distances to other agents
        # omit last agent, since it has no new agents to compare with
        for agent1_id in agents[:-1]:
            agent1_attributes = np.array([v for k, v in network.nodes[agent1_id].items() if k not in exclude])
            
            # each combination of agents should be calculated once, 
            # so omit agents which come before agent1 in the list
            # self-distances are also omitted
            for agent2_id in agents[agents.index(agent1_id + 1):]:
                agent2_attributes = np.array([v for k, v in network.nodes[agent2_id].items() if k not in exclude])
    
                # calculate average distance across all features
                
                distance_pairs = abs(agent2_attributes - agent1_attributes)

                distances.append(sum(distance_pairs) / len(distance_pairs))
        
        # return variance of distances
        return np.var(distances)


class SimulationWithNewTies(ds.Simulation):
    
    def run_simulation(self, initialize: bool = True):
        """
        This method initializes the network if none is given, initializes the attributes of the agents, and also
        computes and sets the distances between each neighbor.
        It then calls different functions that execute the simulation based on which stop criterion was selected.
        
        :param bool=True initialize: Initialize the simulation before running (disable if initialization was
            done separately)
        :returns: A Pandas DataFrame that contains one row of data. To see what output the output contains see
            :func:`~create_output_table`
        """
        if initialize:
            self.initialize_simulation()
        
        if self.influence_function == 'list':
            self.influence_function = self.parameter_dict['influence_function']
            
        if 'initialize_uncertainty' in self.parameter_dict:
            if self.parameter_dict['initialize_uncertainty']:
                for agent in self.network.nodes:
                    self.network.nodes[agent]['uncertainty'] = 1 - abs(self.network.nodes[agent]['f01'] - 0.5)
        
        if all([ties_parameter in self.parameter_dict.keys() for ties_parameter in ['new_ties_probability', 'new_ties_in_iteration']]):
            
            new_ties_probability = self.parameter_dict['new_ties_probability']
            new_ties_in_iteration = self.parameter_dict['new_ties_in_iteration']

            for iteration in range(new_ties_in_iteration):
                self.run_simulation_step()
                
            NewTiesModifier(new_ties_probability = new_ties_probability).rewire_network(self.network)
            self.dissimilarity_calculator.calculate_dissimilarity_networkwide(self.network)

        if self.stop_condition == "pragmatic_convergence":
            self._run_until_pragmatic_convergence()
        elif self.stop_condition == "strict_convergence":
            self._run_until_strict_convergence()

        elif self.stop_condition == "max_iteration":
            self.max_iterations = self.max_iterations - new_ties_in_iteration
            self._run_until_max_iteration()
        else:
            raise ValueError("Can only select from the options ['pragmatic_convergence', 'strict_convergence', 'max_iteration']")

        return self.create_output_table()               