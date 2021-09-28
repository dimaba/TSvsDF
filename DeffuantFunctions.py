import defSim as ds
import networkx as nx
import warnings
import random
import numpy as np

from defSim import NewTiesModifier

from defSim import InfluenceOperator
from defSim.tools.NetworkDistanceUpdater import update_dissimilarity
from defSim.dissimilarity_component.dissimilarity_calculator import DissimilarityCalculator
from typing import List

class InitialExtremistInitializer(ds.RandomContinuousInitializer):
    
    def __init__(self, distribution: str = 'uniform', proportion_extremists = None, uncertainty_extremists = None, uncertainty_moderates = None, **kwargs):

        # uncertainty value defaults half of Deffuant 2006, because opinion scale is also 50% size
    
        self.num_features = 1   

        if proportion_extremists is not None:
            self.proportion_extremists = proportion_extremists
        else:
            warnings.warn("Proportion extremists not specified, using 0.1 as default")
            self.proportion_extremists = 0.1

        if uncertainty_extremists is not None:
            self.uncertainty_extremists = uncertainty_extremists
        else:
            warnings.warn("Uncertainty for extremists not specified, using 0.005 as default")
            self.uncertainty_extremists = 0.005

        if uncertainty_moderates is not None:
            self.uncertainty_moderates = uncertainty_moderates
        else:
            warnings.warn("Uncertainty for extremists not specified, using 0.5 as default")
            self.uncertainty_moderates = 0.5

        self.distribution = distribution

    def initialize_attributes(self, network: nx.Graph, **kwargs):

        for i in range(self.num_features):
            name = 'f' + str("%02d" % (i + 1))
            ds.set_continuous_attribute(network, name, distribution = self.distribution)

            # initially moderate uncertainty for everyone
            for agent in network.nodes:
                network.nodes[agent]['uncertainty'] = self.uncertainty_moderates

            # get list of agents
            agents = network.nodes
            extremists = random.sample(agents, round(self.proportion_extremists * len(agents)))
            extremists_top = extremists[:len(extremists)//2]
            extremists_bottom = extremists[len(extremists)//2:]
            #print(len(extremists_top), " extremists at top, ", len(extremists_bottom), " extremists at bottom")

            for extremist in extremists_top:
                network.nodes[extremist]['f01'] = 1
                network.nodes[extremist]['uncertainty'] = self.uncertainty_extremists

            for extremist in extremists_bottom:
                network.nodes[extremist]['f01'] = 0
                network.nodes[extremist]['uncertainty'] = self.uncertainty_extremists


class RelativeAgreement(InfluenceOperator):
    
    def __init__(self, regime = 'one-to-one', modifiers = [], convergence_rate = 0.3, noise = 0, uncertainty_extremists = None, uncertainty_moderates = None):
        self.regime = regime
        self.modifiers = modifiers
        self.noise = noise
        self.convergence_rate = convergence_rate
        self.uncertainty_extremists = uncertainty_extremists
        self.uncertainty_moderates = uncertainty_moderates

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

        if self.regime != 'one-to-one':
            raise NotImplementedError('Only one-to-one influence is implemented based on DF2006')

        if attributes is None:
            # if no specific attributes were given, take all of them
            attributes = [attribute for attribute in network.nodes[agent_i].keys() if attribute != 'uncertainty']
            #print(attributes)
            
        # variable to return at the end of function
        success = False

        if not all([modifier in ["influence_off", "apply_noise_to_uncertainty", "link_uncertainty", "ts_noise"] for modifier in self.modifiers]):
            warnings.warn("Unrecognized modifier in __class__.")          
                   
        # turn influence off, leaving only noise?
        influence_off = 'influence_off' in self.modifiers       

        apply_noise_to_uncertainty = 'apply_noise_to_uncertainty' in self.modifiers
        link_uncertainty = 'link_uncertainty' in self.modifiers
        ts_noise = 'ts_noise' in self.modifiers
        
        
        # do RA influence
        if type(agents_j) != list:
            agents_j = [agents_j]
            
        if len(agents_j) > 1:
            raise NotImplementedError('Only one-to-one influence is implemented based on DF2006')

        if link_uncertainty:
            # tie uncertainty directly to opinion
            min_uncertainty = 0
            max_uncertainty = 1
            ratio = (max_uncertainty - min_uncertainty) / 0.5
            network.nodes[agent_i]['uncertainty'] = min_uncertainty + ratio * abs(0.5 - network.nodes[agent_i]['f01'])
    
            
        for neighbor in agents_j:

            if link_uncertainty:
                # tie uncertainty directly to opinion
                min_uncertainty = 0.005
                max_uncertainty = 1
                ratio = (max_uncertainty - min_uncertainty) / 0.5
                network.nodes[neighbor]['uncertainty'] = max_uncertainty - ratio * abs(0.5 - network.nodes[neighbor]['f01'])
            #print("opinion i", network.nodes[agent_i]['f01'])
            #print("opinion j", network.nodes[neighbor]['f01'])

            uncertainty_i = network.nodes[agent_i]['uncertainty']
            #print('uncertainty i', uncertainty_i)
            uncertainty_j = network.nodes[neighbor]['uncertainty']
            #print('uncertainty j', uncertainty_j)
            
            if not influence_off:
                
                range_i = [network.nodes[agent_i]['f01'] - uncertainty_i, network.nodes[agent_i]['f01'] + uncertainty_i]
                #print("range i ", range_i)
                range_j = [network.nodes[neighbor]['f01'] - uncertainty_j, network.nodes[neighbor]['f01'] + uncertainty_j]
                #print("range j ", range_j)
                
                overlap = min([range_i[1], range_j[1]]) - max([range_i[0], range_j[0]])
                #print('overlap', overlap)
                non_overlap_i = 2 * uncertainty_i - overlap
                #print('non_overlap_i', non_overlap_i)
                non_overlap_j = 2 * uncertainty_j - overlap
                #print('non_overlap_j', non_overlap_j)            
                agreement_to_change_j = overlap - non_overlap_i
                #print('agreement to change j', agreement_to_change_j)
                agreement_to_change_i = overlap - non_overlap_j
                #print('agreement to change i', agreement_to_change_i)
                
                relative_agreement_to_change_j = agreement_to_change_j / (2 * uncertainty_i)
                #print('relative_agreement_to_change_j', relative_agreement_to_change_j)
                relative_agreement_to_change_i = agreement_to_change_i / (2 * uncertainty_j)
                #print('relative_agreement_to_change_i', relative_agreement_to_change_i)
                
                if overlap > uncertainty_i:
                    #print("j can be influenced")
                    network.nodes[neighbor]['f01'] = network.nodes[neighbor]['f01'] + self.convergence_rate * relative_agreement_to_change_j * (network.nodes[agent_i]['f01'] - network.nodes[neighbor]['f01'])
                    network.nodes[neighbor]['uncertainty'] = network.nodes[neighbor]['uncertainty'] + self.convergence_rate * relative_agreement_to_change_j * (network.nodes[agent_i]['uncertainty'] - network.nodes[neighbor]['uncertainty'])
            
                if overlap > uncertainty_j:
                    #print("i can be influenced")
                    network.nodes[agent_i]['f01'] = network.nodes[agent_i]['f01'] + self.convergence_rate * relative_agreement_to_change_i * (network.nodes[neighbor]['f01'] - network.nodes[agent_i]['f01'])
                    network.nodes[agent_i]['uncertainty'] = network.nodes[agent_i]['uncertainty'] + self.convergence_rate * relative_agreement_to_change_i * (network.nodes[neighbor]['uncertainty'] - network.nodes[agent_i]['uncertainty'])

                #print("new opinion i", network.nodes[agent_i]['f01'])
                #print("new opinion j", network.nodes[neighbor]['f01'])  

                uncertainty_i = network.nodes[agent_i]['uncertainty']
                #print('new uncertainty i', uncertainty_i)
                uncertainty_j = network.nodes[neighbor]['uncertainty']
                #print('new uncertainty j', uncertainty_j)                          
            
            if self.noise > 0:
                noise_prob = 0.1
                noise_int_strength = self.noise * 1000
                if random.random() < noise_prob or ts_noise:
                    if ts_noise:
                        noise_val = np.random.normal(scale = self.noise) # TS18 use normal, with sd to scale noise
                    else:
                        noise_val = random.randrange(-noise_int_strength, +noise_int_strength) / 1000
                    network.nodes[agent_i]['f01'] = network.nodes[agent_i]['f01'] + (noise_val * uncertainty_i)
                    if apply_noise_to_uncertainty:
                        network.nodes[agent_i]['uncertainty'] = network.nodes[agent_i]['uncertainty'] + (noise_val * uncertainty_i)
                if random.random() < noise_prob or ts_noise:
                    if ts_noise:
                        noise_val = np.random.normal(scale = self.noise) # TS18 use normal, with sd to scale noise
                    else:
                        noise_val = random.randrange(-noise_int_strength, +noise_int_strength) / 1000
                    network.nodes[neighbor]['f01'] = network.nodes[neighbor]['f01'] + (noise_val * uncertainty_j)       
                    if apply_noise_to_uncertainty:
                        network.nodes[neighbor]['uncertainty'] = network.nodes[neighbor]['uncertainty'] + (noise_val * uncertainty_j)

            # Bind opinion and uncertainty to range [0,1]
            network.nodes[agent_i]['f01'] = min(1, max(0, network.nodes[agent_i]['f01']))
            network.nodes[neighbor]['f01'] = min(1, max(0, network.nodes[neighbor]['f01']))
            
        success = True
                
        #print("::::::::::::::::END OF INFLUENCE::::::::::::::::")

        return success