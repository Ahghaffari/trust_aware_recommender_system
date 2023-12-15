from surprise.prediction_algorithms.knns import SymmetricAlgo
import heapq

import numpy as np

import pandas as pd

from surprise.prediction_algorithms.algo_base import AlgoBase

from surprise.prediction_algorithms.predictions import PredictionImpossible

class CollabTrustRecSys(SymmetricAlgo):

    def __init__(self, k=40, min_k=1, sim_options={}, model_type="collaborative", verbose=True, **kwargs):

        SymmetricAlgo.__init__(self, sim_options=sim_options, verbose=verbose, **kwargs)
        self.k = k
        self.min_k = min_k
        self.trust_mat = None
        self.model_type = model_type

    def calculate_weights_sim_trust(self):
        self.mat_weight = np.zeros_like(self.sim)
        
        non_zero_condition = ((self.sim + self.trust_mat) != 0) & ((self.trust_mat*self.sim) != 0)
        
        similarity_non_zero_condition = (self.sim != 0) & (self.trust_mat == 0)
        
        trust_non_zero_condition = (self.sim == 0) & (self.trust_mat != 0)
        
        self.mat_weight[non_zero_condition] = (2 * self.sim[non_zero_condition] * self.trust_mat[non_zero_condition]) / (self.sim[non_zero_condition] + self.trust_mat[non_zero_condition])
        
        self.mat_weight[similarity_non_zero_condition] = self.sim[similarity_non_zero_condition]
        self.mat_weight[trust_non_zero_condition] = self.trust_mat[trust_non_zero_condition]
        

    def fit(self, trainset, trust_data=None, nu_u=1508):
        if self.model_type == "collaborative" or self.model_type == "1":
            SymmetricAlgo.fit(self, trainset)
            self.mat_weight = self.compute_similarities()

        elif self.model_type == "trust_aware" or self.model_type == "2":
            SymmetricAlgo.fit(self, trainset)
            self.trust_mat = np.zeros((nu_u, nu_u))
            for _, row in trust_data.iterrows():
                try:
                    self.trust_mat[row["trustor"], row["trustee"]] = row["trust_value"]
                except IndexError:
                    pass
            self.mat_weight = self.trust_mat

        elif self.model_type == "hybrid_collaborative_trust" or self.model_type == "3":
            SymmetricAlgo.fit(self, trainset)
            self.sim = self.compute_similarities()
            self.trust_mat = np.zeros((nu_u, nu_u))
            for _, row in trust_data.iterrows():
                try:
                    self.trust_mat[row["trustor"], row["trustee"]] = row["trust_value"]
                except IndexError:
                    pass

            self.calculate_weights_sim_trust()

        return self


    def estimate(self, u, i):
        if self.model_type == "collaborative" or self.model_type == "trust_aware" or self.model_type == "1" or self.model_type == "2":
            if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
                raise PredictionImpossible("User and/or item is unknown.")

            x, y = self.switch(u, i)

            neighbors = [(self.mat_weight[x, x2], r) for (x2, r) in self.yr[y]]
            k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])

            # compute weighted average
            sum_sim = sum_ratings = actual_k = 0
            for (sim, r) in k_neighbors:
                if sim > 0:
                    sum_sim += sim
                    sum_ratings += sim * r
                    actual_k += 1

            if actual_k < self.min_k:
                raise PredictionImpossible("Not enough neighbors.")

            est = sum_ratings / sum_sim

            details = {"actual_k": actual_k}
            return est, details
    
        elif self.model_type == "hybrid_collaborative_trust" or self.model_type == "3":
            if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
                raise PredictionImpossible("User and/or item is unknown.")

            x, y = self.switch(u, i)

            neighbors = [(self.mat_weight[x, x2], r) for (x2, r) in self.yr[y]]
            k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])

            # compute weighted average
            sum_sim = sum_ratings = actual_k = 0
            for (sim, r) in k_neighbors:
                if sim > 0:
                    sum_sim += sim
                    sum_ratings += sim * r
                    actual_k += 1

            if actual_k < self.min_k:
                raise PredictionImpossible("Not enough neighbors.")

            est = sum_ratings / sum_sim

            details = {"actual_k": actual_k}
            return est, details
    