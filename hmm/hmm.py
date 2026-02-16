import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        This function runs the forward algorithm on an input sequence of observation states.

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        
        # Edge case: empty observation sequence
        if len(input_observation_states) == 0:
            return 0.0
        
        # Edge case: observation state not in model
        for obs in input_observation_states:
            if obs not in self.observation_states_dict:
                raise ValueError(f"Observation '{obs}' not found in observation states.")

        num_hidden = len(self.hidden_states)
        T = len(input_observation_states)

        # Step 1. Initialize forward table (alpha)
        # alpha[t, j] = P(o_1, ..., o_t, q_t = j | model)
        alpha = np.zeros((T, num_hidden))

        # Initialize first time step: alpha[0, j] = prior(j) * emission(j, obs_0)
        first_obs_idx = self.observation_states_dict[input_observation_states[0]]
        for j in range(num_hidden):
            alpha[0, j] = self.prior_p[j] * self.emission_p[j, first_obs_idx]

        # Step 2. Recursion: calculate probabilities for t = 1, ..., T-1
        for t in range(1, T):
            obs_idx = self.observation_states_dict[input_observation_states[t]]
            for j in range(num_hidden):
                # Sum over all previous hidden states
                alpha[t, j] = np.sum(alpha[t - 1, :] * self.transition_p[:, j]) * self.emission_p[j, obs_idx]

        # Step 3. Return final probability: sum over all states at last time step
        forward_probability = np.sum(alpha[-1, :])
        return forward_probability


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        This function runs the viterbi algorithm on an input sequence of observation states.

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        
        # Edge case: empty observation sequence
        if len(decode_observation_states) == 0:
            return []

        # Edge case: observation state not in model
        for obs in decode_observation_states:
            if obs not in self.observation_states_dict:
                raise ValueError(f"Observation '{obs}' not found in observation states.")

        num_hidden = len(self.hidden_states)
        T = len(decode_observation_states)

        # Step 1. Initialize variables
        # viterbi_table[t, j] = max probability of any path ending in state j at time t
        viterbi_table = np.zeros((T, num_hidden))
        # backpointer[t, j] = the state at t-1 that led to the best path ending at state j at time t
        backpointer = np.zeros((T, num_hidden), dtype=int)

        # Initialize first time step
        first_obs_idx = self.observation_states_dict[decode_observation_states[0]]
        for j in range(num_hidden):
            viterbi_table[0, j] = self.prior_p[j] * self.emission_p[j, first_obs_idx]
            backpointer[0, j] = 0  # no previous state

        # Step 2. Recursion: calculate probabilities for t = 1, ..., T-1
        for t in range(1, T):
            obs_idx = self.observation_states_dict[decode_observation_states[t]]
            for j in range(num_hidden):
                # Find the max over all previous states
                trans_probs = viterbi_table[t - 1, :] * self.transition_p[:, j]
                best_prev = np.argmax(trans_probs)
                viterbi_table[t, j] = trans_probs[best_prev] * self.emission_p[j, obs_idx]
                backpointer[t, j] = best_prev

        # Step 3. Traceback
        best_path = np.zeros(T, dtype=int)
        best_path[T - 1] = np.argmax(viterbi_table[T - 1, :])

        for t in range(T - 2, -1, -1):
            best_path[t] = backpointer[t + 1, best_path[t + 1]]

        # Step 4. Return best hidden state sequence (convert indices to state names)
        best_hidden_state_sequence = [self.hidden_states_dict[idx] for idx in best_path]
        return best_hidden_state_sequence