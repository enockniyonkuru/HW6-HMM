# HW6-HMM

In this assignment, you'll implement the Forward and Viterbi Algorithms (dynamic programming). 

## Methods

### Forward Algorithm
The Forward algorithm computes the total probability (likelihood) of an observed sequence given the HMM parameters. It uses a dynamic programming table `alpha[t, j]` representing the probability of observing the first `t` observations and being in hidden state `j` at time `t`.

1. **Initialization**: `alpha[0, j] = prior(j) * emission(j, obs_0)` for each hidden state `j`.
2. **Recursion**: For each subsequent time step `t`, `alpha[t, j] = sum_i(alpha[t-1, i] * transition(i, j)) * emission(j, obs_t)`.
3. **Termination**: The total sequence probability is the sum of `alpha` values across all hidden states at the last time step.

### Viterbi Algorithm
The Viterbi algorithm finds the most likely sequence of hidden states that produced the observed sequence. It is similar to the Forward algorithm but replaces summation with maximization and adds backtracking.

1. **Initialization**: `delta[0, j] = prior(j) * emission(j, obs_0)` for each hidden state `j`.
2. **Recursion**: For each time step `t`, `delta[t, j] = max_i(delta[t-1, i] * transition(i, j)) * emission(j, obs_t)`. A backpointer table records which previous state `i` achieved the maximum.
3. **Termination**: The final state is `argmax_j(delta[T-1, j])`.
4. **Traceback**: Follow backpointers from the final state back to time 0 to recover the best hidden state sequence.

### Edge Cases Handled
- **Empty observation sequence**: Returns 0.0 (forward) or an empty list (viterbi).
- **Unknown observation states**: Raises a `ValueError` if any observation is not in the model's vocabulary.


# Assignment

## Overview 

The goal of this assignment is to implement the Forward and Viterbi Algorithms for Hidden Markov Models (HMMs).

For a helpful refresher on HMMs and the Forward and Viterbi Algorithms you can check out the resources [here](https://web.stanford.edu/~jurafsky/slp3/A.pdf), 
[here](https://towardsdatascience.com/markov-and-hidden-markov-model-3eec42298d75), and [here](https://pieriantraining.com/viterbi-algorithm-implementation-in-python-a-practical-guide/). 





## Tasks and Data 
Please complete the `forward` and `viterbi` functions in the HiddenMarkovModel class. 

We have provided two HMM models (mini_weather_hmm.npz and full_weather_hmm.npz) which explore the relationships between observable weather phenomenon and the temperature outside. Start with the mini_weather_hmm model for testing and debugging. Both include the following arrays:
* `hidden_states`: list of possible hidden states 
* `observation_states`: list of possible observation states 
* `prior_p`: prior probabilities of hidden states (in order given in `hidden_states`) 
* `transition_p`: transition probabilities of hidden states (in order given in `hidden_states`)
* `emission_p`: emission probabilities (`hidden_states` --> `observation_states`)



For both datasets, we also provide input observation sequences and the solution for their best hidden state sequences. 
 * `observation_state_sequence`: observation sequence to test 
* `best_hidden_state_sequence`: correct viterbi hidden state sequence 


Create an HMM class instance for both models and test that your Forward and Viterbi implementation returns the correct probabilities and hidden state sequence for each of the observation sequences.

Within your code, consider the scope of the inputs and how the different parameters of the input data could break the bounds of your implementation.
  * Do your model probabilites add up to the correct values? Is scaling required?
  * How will your model handle zero-probability transitions? 
  * Are the inputs in compatible shapes/sizes which each other? 
  * Any other edge cases you can think of?
  * Ensure that your code accomodates at least 2 possible edge cases. 

Finally, please update your README with a brief description of your methods. 



## Task List

[TODO] Complete the HiddenMarkovModel Class methods  <br>
  [ ] complete the `forward` function in the HiddenMarkovModelClass <br>
  [ ] complete the `viterbi` function in the HiddenMarkovModelClass <br>

[TODO] Unit Testing  <br>
  [ ] Ensure functionality on mini and full weather dataset <br>
  [ ] Account for edge cases 

[TODO] Packaging <br>
  [ ] Update README with description of your methods <br>
  [ ] pip installable module (optional)<br>
  [ ] github actions (install + pytest) (optional)


## Completing the Assignment 
Push your code to GitHub with passing unit tests, and submit a link to your repository [here](https://forms.gle/xw98ZVQjaJvZaAzSA)

### Grading 

* Algorithm implementation (6 points)
    * Forward algorithm is correct (2)
    * Viterbi is correct (2)
    * Output is correct on small weather dataset (1)
    * Output is correct on full weather dataset (1)

* Unit Tests (3 points)
    * Mini model unit test (1)
    * Full model unit test (1)
    * Edge cases (1)

* Style (1 point)
    * Readable code and updated README with a description of your methods 

* Extra credit (0.5 points)
    * Pip installable and Github actions (0.5)
