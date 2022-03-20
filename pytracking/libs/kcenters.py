import torch
import numpy as np

class IndexedTensor:

    def __init__(self, tensor, original_index):
        self.tensor = tensor
        self.original_index = original_index

    def get_tensor(self):
        return self.tensor
    def get_index(self):
        return self.original_index

def l2_dist(A, B):
    return torch.norm(A-B)

def l2_normalised_dist(A, B):
    return torch.norm(A/torch.norm(A) - B/torch.norm(B))

def cosine_dist(A, B):
    return 1-torch.dot(A.flatten(), B.flatten())/(torch.norm(A) * torch.norm(B))

def Tensor_to_IndexedTensor(input_tensor):
  IndexedTensorList = list()
  for index, individual_tensor in enumerate(input_tensor):
    IndexedTensorList.append(IndexedTensor(individual_tensor, index))
  return IndexedTensorList

def distance_cost(summary_set, observation, distance_function = l2_dist):
  '''This function takes as input a summary set and a new observation and returns the distance and
  index of the item in the summary set that is the closet item to the new observation'''

  '''curr_min_distance = 10000000
  best_index = 0

   for index_, summary_instance in enumerate(summary_set):
     vector_diff = distance_function(observation - summary_instance)
     if vector_diff < curr_min_distance:
       curr_min_distance = vector_diff
       best_index = index_

   return curr_min_distance, best_index '''

  distances = torch.tensor([distance_function(observation,x) for x in summary_set])
  return torch.min(distances), torch.argmin(distances)



def return_furthest_point(summary_set, observation_set, distance_function = l2_dist):
  '''This function works in more or less an offline fashion. It takes as input an entire summary
  set and the entire set of observations and returns the index of the observation set that is
  farthest away from samples in the current summary '''
  furthest_min_distance = 0
  best_index = 0
  for index, observation in enumerate(observation_set):
    obs_min_dist, _ = distance_cost(summary_set, observation, distance_function)
    if obs_min_dist > furthest_min_distance:
      furthest_min_distance = obs_min_dist
      best_index = index

  return furthest_min_distance, best_index

def score_observation(summary_set, observation, distance_function = l2_dist):
  '''Takes the summary set as input along with an observation and returns the
  minimum distance to any point in the summary set '''
  '''
  min_distance = 10000000
  for summary_instance in summary_set:
    distance = distance_function(summary_instance - observation)
    if distance < min_distance:
      min_distance = distance

  return min_distance
  '''
  return torch.min(torch.tensor([distance_function(x, observation) for x in summary_set]))

def threshold_cost(summary_set, distance_function = l2_dist):
  '''Calculates the threshold as the mean score of each of the samples currently in the summary '''
  length = summary_set.size()[0]
  total_score = 0
  for index_, summary_instance in enumerate(summary_set):
    #set_without_summary_instance = summary_set[np.r_[:index_, (index_ + 1):length], :]
    best_distance = score_observation(summary_set[np.r_[:index_, (index_ + 1):length], :], summary_set[index_], distance_function=distance_function)
    total_score = total_score + best_distance

  if total_score == 0 or length == 0:

    print("defaulted to zero")
    final_threshold = 0
  if length == 0:
    print("length of 0")
  else:
    final_threshold = total_score/length
  return final_threshold

def extremumsummary(observation_set, num_observations, distance_function=l2_dist):
  '''Computes a summary as a subset of input samples (observationconvert tensor to tuple_set) by greedily picking the samples
  farthest away from sapmles in the current summary. k is the desired number of samples in the summary'''
  # initialize the summary set with a random index:
  length = observation_set.size()[0]
  #rand_index = np.random.randint(length)
  rand_index = length - 1
  summary = torch.reshape(observation_set[rand_index], (1,) + tuple(torch.tensor(observation_set.size()[1:]).numpy()))
  observation_set = observation_set[np.r_[:rand_index, (rand_index + 1):length], ]
  # summary = observation_set[0]
  # observation_set = observation_set[1:, :]
  while summary.size()[0] < num_observations:
    #index = argmax_i min_j d(Zi, Sj)
    #choose samples to add to summary by finding samples farthest away from samples in current summary
    _, best_index = return_furthest_point(summary, observation_set, distance_function=distance_function)
    summary = torch.cat((summary, torch.reshape(observation_set[best_index], (1,) + tuple(torch.tensor(observation_set.size()[1:]).numpy()))), dim = 0)
    observation_set = observation_set[np.r_[:best_index, (best_index + 1):observation_set.size()[0]], ]

  return summary

def online_summary_update(summary_set, observation, k):
  '''Performs an online summary update of a summary set given a new observation where k is the maximum
  desired size of the summary to be created '''
  threshold = threshold_cost(summary_set)
  if score_observation(summary_set, observation) > threshold:
    #add to summary set
    summary_set = torch.cat((summary_set, observation), dim = 0)
  if summary_set.size()[0] > k:
    summary_set = extremumsummary(summary_set, k)
  return summary_set

def k_online_summary_update_index(summary_set, observation, k, threshold = None, indices_to_keep = None):
    """
    NOT DONE
    summary_set torch tensor, first dim is number of elements
    k is the maximum summary size
    """
    raise NotImplementedError

    summary_size = summary_set.size()[0]

    # Initialize a threshold cost if none-existent
    if threshold is None and summary_size > 1:
        threshold = threshold_cost(summary_set)

    # Added without replacement
    if score_observation(summary_set, observation) > threshold and summary_size < k:
        return summary_size, threshold

    # Added with replacement
    # todo

    # Not added
    return -1, threshold

def online_summary_update_index_extremum(summary_set, observation, k, threshold = None, distance_function=l2_dist):
    '''Returns the index of the entry in the original summary set to be replaced when
    the summary size == k upon addition of the new observation '''
    length = summary_set.size()[0]

    # If the set is empty, add the observation
    if length is 0:
        return length, threshold

    if threshold is None and length > 1:
        threshold = threshold_cost(summary_set)
        #print(f"threshold is {threshold}")
    #threshold = threshold_cost(summary_set)

    score = score_observation(summary_set, observation, distance_function=distance_function)

    #print(f"Score, Threshold: {score} | {threshold}")

    # New observation does not meet threshold, do not add
    if score <= threshold:
        return -1, threshold

    # New observation meets threshold, set is not max size, simply add it
    if score > threshold and length < k:
        return length, threshold

    # New observation meets threshold, but set is full, need to prune and run extremum summary
    dummy_summary_set = [x for x in torch.cat((summary_set, observation), dim = 0)]
    dummy_summary_set = Tensor_to_IndexedTensor(dummy_summary_set)
    if len(dummy_summary_set) > k:
      #we are getting problems because of something in extremum_summary
      indices_in_new_summary = extremumsummary_indexes(dummy_summary_set, k, distance_function=distance_function)
      indices_before = set(range(k))
      index_to_remove = indices_before - indices_in_new_summary

      return index_to_remove.pop(), threshold #, dummy_summary_set

    return -1, threshold

def extremumsummary_indexes(observation_set, num_observations, distance_function=l2_dist):
  '''Computes a summary as a subset of input samples by greedily picking the samples
  farthest away from sapmles in the current summary. k is the desired number of samples in the summary. observation_set is assumed to be
  a list of Indexed Tensors'''

  # initialize the summary set with a random index:
  length = len(observation_set)
  individ_obs_size =  tuple(torch.tensor(observation_set[0].get_tensor().size()).numpy())

  #don't use a random index. instead, use the newly added item.
  #rand_index = np.random.randint(length)

  #don't use random index. use the first image!
  rand_index = length - 1
  #rand_index = 0

  #reshape the item at the random index to have a shape of (1,512,22,22), etc.
  summary = torch.reshape(observation_set[rand_index].get_tensor(), (1,) + individ_obs_size)

  summary_indices = set()
  summary_indices.add(observation_set[rand_index].get_index())
  #observation_set = observation_set[np.r_[:rand_index, (rand_index + 1):length], ]
  # don't delete, instead just
  observation_set = observation_set[:rand_index] + observation_set[(rand_index + 1):]
  #del observation_set[rand_index]

  # summary = observation_set[0]
  # observation_set = observation_set[1:, :]

  while summary.size()[0] < num_observations:
    #choose samples to add to summary by finding samples farthest away from samples in current summary
    _, best_index, original_index = return_furthest_point_index(summary, observation_set, distance_function=distance_function) ## here we would like to ensure that we return the original index as well as current index
    summary = torch.cat((summary, torch.reshape(observation_set[best_index].get_tensor(), (1,) + individ_obs_size)), dim = 0)
    #observation_set = observation_set[np.r_[:best_index, (best_index + 1):observation_set.size()[0]], ]
    observation_set = observation_set[:best_index] + observation_set[(best_index + 1):]
    summary_indices.add(original_index)

  return summary_indices

def return_furthest_point_index(summary_set, observation_set: IndexedTensor, distance_function = l2_dist):
  '''This function works in more or less an offline fashion. It takes as input an entire summary
  set and the entire set of observations and returns the index of the observation set that is
  farthest away from samples in the current summary. Takes an IndexedTensor as input '''
  # this function leaves the summary_set and observation_set untouched
  #furthest_min_distance = 0
  #best_index = 0
  #original_index = 0
  #import pdb; pdb.set_trace()
  #get index of best_index
  distances = torch.tensor([distance_cost(summary_set, observation.get_tensor(), distance_function)[0] for observation in observation_set])
  #pdb.set_trace()
  furthest_min_distance = torch.argmax(distances)
  best_index = torch.argmin(distances)
  original_index = observation_set[best_index].get_index()
  '''
  best_index = torch.argmin(torch.tensor([distance_cost(summary_set, observation.get_tensor(), distance_function) for observation in observation_set]))


  for index, observation in enumerate(observation_set):
    obs_min_dist, _ = distance_cost(summary_set, observation.get_tensor(), distance_function)
    if obs_min_dist > furthest_min_distance:
      furthest_min_distance = obs_min_dist
      best_index = index
      original_index = observation.get_index()
      '''
  return furthest_min_distance, best_index, original_index


def calc_pairwise_distances(input_tensor, distance_function = l2_dist):
    #this should return a matrix with values in the upper top left. probably could be improved but whatever
    length = input_tensor.size()[0]
    distance_matrix = torch.zeros((length+1),(length+1))
    for row in list(range(length)):
        for col in list(range(length - row)):
            distance_matrix[row,col] = distance_function(input_tensor[row], input_tensor[col])
    return distance_matrix
