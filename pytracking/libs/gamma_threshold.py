import torch
import numpy as np

def threshold_cost(summary_set, distance_function = torch.norm):
  '''Calculates the threshold as the mean score of each of the samples currently in the summary '''
  length = summary_set.size()[0]
  total_score = 0
  for index_, summary_instance in enumerate(summary_set):
    #set_without_summary_instance = summary_set[np.r_[:index_, (index_ + 1):length], :]
    best_distance = score_observation(summary_set[np.r_[:index_, (index_ + 1):length], :], summary_set[index_])
    total_score = total_score + best_distance

  if total_score == 0 or length == 0:

    print("defaulted to zero")
    final_threshold = 0
  if length == 0:
    print("length of 0")
  else:
    final_threshold = total_score/length
  return final_threshold


def score_observation(data_tensor, observation, distance_function = torch.norm):
  '''Takes the summary set as input along with an observation and returns the
  minimum distance to any point in the summary set '''

  return torch.min(torch.tensor([distance_function(x - observation) for x in data_tensor]))

def update_summary(data_tensor, observation, threshold):
  # determines if it is necessary to add to the summary set, and if it is, it adds it
  score = score_observation(data_tensor, observation)
  if score > threshold:
    #observation = torch.reshape(observation, (1,) + observation.size())
    #updated_data_tensor = torch.cat((data_tensor, observation), dim = 0)

    return 1, score
  return 0, score
