import numpy as np
file_path_data = 'C:/Users/robinaki/Documents/NTNU-EEG/src/Data_games/Create.npz'
# test_array = np.random.rand(3, 2)
# test_vector = np.random.rand(4)
# np.savez_compressed(file_path_data, a=test_array, b=test_vector)
loaded = np.load(file_path_data)
# print(loaded)