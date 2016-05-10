import numpy as np
from fields_config import categorial_fields, embbeding_fields, value_fields, sample_len_data_dict

'''
    Generator, Class witch fits a generator function given the requested arguments.
    There are 2 generators functions, depends on the batch_size the right generator function chosen.
'''
#TODO repeat vector should be in the net, or should be given already as an input from here?

class Generator():
    def __init__(self, data_dict, length_of_the_data, batch_size=1, shuffle_per_epoch=True, voc_dict={}):
        self.data_dict = data_dict
        self.length_of_the_data = length_of_the_data
        self.shuffle_per_epoch = shuffle_per_epoch
        self.batch_size = batch_size
        self.voc_dict = voc_dict
        # if self.batch_size == 1:
        #     #self.generator = self.generator_func_for_batch_size_1()
        #     self.generator = self.generator_func_for_batch_size_grater_then_1()
        # else:
        #     self.generator = self.generator_func_for_batch_size_grater_then_1()
        self.generator = self.generator_func_for_batch_size_grater_then_1()

    def generator_func_for_batch_size_1(self):
        while 1:
            temp_index_arr = range(0, self.length_of_the_data)
            if self.shuffle_per_epoch: temp_index_arr = np.random.permutation(temp_index_arr)
            for i in temp_index_arr:
                temp_batch_sample = {}
                for field in self.data_dict:
                    if (field in categorial_fields) or (field in value_fields):
                        temp_batch_sample["input_{F}".format(F=field)] = self.voc_dict[field][self.data_dict[field][i]]
                    elif field in embbeding_fields:
                        temp_arr_sample = self.voc_dict[field].texts_to_matrix([self.data_dict[field][i]])[0]
                        temp_arr_sample = temp_arr_sample.reshape((1, sample_len_data_dict[field]))
                        temp_batch_sample["input_{F}".format(F=field)] = temp_arr_sample
                    elif field == 'Greengeeks_clicks':
                        output_dict = {}
                        if self.data_dict[field][i] == '1':
                            output_dict["output"] = np.asarray([1.0])
                        else:
                            output_dict["output"] = np.asarray([0.0])
                yield (temp_batch_sample, output_dict)

    def generator_func_for_batch_size_grater_then_1(self):
        while 1:
            temp_index_arr = range(0, self.length_of_the_data)
            if self.shuffle_per_epoch: temp_index_arr = np.random.permutation(temp_index_arr)
            for i in range(0, len(temp_index_arr), self.batch_size):
                temp_batch_sample = {}
                output_batch_dict = {'output': []}
                for field in self.data_dict:
                    temp_batch_sample["input_{F}".format(F=field)] = []
                for j in temp_index_arr[i:min(i + self.batch_size, len(temp_index_arr))]:
                    for field in self.data_dict:
                        if (field in categorial_fields) or (field in value_fields):
                            temp_arr_sample = self.voc_dict[field][self.data_dict[field][j]]
                            temp_arr_sample = np.array(temp_arr_sample, np.float32).reshape(sample_len_data_dict[field])
                            temp_batch_sample["input_{F}".format(F=field)].append(temp_arr_sample)
                        elif field in embbeding_fields:
                            temp_arr_sample = self.voc_dict[field].texts_to_matrix([self.data_dict[field][j]])[0]
                            temp_arr_sample = np.array(temp_arr_sample, np.float32).reshape(sample_len_data_dict[field])
                            temp_batch_sample["input_{F}".format(F=field)].append(temp_arr_sample)
                        elif field == 'Greengeeks_clicks':
                            if self.data_dict[field][j] == '1':
                                output_batch_dict["output"].append(np.asarray([1.0]))
                            else:
                                output_batch_dict["output"].append(np.asarray([0.0]))
                for f in temp_batch_sample:
                    temp_batch_sample[f] = np.asarray(temp_batch_sample[f])
                for f in output_batch_dict:
                    output_batch_dict[f] = np.asarray(output_batch_dict[f])
                yield (temp_batch_sample, output_batch_dict)



