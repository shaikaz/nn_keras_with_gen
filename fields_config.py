import os
import csv
import numpy as np

'''
    Date configuration, here all the fields should categorized.
    Given table, all the vocabulary functions (except embedding) defined here, and all the function to support the data config are here.
    Note: maybe in the future there is a need the make some changes if some fields need special attention.
'''

path = os.path.dirname(os.path.abspath(__file__)) +'/'

fields = ['vt_id', 'date', 'hour', 'month_of_year', 'aud_id', 'keyword', 'ad_group_name', 'ad_network_type1', 'ad_network_type2',
     'campaign_name', 'Page', 'Slot', 'Position',
     'slug', 'click_type', 'device', 'loc_physical_ms', 'day_of_week', 'quality_score', 'keyword_match_type',
     'description1', 'description2', 'display_url', 'headline', 'Greengeeks_clicks']

data_fields = ['hour', 'month_of_year', 'aud_id', 'keyword', 'ad_group_name', 'ad_network_type1', 'ad_network_type2',
              'campaign_name', 'Page', 'Slot', 'Position',
              'slug', 'click_type', 'device', 'loc_physical_ms', 'day_of_week', 'quality_score', 'keyword_match_type',
              'description1', 'description2', 'display_url', 'headline']

embbeding_fields = ['keyword', 'ad_group_name',
                  'campaign_name', 'slug', 'loc_physical_ms',
                  'description1', 'description2', 'display_url', 'headline']

categorial_fields = ['month_of_year', 'aud_id', 'ad_network_type1', 'ad_network_type2',
                     'Slot', 'hour',
                    'click_type', 'device', 'day_of_week', 'keyword_match_type']

value_fields = ['Page', 'Position', 'quality_score']

location = []
id_list = []
loc_text = ""
with open(path + 'AdWords API Location Criteria 2015-10-13.csv', 'rb') as f:
    data=csv.reader(f)
    for line in data:
        id_list.append(line[0])
        location.append(line[2])
        loc_text += str(line[2]) + " "

def location_name(loc_id):
    temp_index = id_list.index(loc_id)
    loca_name = location[temp_index]
    return loca_name

sample_len_data_dict = {}
def cat_field_voc(field_name, data_dict):
    temp_set = set(data_dict[field_name])
    len_of_voc = len(temp_set)
    sample_len_data_dict[field_name] = len_of_voc
    globals()["voc_dict_{F}".format(F=field_name)] = {}
    for i, value in enumerate(temp_set):
        temp_arr = np.zeros(len_of_voc)
        temp_arr[i] += 1.0
        temp_arr = temp_arr.reshape((1, len_of_voc))
        globals()["voc_dict_{F}".format(F=field_name)][str(value)] = temp_arr

    return globals()["voc_dict_{F}".format(F=field_name)]

def value_field_voc(field_name, data_dict):
    temp_arr = data_dict[field_name]
    globals()["voc_dict_{F}".format(F=field_name)] = {}
    temp_numeric_set = []
    sample_len_data_dict[field_name] = 1
    sum_of_non_valid_values = 0.0
    temp_set = []
    for val in temp_arr:
        try:
            temp_val = float(val)
            if val not in temp_set:
                temp_set.append(val)
                globals()["voc_dict_{F}".format(F=field_name)][val] = np.asarray([temp_val])
            temp_numeric_set.append(temp_val)
        except:
            if val not in temp_set:
                temp_set.append(val)
                globals()["voc_dict_{F}".format(F=field_name)][val] = np.asarray([0.0])
            temp_numeric_set.append(0.0)
            sum_of_non_valid_values += 1.0

    avg_val = float(sum(temp_numeric_set))/float(len(temp_numeric_set) - sum_of_non_valid_values)
    for val in globals()["voc_dict_{F}".format(F=field_name)]:
        if globals()["voc_dict_{F}".format(F=field_name)][val][0] == 0:
            globals()["voc_dict_{F}".format(F=field_name)][val][0] += avg_val

    for i, num in enumerate(temp_numeric_set):
        if num == 0:
            temp_numeric_set[i] += avg_val
    temp_num_set = set(temp_numeric_set)
    '''
        Normalizing the values
    '''
    max_val = max(temp_num_set)
    min_val = min(temp_num_set)
    if max_val == min_val:
        max_val += 0.5
    for val in globals()["voc_dict_{F}".format(F=field_name)]:
        globals()["voc_dict_{F}".format(F=field_name)][val][0] = float((globals()["voc_dict_{F}".format(F=field_name)][val][0] - min_val)) / \
                                                                 float((max_val - min_val))
    return globals()["voc_dict_{F}".format(F=field_name)]

def validation_of_samples_length(data_dict):
    '''
    checks if all samples from the diffrent fields is the same length
    :param data_dict: data dict form
    :return: the length of sample
    '''
    data_len = len(data_dict['vt_id'])
    for field in data_dict:
        if len(data_dict[field]) != data_len:
            raise "Number of samples from field {F} is not the same length of the id's filed.".format(F=field)
    return data_len

def split_data(data_dict, split_fraction = 0.8, shuffle = True):
    '''
    :param data_dict: data in dict form
    :param split_fraction: the fraction of the spliting
    :param shuffle: if to shuffle the data
    :return: two data dicts in the same format
    '''
    data_length = len(data_dict['vt_id'])
    index_data_arr = range(data_length)
    if shuffle: index_data_arr = np.random.permutation(index_data_arr)
    arr_index_split = int(split_fraction*data_length)
    dataset_index_train = list(index_data_arr[:arr_index_split])
    dataset_index_test = list(index_data_arr[arr_index_split:])
    train_dict = {}
    test_dict = {}
    for field in data_dict:
        train_dict[field] = []
        test_dict[field] = []

    for index in dataset_index_train:
        for field in data_dict:
            train_dict[field].append(data_dict[field][index])

    for index in dataset_index_test:
        for field in data_dict:
            test_dict[field].append(data_dict[field][index])

    return train_dict, test_dict

def get_data_train_det_dict(train_test_fraction = 0.8, do_shuffle_on_data_when_split_train_test = True):
    with open(path + 'good_data_final.csv', 'r') as f:
        data_temp = csv.reader(f)
        data = []
        for line in data_temp:
            data.append(line)

        all_fields = {}
        data_dict = {}
        first_row = True
        for line in data:
            if first_row:
                for i, F in enumerate(line):
                    if F != fields[i]:
                        raise "Error, not the same index field for: {Fi}, from the data set, and: {f} from fields array".format(Fi=F, f=fields[i])
                    all_fields[F] = i
                    data_dict[F] = []
                    if F == 'loc_physical_ms':
                        location_index = i
                first_row = False
            else:
                for i, val in enumerate(line):
                    if i != location_index: data_dict[fields[i]].append(val)
                    else: data_dict[fields[i]].append(location_name(val))

        del data
        train_dict, test_dict = split_data(data_dict, split_fraction=train_test_fraction, shuffle=do_shuffle_on_data_when_split_train_test)
    return data_dict, train_dict, test_dict

#def